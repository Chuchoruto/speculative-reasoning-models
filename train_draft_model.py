"""
Training script for draft model.
Uses mixed loss: KL divergence for logits + Cosine similarity for latent thoughts.
Includes WandB logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from draft_model import DraftModel
from draft_dataset import DraftDataset, DraftCollator
from tqdm import tqdm
import argparse
import yaml
import os
import wandb


def compute_loss(
    student_logits,  # [batch, seq, vocab]
    teacher_logits_list,  # List of [num_targets, vocab] tensors (one per sample)
    student_latent_thoughts_list,  # List of [768] tensors
    teacher_latent_thoughts_list,  # List of [num_latent, 768] tensors
    target_positions_list,  # List of position lists
    kl_weight=1.0,
    cosine_weight=1.0,
    temperature=1.0,
):
    """
    Compute mixed loss: KL divergence for logits + Cosine similarity for latent thoughts.
    
    Args:
        student_logits: Student model logits [batch, seq, vocab]
        teacher_logits_list: Teacher logits per sample
        student_latent_thoughts_list: Student latent thoughts per sample
        teacher_latent_thoughts_list: Teacher latent thoughts per sample
        target_positions_list: Target positions per sample
        kl_weight: Weight for KL divergence loss
        cosine_weight: Weight for cosine similarity loss (1 - cosine_sim)
        temperature: Temperature for softmax in KL divergence
    """
    batch_size = student_logits.shape[0]
    device = student_logits.device
    
    total_kl_loss = 0.0
    total_cosine_loss = 0.0
    num_kl_samples = 0
    num_cosine_samples = 0
    
    # Process each sample in the batch
    for batch_idx in range(batch_size):
        # Get teacher logits for this sample
        teacher_logits_sample = teacher_logits_list[batch_idx]  # [num_targets, vocab]
        target_positions_sample = target_positions_list[batch_idx]  # List of positions
        
        # Get corresponding student logits at target positions
        # Student logits at position i predict token at position i+1
        # So for target position pos, we use logits at pos-1
        student_logits_sample_list = []
        for pos in target_positions_sample:
            if pos > 0:  # Can't predict from position 0
                logit_pos = pos - 1
                if logit_pos < student_logits.shape[1]:
                    student_logits_sample_list.append(student_logits[batch_idx, logit_pos, :])
        
        if len(student_logits_sample_list) == 0:
            continue
        
        student_logits_sample = torch.stack(student_logits_sample_list)  # [num_targets, vocab]
        
        # Align with teacher logits (take minimum length)
        min_len = min(len(student_logits_sample), len(teacher_logits_sample))
        if min_len == 0:
            continue
        
        student_logits_aligned = student_logits_sample[:min_len]  # [min_len, vocab]
        teacher_logits_aligned = teacher_logits_sample[:min_len]  # [min_len, vocab]
        
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits_aligned / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits_aligned / temperature, dim=-1)
        
        # KL divergence: KL(student || teacher)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        )
        
        total_kl_loss += kl_loss
        num_kl_samples += 1
        
        # Cosine similarity loss for latent thoughts
        if (batch_idx < len(student_latent_thoughts_list) and 
            batch_idx < len(teacher_latent_thoughts_list)):
            
            student_latents = student_latent_thoughts_list[batch_idx]  # List of [hidden_dim] tensors
            teacher_latents = teacher_latent_thoughts_list[batch_idx]  # [num_latent, hidden_dim]
            
            student_count = len(student_latents) if student_latents else 0
            teacher_count = len(teacher_latents) if teacher_latents is not None else 0
            
            if student_count > 0 and teacher_count > 0:
                # Stack student latent thoughts
                try:
                    student_latents_stacked = torch.stack([
                        thought.to(teacher_latents.device) 
                        for thought in student_latents
                    ])  # [num_student_latents, hidden_dim]
                    
                    # Ensure shapes match
                    min_latent_len = min(len(student_latents_stacked), len(teacher_latents))
                    if min_latent_len > 0:
                        student_aligned = student_latents_stacked[:min_latent_len]  # [min_len, hidden_dim]
                        teacher_aligned = teacher_latents[:min_latent_len]  # [min_len, hidden_dim]
                        
                        # Cosine similarity loss: 1 - cosine_similarity (minimize distance, maximize similarity)
                        # Compute cosine similarity per latent thought
                        cosine_sims = F.cosine_similarity(
                            student_aligned, 
                            teacher_aligned, 
                            dim=-1
                        )  # [min_len] - one similarity per latent thought
                        
                        # Loss is 1 - cosine_similarity (ranges from 0 to 2, 0 when identical)
                        cosine_loss = (1.0 - cosine_sims).mean()
                        total_cosine_loss += cosine_loss
                        num_cosine_samples += 1
                except Exception as e:
                    # Skip if stacking fails (e.g., empty list or shape mismatch)
                    pass
    
    # Average losses
    avg_kl_loss = total_kl_loss / num_kl_samples if num_kl_samples > 0 else torch.tensor(0.0, device=device)
    avg_cosine_loss = total_cosine_loss / num_cosine_samples if num_cosine_samples > 0 else torch.tensor(0.0, device=device)
    
    # Combined loss (only KL + Cosine)
    total_loss = (
        kl_weight * avg_kl_loss +
        cosine_weight * avg_cosine_loss
    )
    
    return {
        'total_loss': total_loss,
        'kl_loss': avg_kl_loss,
        'cosine_loss': avg_cosine_loss,
        'num_kl_samples': num_kl_samples,
        'num_cosine_samples': num_cosine_samples,
    }


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    kl_weight=1.0,
    cosine_weight=1.0,
    temperature=1.0,
    wandb_run=None,
    global_step=0,
    rank=0,
    world_size=1,
    gradient_accumulation_steps=1,
    lr_scheduler=None,
    current_epoch=0,
):
    """Train for one epoch with gradient accumulation."""
    model.train()
    
    # Use gradient accumulation only after epoch 4 (0-indexed: epochs 5+)
    # Before epoch 4, use gradient_accumulation_steps=1 for richer updates
    if current_epoch < 4:
        effective_grad_accum_steps = 1
    else:
        effective_grad_accum_steps = gradient_accumulation_steps
    
    if rank == 0 and current_epoch == 4:
        print(f"Switching to gradient accumulation (steps={effective_grad_accum_steps}) starting from epoch 5")
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_cosine = 0.0
    num_batches = 0
    
    # Only show progress bar on rank 0
    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))
    
    # Initialize gradients at the start of the epoch
    optimizer.zero_grad()
    
    # We'll accumulate gradients and update periodically
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['position_ids'].to(device)
        
        # Move target data to device (lists of tensors)
        teacher_logits = [logits.to(device) for logits in batch['target_logits']]
        teacher_latent_thoughts = [thoughts.to(device) for thoughts in batch['latent_thoughts']]
        
        # Forward pass - use known latent positions from dataset
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            collect_latent_thoughts=True,
            latent_positions=batch['latent_positions'],  # Pass known positions
        )
        
        # Move student latent thoughts to device (already organized by sample)
        student_latent_thoughts_by_sample = [
            [thought.to(device) for thought in sample_latents]
            for sample_latents in outputs.latent_thoughts
        ] if outputs.latent_thoughts else [[] for _ in range(input_ids.shape[0])]
        
        # Debug: Check how many samples have latent thoughts
        if batch_idx == 0 and rank == 0:
            num_samples_with_student_latents = sum(1 for latents in student_latent_thoughts_by_sample if len(latents) > 0)
            num_samples_with_teacher_latents = sum(1 for latents in teacher_latent_thoughts if latents is not None and len(latents) > 0)
            print(f"Debug batch {batch_idx}: {num_samples_with_student_latents}/{len(student_latent_thoughts_by_sample)} samples have student latents")
            print(f"Debug batch {batch_idx}: {num_samples_with_teacher_latents}/{len(teacher_latent_thoughts)} samples have teacher latents")
        
        # Compute loss
        loss_dict = compute_loss(
            outputs.logits,
            teacher_logits,
            student_latent_thoughts_by_sample,
            teacher_latent_thoughts,
            batch['target_positions'],
            kl_weight=kl_weight,
            cosine_weight=cosine_weight,
            temperature=temperature,
        )
        
        loss = loss_dict['total_loss']
        
        # Scale loss by accumulation steps (to average over accumulated batches)
        loss = loss / effective_grad_accum_steps
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        # Update metrics (use unscaled loss for logging)
        unscaled_loss = loss.item() * effective_grad_accum_steps
        total_loss += unscaled_loss
        total_kl += loss_dict['kl_loss'].item()
        total_cosine += loss_dict['cosine_loss'].item()
        num_batches += 1
        
        # Update weights every effective_grad_accum_steps batches
        if (batch_idx + 1) % effective_grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate scheduler after optimizer step (per step, not per epoch)
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # Zero gradients after update (not before each batch)
            optimizer.zero_grad()
            
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': unscaled_loss,
                'kl': loss_dict['kl_loss'].item(),
                'cos': loss_dict['cosine_loss'].item(),
                'step': global_step,
            })
            
            # Log to wandb
            if wandb_run is not None:
                wandb_run.log({
                    'train/loss': unscaled_loss,
                    'train/kl_loss': loss_dict['kl_loss'].item(),
                    'train/cosine_loss': loss_dict['cosine_loss'].item(),
                    'train/num_kl_samples': loss_dict['num_kl_samples'],
                    'train/num_cosine_samples': loss_dict['num_cosine_samples'],
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/step': global_step,
                })
        else:
            # Just update progress bar without logging to wandb
            pbar.set_postfix({
                'loss': unscaled_loss,
                'kl': loss_dict['kl_loss'].item(),
                'cos': loss_dict['cosine_loss'].item(),
                'acc': f'{(batch_idx + 1) % effective_grad_accum_steps}/{effective_grad_accum_steps}',
            })
    
    # Handle remaining gradients if num_batches is not divisible by effective_grad_accum_steps
    if num_batches % effective_grad_accum_steps != 0:
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate scheduler after optimizer step
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Zero gradients
        optimizer.zero_grad()
        
        global_step += 1
    
    return {
        'avg_loss': total_loss / num_batches,
        'avg_kl': total_kl / num_batches,
        'avg_cosine': total_cosine / num_batches,
    }, global_step


def validate_epoch(
    model,
    dataloader,
    device,
    tokenizer,
    kl_weight=1.0,
    cosine_weight=1.0,
    temperature=1.0,
    wandb_run=None,
    rank=0,
    world_size=1,
    num_samples_to_show=3,
):
    """Validate for one epoch (no gradient updates) with exact match accuracy."""
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    total_cosine = 0.0
    num_batches = 0
    
    # For exact match accuracy
    exact_matches = 0
    total_token_predictions = 0
    
    # Sample outputs for display
    sample_outputs = []
    samples_collected = 0
    
    with torch.no_grad():
        # Only show progress bar on rank 0
        pbar = tqdm(dataloader, desc="Validation", disable=(rank != 0))
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = batch['position_ids'].to(device)
            
            # Move target data to device (lists of tensors)
            teacher_logits = [logits.to(device) for logits in batch['target_logits']]
            teacher_latent_thoughts = [thoughts.to(device) for thoughts in batch['latent_thoughts']]
            target_tokens = [tokens.to(device) for tokens in batch['target_tokens']]
            
            # Forward pass - use known latent positions from dataset
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                collect_latent_thoughts=True,
                latent_positions=batch['latent_positions'],  # Pass known positions
            )
            
            # Move student latent thoughts to device (already organized by sample)
            student_latent_thoughts_by_sample = [
                [thought.to(device) for thought in sample_latents]
                for sample_latents in outputs.latent_thoughts
            ] if outputs.latent_thoughts else [[] for _ in range(input_ids.shape[0])]
            
            # Compute loss
            loss_dict = compute_loss(
                outputs.logits,
                teacher_logits,
                student_latent_thoughts_by_sample,
                teacher_latent_thoughts,
                batch['target_positions'],
                kl_weight=kl_weight,
                cosine_weight=cosine_weight,
                temperature=temperature,
            )
            
            loss = loss_dict['total_loss']
            
            # Update metrics
            total_loss += loss.item()
            total_kl += loss_dict['kl_loss'].item()
            total_cosine += loss_dict['cosine_loss'].item()
            num_batches += 1
            
            # Compute exact match accuracy
            batch_size = input_ids.shape[0]
            for sample_idx in range(batch_size):
                target_tokens_sample = target_tokens[sample_idx]  # [num_target_tokens]
                target_positions_sample = batch['target_positions'][sample_idx]  # List of positions
                
                # Get predicted tokens from student logits at target positions
                predicted_tokens = []
                for i, pos in enumerate(target_positions_sample):
                    if pos > 0 and pos - 1 < outputs.logits.shape[1]:
                        logit_pos = pos - 1
                        logits_at_pos = outputs.logits[sample_idx, logit_pos, :]  # [vocab_size]
                        predicted_token = torch.argmax(logits_at_pos, dim=-1).item()
                        predicted_tokens.append(predicted_token)
                
                # Compare predicted vs target tokens
                min_len = min(len(predicted_tokens), len(target_tokens_sample))
                if min_len > 0:
                    predicted_tokens_aligned = predicted_tokens[:min_len]
                    target_tokens_aligned = target_tokens_sample[:min_len].cpu().tolist()
                    
                    # Exact match: all tokens must match
                    is_exact_match = predicted_tokens_aligned == target_tokens_aligned
                    if is_exact_match:
                        exact_matches += 1
                    
                    total_token_predictions += 1
                    
                    # Collect sample outputs for display
                    if samples_collected < num_samples_to_show and rank == 0:
                        # Get input text (remove padding and special tokens for readability)
                        input_sample = input_ids[sample_idx].cpu().tolist()
                        # Remove padding
                        input_sample = [t for t in input_sample if t != tokenizer.pad_token_id]
                        
                        try:
                            input_text = tokenizer.decode(input_sample, skip_special_tokens=False)
                            predicted_text = tokenizer.decode(predicted_tokens_aligned, skip_special_tokens=False)
                            target_text = tokenizer.decode(target_tokens_aligned, skip_special_tokens=False)
                            
                            sample_outputs.append({
                                'sample_idx': samples_collected,
                                'input': input_text[:200] + "..." if len(input_text) > 200 else input_text,
                                'predicted_tokens': predicted_tokens_aligned,
                                'predicted_text': predicted_text,
                                'target_tokens': target_tokens_aligned,
                                'target_text': target_text,
                                'exact_match': is_exact_match,
                            })
                            samples_collected += 1
                        except Exception as e:
                            # Skip if decoding fails
                            pass
            
            # Update progress bar
            current_accuracy = exact_matches / total_token_predictions if total_token_predictions > 0 else 0.0
            pbar.set_postfix({
                'loss': loss.item(),
                'kl': loss_dict['kl_loss'].item(),
                'cos': loss_dict['cosine_loss'].item(),
                'acc': f'{current_accuracy:.2%}',
            })
    
    # Calculate final accuracy
    exact_match_accuracy = exact_matches / total_token_predictions if total_token_predictions > 0 else 0.0
    
    # Print sample outputs on rank 0
    if rank == 0 and len(sample_outputs) > 0:
        print("\n" + "="*80)
        print("VALIDATION SAMPLE OUTPUTS")
        print("="*80)
        for sample in sample_outputs:
            print(f"\nSample {sample['sample_idx'] + 1}:")
            print(f"  Input (first 200 chars): {sample['input']}")
            print(f"  Target tokens: {sample['target_tokens'][:20]}..." if len(sample['target_tokens']) > 20 else f"  Target tokens: {sample['target_tokens']}")
            print(f"  Predicted tokens: {sample['predicted_tokens'][:20]}..." if len(sample['predicted_tokens']) > 20 else f"  Predicted tokens: {sample['predicted_tokens']}")
            print(f"  Target text: {sample['target_text'][:100]}..." if len(sample['target_text']) > 100 else f"  Target text: {sample['target_text']}")
            print(f"  Predicted text: {sample['predicted_text'][:100]}..." if len(sample['predicted_text']) > 100 else f"  Predicted text: {sample['predicted_text']}")
            print(f"  Exact Match: {'✓' if sample['exact_match'] else '✗'}")
        print("="*80 + "\n")
    
    return {
        'avg_loss': total_loss / num_batches,
        'avg_kl': total_kl / num_batches,
        'avg_cosine': total_cosine / num_batches,
        'exact_match_accuracy': exact_match_accuracy,
        'exact_matches': exact_matches,
        'total_predictions': total_token_predictions,
    }


def main():
    # Initialize distributed training if using torchrun
    use_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        if rank == 0:
            print(f"Initialized distributed training with {world_size} GPUs")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using single device: {device}")
    
    parser = argparse.ArgumentParser(description="Train draft model")
    parser.add_argument("config_file", help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb only on rank 0
    wandb_run = None
    if config.get('use_wandb', True) and rank == 0:
        wandb.init(
            project=config.get('wandb_project', 'draft-model-training'),
            name=config.get('wandb_run_name', None),
            config=config,
        )
        wandb_run = wandb.run
    
    # Load tokenizer and model (only print on rank 0)
    if rank == 0:
        print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Coconut tokens
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(config['model_id'])
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Get hidden dimensions
    hidden_dim_base = base_model.config.n_embd  # GPT2 hidden dim
    hidden_dim_target = config.get('teacher_hidden_dim', 768)  # Coconut hidden dim
    
    if rank == 0:
        print(f"Base model hidden dim: {hidden_dim_base}")
        print(f"Target (teacher) hidden dim: {hidden_dim_target}")
    
    # Wrap in DraftModel
    model = DraftModel(
        base_model,
        latent_id,
        start_id,
        end_id,
        tokenizer.eos_token_id,
        hidden_dim_base=hidden_dim_base,
        hidden_dim_target=hidden_dim_target,
    ).to(device)
    
    # Wrap model with DDP for distributed training
    if use_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Load training dataset
    if rank == 0:
        print(f"Loading training dataset from {config['data_json_path']}...")
    train_dataset = DraftDataset(
        json_path=config['data_json_path'],
        data_dir=config.get('data_dir', None),
    )
    
    collator = DraftCollator(pad_token_id=tokenizer.pad_token_id, latent_token_id=latent_id)
    
    # Create distributed sampler if using distributed training
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        shuffle = True
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config.get('num_workers', 4),
        collate_fn=collator,
    )
    
    # Load validation dataset if provided
    val_dataloader = None
    val_sampler = None
    if config.get('val_json_path'):
        if rank == 0:
            print(f"Loading validation dataset from {config['val_json_path']}...")
        val_dataset = DraftDataset(
            json_path=config['val_json_path'],
            data_dir=config.get('val_data_dir', config.get('data_dir', None)),
        )
        
        # Validation sampler (no shuffle, no distributed needed but can use it for consistency)
        if use_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,  # No shuffle for validation
            )
        else:
            val_sampler = None
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.get('num_workers', 4),
            collate_fn=collator,
        )
    
    # Setup optimizer
    initial_lr = config.get('lr', 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=config.get('weight_decay', 0.01),
    )
    
    # Training loop
    num_epochs = config.get('num_epochs', 10)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    warmup_steps = config.get('warmup_steps', 20)  # Default to 20 steps
    peak_lr = config.get('peak_lr', 4e-4)  # Cap LR after warmup
    
    # Setup learning rate schedulers
    # Combined scheduler: Linear warmup + exponential decay per step
    from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
    
    # Exponential decay factor per step
    # Tie gamma to weight_decay argument: gamma = 1 - weight_decay
    # e.g., weight_decay=0.02 -> gamma=0.98 per step
    wd_value = config.get('weight_decay', 0.01)
    decay_gamma_per_step = max(0.0, min(1.0, 1.0 - wd_value))
    
    def lr_lambda(step):
        # Scale so warmup tops at peak_lr (not initial_lr)
        peak_scale = (peak_lr / initial_lr) if initial_lr > 0 else 1.0
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup to peak_lr
            return (step / warmup_steps) * peak_scale
        else:
            # After warmup: exponential decay per step from peak
            steps_after_warmup = max(0, step - warmup_steps)
            return peak_scale * (decay_gamma_per_step ** steps_after_warmup)
    
    # Combined warmup + exponential decay scheduler (applied per step)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Additional scheduler: Reduce on plateau for CE loss (optional, for extra safety)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Reduce LR by half when plateau detected
        patience=2,  # Wait 2 epochs without improvement before reducing
        verbose=True,  # Print LR reduction messages
        min_lr=1e-6,  # Minimum learning rate
        threshold=0.03,  # Require at least 3% relative improvement to count as progress
        threshold_mode='rel',  # Relative threshold mode
    )
    
    kl_weight = config.get('kl_weight', 1.0)
    cosine_weight = config.get('cosine_weight', 1.0)
    temperature = config.get('temperature', 1.0)
    
    if rank == 0:
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Loss weights: KL={kl_weight}, Cosine={cosine_weight}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Warmup peak LR: {peak_lr}")
        print(f"Exponential decay per step (gamma = 1 - weight_decay): gamma={decay_gamma_per_step} ({(1-decay_gamma_per_step)*100:.1f}% reduction per step)")
        print(f"Initial learning rate: {initial_lr}")
    
    global_step = 0
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        if use_distributed:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_metrics, global_step = train_epoch(
            model,
            train_dataloader,
            optimizer,
            device,
            kl_weight=kl_weight,
            cosine_weight=cosine_weight,
            temperature=temperature,
            wandb_run=wandb_run,
            global_step=global_step,
            rank=rank,
            world_size=world_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler=lr_scheduler,
            current_epoch=epoch,
        )
        
        if rank == 0:
            print(f"Epoch {epoch + 1} - Train Loss: {train_metrics['avg_loss']:.4f}, "
                  f"KL: {train_metrics['avg_kl']:.4f}, Cosine: {train_metrics['avg_cosine']:.4f}")
        
        # Validation
        val_metrics = None
        if val_dataloader is not None:
            if rank == 0:
                print("Running validation...")
            val_metrics = validate_epoch(
                model,
                val_dataloader,
                device,
                tokenizer,
                kl_weight=kl_weight,
                cosine_weight=cosine_weight,
                temperature=temperature,
                wandb_run=wandb_run,
                rank=rank,
                world_size=world_size,
                num_samples_to_show=3,
            )
            
            if rank == 0:
                print(f"Epoch {epoch + 1} - Val Loss: {val_metrics['avg_loss']:.4f}, "
                      f"KL: {val_metrics['avg_kl']:.4f}, Cosine: {val_metrics['avg_cosine']:.4f}")
                print(f"Epoch {epoch + 1} - Val Exact Match Accuracy: {val_metrics['exact_match_accuracy']:.2%} "
                      f"({val_metrics['exact_matches']}/{val_metrics['total_predictions']})")
        
        # Log epoch metrics to wandb (only on rank 0)
        if wandb_run is not None:
            log_dict = {
                'epoch': epoch + 1,
                'epoch/train_loss': train_metrics['avg_loss'],
                'epoch/train_kl_loss': train_metrics['avg_kl'],
                'epoch/train_cosine_loss': train_metrics['avg_cosine'],
            }
            if val_metrics is not None:
                log_dict.update({
                    'epoch/val_loss': val_metrics['avg_loss'],
                    'epoch/val_kl_loss': val_metrics['avg_kl'],
                    'epoch/val_cosine_loss': val_metrics['avg_cosine'],
                    'epoch/val_exact_match_accuracy': val_metrics['exact_match_accuracy'],
                })
            wandb_run.log(log_dict)
        
        # Update learning rate schedulers
        # Only on rank 0 to avoid multiple print statements
        if rank == 0:
            # Note: lr_scheduler is already stepped per update in train_epoch
            # Here we only step the plateau scheduler (per epoch)
            
            # Use validation KL loss if available, otherwise training KL loss
            kl_loss_for_scheduler = val_metrics['avg_kl'] if val_metrics is not None else train_metrics['avg_kl']
            plateau_scheduler.step(kl_loss_for_scheduler)
            
            current_lr = optimizer.param_groups[0]['lr']
            if wandb_run is not None:
                wandb_run.log({'train/learning_rate': current_lr})
        
        
        # Save checkpoint (only on rank 0)
        if config.get('save_path') and rank == 0:
            os.makedirs(config['save_path'], exist_ok=True)
            # Get model state dict (unwrap DDP if needed)
            model_state = model.module.state_dict() if use_distributed else model.state_dict()
            
            checkpoint_path = os.path.join(
                config['save_path'],
                f"draft_model_epoch_{epoch + 1}.pt"
            )
            # Save both training and validation metrics
            checkpoint_metrics = {
                'train_metrics': train_metrics,
            }
            if val_metrics is not None:
                checkpoint_metrics['val_metrics'] = val_metrics
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': checkpoint_metrics,
                'config': config,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Synchronize all processes before next epoch
        if use_distributed:
            dist.barrier()
    
    if wandb_run is not None:
        wandb_run.finish()
    
    if rank == 0:
        print("\nTraining completed!")
    
    # Clean up distributed training
    if use_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

