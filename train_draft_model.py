"""
Training script for draft model.
Uses mixed loss: KL divergence for logits + MSE for latent thoughts.
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
    target_tokens_list,  # List of [num_targets] tensors
    target_positions_list,  # List of position lists
    ce_weight=1.0,
    kl_weight=1.0,
    cosine_weight=1.0,
    temperature=1.0,
):
    """
    Compute mixed loss: Causal LM (CE) + KL divergence for logits + Cosine similarity for latent thoughts.
    
    Args:
        student_logits: Student model logits [batch, seq, vocab]
        teacher_logits_list: Teacher logits per sample
        student_latent_thoughts_list: Student latent thoughts per sample
        teacher_latent_thoughts_list: Teacher latent thoughts per sample
        target_tokens_list: Target token IDs per sample
        target_positions_list: Target positions per sample
        ce_weight: Weight for causal LM (CrossEntropy) loss
        kl_weight: Weight for KL divergence loss
        cosine_weight: Weight for cosine similarity loss (1 - cosine_sim)
        temperature: Temperature for softmax in KL divergence
    """
    batch_size = student_logits.shape[0]
    device = student_logits.device
    
    # For causal LM loss: collect all logits and labels
    ce_logits_list = []
    ce_labels_list = []
    
    total_kl_loss = 0.0
    total_cosine_loss = 0.0
    num_kl_samples = 0
    num_cosine_samples = 0
    
    # Process each sample in the batch
    for batch_idx in range(batch_size):
        # Get teacher logits and target tokens for this sample
        teacher_logits_sample = teacher_logits_list[batch_idx]  # [num_targets, vocab]
        target_tokens_sample = target_tokens_list[batch_idx]  # [num_targets]
        target_positions_sample = target_positions_list[batch_idx]  # List of positions
        
        # Get corresponding student logits at target positions
        # Student logits at position i predict token at position i+1
        # So for target position pos, we use logits at pos-1
        student_logits_sample_list = []
        target_tokens_aligned = []
        for i, pos in enumerate(target_positions_sample):
            if pos > 0:  # Can't predict from position 0
                logit_pos = pos - 1
                if logit_pos < student_logits.shape[1]:
                    student_logits_sample_list.append(student_logits[batch_idx, logit_pos, :])
                    # Get corresponding target token
                    if i < len(target_tokens_sample):
                        target_tokens_aligned.append(target_tokens_sample[i])
        
        if len(student_logits_sample_list) == 0:
            continue
        
        student_logits_sample = torch.stack(student_logits_sample_list)  # [num_targets, vocab]
        
        # Collect for CE loss: align logits with target tokens
        if len(target_tokens_aligned) > 0:
            target_tokens_tensor = torch.stack(target_tokens_aligned).to(device)
            # Align lengths
            min_len = min(len(student_logits_sample), len(target_tokens_tensor))
            if min_len > 0:
                ce_logits_list.append(student_logits_sample[:min_len])
                ce_labels_list.append(target_tokens_tensor[:min_len])
        
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
        
        # MSE loss for latent thoughts
        if (batch_idx < len(student_latent_thoughts_list) and 
            batch_idx < len(teacher_latent_thoughts_list)):
            
            student_latents = student_latent_thoughts_list[batch_idx]  # List of [768] tensors
            teacher_latents = teacher_latent_thoughts_list[batch_idx]  # [num_latent, 768]
            
            # Debug: check what we have
            student_count = len(student_latents) if student_latents else 0
            teacher_count = len(teacher_latents) if teacher_latents is not None else 0
            
            if student_count > 0 and teacher_count > 0:
                # Stack student latent thoughts
                try:
                    student_latents_stacked = torch.stack([
                        thought.to(teacher_latents.device) 
                        for thought in student_latents
                    ])  # [num_student_latents, 768]
                    
                    # Ensure shapes match
                    min_latent_len = min(len(student_latents_stacked), len(teacher_latents))
                    if min_latent_len > 0:
                        student_aligned = student_latents_stacked[:min_latent_len]  # [min_len, 768]
                        teacher_aligned = teacher_latents[:min_latent_len]  # [min_len, 768]
                        
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
                    # This should not happen often - log if it does
                    pass
    
    # Compute CE loss (causal LM loss)
    ce_loss = torch.tensor(0.0, device=device)
    num_ce_samples = 0
    if len(ce_logits_list) > 0 and len(ce_labels_list) > 0:
        # Concatenate all logits and labels
        all_ce_logits = torch.cat(ce_logits_list, dim=0)  # [total_targets, vocab]
        all_ce_labels = torch.cat(ce_labels_list, dim=0)  # [total_targets]
        
        # Compute CrossEntropyLoss
        ce_criterion = torch.nn.CrossEntropyLoss()
        ce_loss = ce_criterion(all_ce_logits, all_ce_labels)
        num_ce_samples = len(ce_logits_list)
    
    # Average losses
    avg_ce_loss = ce_loss  # Already averaged across all tokens
    avg_kl_loss = total_kl_loss / num_kl_samples if num_kl_samples > 0 else torch.tensor(0.0, device=device)
    avg_cosine_loss = total_cosine_loss / num_cosine_samples if num_cosine_samples > 0 else torch.tensor(0.0, device=device)
    
    # Combined loss
    total_loss = (
        ce_weight * avg_ce_loss +
        kl_weight * avg_kl_loss +
        cosine_weight * avg_cosine_loss
    )
    
    return {
        'total_loss': total_loss,
        'ce_loss': avg_ce_loss,
        'kl_loss': avg_kl_loss,
        'cosine_loss': avg_cosine_loss,
        'num_ce_samples': num_ce_samples,
        'num_kl_samples': num_kl_samples,
        'num_cosine_samples': num_cosine_samples,
    }


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    ce_weight=1.0,
    kl_weight=1.0,
    cosine_weight=1.0,
    temperature=1.0,
    wandb_run=None,
    global_step=0,
    rank=0,
    world_size=1,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_cosine = 0.0
    num_batches = 0
    
    # Only show progress bar on rank 0
    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))
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
            target_tokens,
            batch['target_positions'],
            ce_weight=ce_weight,
            kl_weight=kl_weight,
            cosine_weight=cosine_weight,
            temperature=temperature,
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_ce += loss_dict['ce_loss'].item()
        total_kl += loss_dict['kl_loss'].item()
        total_cosine += loss_dict['cosine_loss'].item()
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'ce': loss_dict['ce_loss'].item(),
            'kl': loss_dict['kl_loss'].item(),
            'cos': loss_dict['cosine_loss'].item(),
        })
        
        # Log to wandb
        if wandb_run is not None and batch_idx % 10 == 0:  # Log every 10 batches
            wandb_run.log({
                'train/loss': loss.item(),
                'train/ce_loss': loss_dict['ce_loss'].item(),
                'train/kl_loss': loss_dict['kl_loss'].item(),
                'train/cosine_loss': loss_dict['cosine_loss'].item(),
                'train/num_ce_samples': loss_dict['num_ce_samples'],
                'train/num_kl_samples': loss_dict['num_kl_samples'],
                'train/num_cosine_samples': loss_dict['num_cosine_samples'],
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/step': global_step,
            })
    
    return {
        'avg_loss': total_loss / num_batches,
        'avg_ce': total_ce / num_batches,
        'avg_kl': total_kl / num_batches,
        'avg_cosine': total_cosine / num_batches,
    }, global_step


def validate_epoch(
    model,
    dataloader,
    device,
    ce_weight=1.0,
    kl_weight=1.0,
    cosine_weight=1.0,
    temperature=1.0,
    wandb_run=None,
    rank=0,
    world_size=1,
):
    """Validate for one epoch (no gradient updates)."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_cosine = 0.0
    num_batches = 0
    
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
                target_tokens,
                batch['target_positions'],
                ce_weight=ce_weight,
                kl_weight=kl_weight,
                cosine_weight=cosine_weight,
                temperature=temperature,
            )
            
            loss = loss_dict['total_loss']
            
            # Update metrics
            total_loss += loss.item()
            total_ce += loss_dict['ce_loss'].item()
            total_kl += loss_dict['kl_loss'].item()
            total_cosine += loss_dict['cosine_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'ce': loss_dict['ce_loss'].item(),
                'kl': loss_dict['kl_loss'].item(),
                'cos': loss_dict['cosine_loss'].item(),
            })
    
    return {
        'avg_loss': total_loss / num_batches,
        'avg_ce': total_ce / num_batches,
        'avg_kl': total_kl / num_batches,
        'avg_cosine': total_cosine / num_batches,
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.01),
    )
    
    # Training loop
    num_epochs = config.get('num_epochs', 10)
    
    # Setup learning rate schedulers
    # Base scheduler: Exponential decay (decays every epoch by a fixed factor)
    from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
    
    initial_lr = config.get('lr', 1e-4)
    # Calculate decay factor to reach ~10% of initial LR by end of training
    # gamma^num_epochs = 0.1 => gamma = 0.1^(1/num_epochs)
    gamma = 0.1 ** (1.0 / num_epochs)
    
    # Base scheduler: exponential decay (decays every epoch)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    # Additional scheduler: Reduce on plateau for KL loss
    # This will provide additional LR reduction when KL loss plateaus
    # Threshold: requires at least 3% relative improvement to count as progress
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
    ce_weight = config.get('ce_weight', 1.0)
    kl_weight = config.get('kl_weight', 1.0)
    cosine_weight = config.get('cosine_weight', 1.0)
    temperature = config.get('temperature', 1.0)
    
    if rank == 0:
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Loss weights: CE={ce_weight}, KL={kl_weight}, Cosine={cosine_weight}")
    
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
            ce_weight=ce_weight,
            kl_weight=kl_weight,
            cosine_weight=cosine_weight,
            temperature=temperature,
            wandb_run=wandb_run,
            global_step=global_step,
            rank=rank,
            world_size=world_size,
        )
        
        if rank == 0:
            print(f"Epoch {epoch + 1} - Train Loss: {train_metrics['avg_loss']:.4f}, "
                  f"CE: {train_metrics['avg_ce']:.4f}, KL: {train_metrics['avg_kl']:.4f}, Cosine: {train_metrics['avg_cosine']:.4f}")
        
        # Validation
        val_metrics = None
        if val_dataloader is not None:
            if rank == 0:
                print("Running validation...")
            val_metrics = validate_epoch(
                model,
                val_dataloader,
                device,
                ce_weight=ce_weight,
                kl_weight=kl_weight,
                cosine_weight=cosine_weight,
                temperature=temperature,
                wandb_run=wandb_run,
                rank=rank,
                world_size=world_size,
            )
            
            if rank == 0:
                print(f"Epoch {epoch + 1} - Val Loss: {val_metrics['avg_loss']:.4f}, "
                      f"CE: {val_metrics['avg_ce']:.4f}, KL: {val_metrics['avg_kl']:.4f}, Cosine: {val_metrics['avg_cosine']:.4f}")
        
        # Log epoch metrics to wandb (only on rank 0)
        if wandb_run is not None:
            log_dict = {
                'epoch': epoch + 1,
                'epoch/train_loss': train_metrics['avg_loss'],
                'epoch/train_ce_loss': train_metrics['avg_ce'],
                'epoch/train_kl_loss': train_metrics['avg_kl'],
                'epoch/train_cosine_loss': train_metrics['avg_cosine'],
            }
            if val_metrics is not None:
                log_dict.update({
                    'epoch/val_loss': val_metrics['avg_loss'],
                    'epoch/val_ce_loss': val_metrics['avg_ce'],
                    'epoch/val_kl_loss': val_metrics['avg_kl'],
                    'epoch/val_cosine_loss': val_metrics['avg_cosine'],
                })
            wandb_run.log(log_dict)
        
        # Update learning rate schedulers
        # Only on rank 0 to avoid multiple print statements
        if rank == 0:
            # Step exponential decay scheduler (decays every epoch)
            scheduler.step()
            
            # Also check plateau scheduler for additional reduction when CE loss plateaus
            # Use validation CE loss if available, otherwise training CE loss
            # (CE loss is more critical for sequence coherence)
            ce_loss_for_scheduler = val_metrics['avg_ce'] if val_metrics is not None else train_metrics['avg_ce']
            plateau_scheduler.step(ce_loss_for_scheduler)
            
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

