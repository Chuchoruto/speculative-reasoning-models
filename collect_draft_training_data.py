# Copyright (c) Meta Platforms, Inc. and affiliates.
# Script to collect latent thought vectors and logits for training a draft model

import torch
import torch.distributed as dist
import json
import os
import argparse
import yaml
from coconut import Coconut
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import get_dataset, get_cot_latent_dataset
from torch.utils.data import DataLoader, DistributedSampler
from dataset import MyCollator
from types import SimpleNamespace
from tqdm import tqdm

from utils import Config, set_seed


def collect_data_distributed(model, tokenizer, dataloader, output_path, rank, world_size, max_samples=None, total_dataset_size=None):
    """
    Collect latent thought vectors and logits from the Coconut model in distributed mode.
    
    Each rank processes its subset of data and saves to a separate file.
    Rank 0 combines all files at the end.
    """
    model.eval()
    collected_data = []
    
    device = next(model.parameters()).device
    
    # Progress bar only on rank 0
    # DistributedSampler already splits the dataset, so each rank sees a subset
    if rank == 0:
        total = min(total_dataset_size, max_samples) if (max_samples and total_dataset_size) else (total_dataset_size or len(dataloader) * world_size)
        pbar = tqdm(total=total)
    
    samples_collected = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Global check: stop if we've collected enough total samples across all ranks
            # Since we can't coordinate exactly, we approximate by checking each rank's count
            if max_samples:
                # Each rank processes roughly 1/world_size of max_samples
                rank_quota = (max_samples + world_size - 1) // world_size  # Ceiling division
                if samples_collected >= rank_quota:
                    break
                
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            labels = batch["labels"].to(device) if "labels" in batch else None
            position_ids = batch["position_ids"].to(device) if "position_ids" in batch else None
            
            # Extract latent token positions
            latent_token_id = model.latent_token_id
            latent_indices = (input_ids == latent_token_id).nonzero()
            
            if len(latent_indices) == 0:
                if rank == 0:
                    pbar.update(1)
                continue  # Skip samples without latent tokens
            
            # Get latent positions for this batch (assuming batch_size=1)
            latent_positions = sorted([idx[1].item() for idx in latent_indices if idx[0] == 0])
            
            # Forward pass with data collection
            outputs = model.forward(
                input_ids,
                attention_mask if attention_mask is not None else torch.ones_like(input_ids),
                labels if labels is not None else input_ids,
                position_ids if position_ids is not None else torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
                collect_latent_thoughts=True,
            )
            
            # Extract latent thought vectors directly from outputs
            if hasattr(outputs, 'latent_thoughts'):
                # Use collected latent thoughts (already on CPU)
                latent_thoughts = [thought.tolist() for thought in outputs.latent_thoughts]
            else:
                # Fallback: extract from inputs_embeds
                latent_thoughts = []
                for lat_pos in latent_positions:
                    latent_vec = outputs.inputs_embeds[0, lat_pos, :].cpu()
                    latent_thoughts.append(latent_vec.tolist())
            
            # Extract logits for positions where we actually predict tokens
            logits = outputs.logits[0].cpu()  # [seq_len, vocab_size]
            
            # Find positions where labels != -100 (these are the target tokens)
            if labels is not None:
                target_mask = (labels[0] != -100).cpu()
                target_positions = torch.where(target_mask)[0].tolist()
                
                # Get logits for target positions (shifted by 1 for next-token prediction)
                token_logits = []
                for pos in target_positions:
                    if pos > 0:  # logits[pos-1] predicts token at pos
                        token_logits.append(logits[pos - 1].tolist())
                
                # Get the actual target tokens
                target_tokens = labels[0][target_mask].cpu().tolist()
            else:
                token_logits = []
                target_tokens = []
            
            # Store collected data
            sample_data = {
                "input_ids": input_ids[0].cpu().tolist(),
                "latent_positions": latent_positions,
                "latent_thoughts": latent_thoughts,
                "target_positions": target_positions if labels is not None else [],
                "token_logits": token_logits,
                "target_tokens": target_tokens,
                "num_latent_tokens": len(latent_thoughts),
                "num_target_tokens": len(target_tokens),
            }
            
            collected_data.append(sample_data)
            samples_collected += 1
            
            if rank == 0:
                pbar.update(1)
                if (batch_idx + 1) % 10 == 0:
                    pbar.set_description(f"Collected {len(collected_data)} samples on rank {rank}")
    
    if rank == 0:
        pbar.close()
    
    # Each rank saves its own data to a separate file
    rank_output_path = output_path.replace('.json', f'_rank{rank}.json')
    os.makedirs(os.path.dirname(rank_output_path) if os.path.dirname(rank_output_path) else ".", exist_ok=True)
    
    with open(rank_output_path, 'w') as f:
        json.dump(collected_data, f, indent=2)
    
    print(f"Rank {rank}: Collected {len(collected_data)} samples. Saved to {rank_output_path}")
    
    # Wait for all ranks to finish writing
    dist.barrier()
    
    # Rank 0 combines all files
    if rank == 0:
        print("Combining data from all ranks...")
        all_data = collected_data.copy()
        
        for other_rank in range(1, world_size):
            other_rank_path = output_path.replace('.json', f'_rank{other_rank}.json')
            if os.path.exists(other_rank_path):
                with open(other_rank_path, 'r') as f:
                    other_data = json.load(f)
                    all_data.extend(other_data)
                # Optionally remove the rank-specific file
                # os.remove(other_rank_path)
        
        # Save combined data
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"✅ Total collected: {len(all_data)} samples. Combined file saved to {output_path}")
        
        # Clean up rank-specific files
        for other_rank in range(world_size):
            rank_path = output_path.replace('.json', f'_rank{other_rank}.json')
            if os.path.exists(rank_path):
                os.remove(rank_path)
    
    dist.barrier()
    return collected_data


def main():
    # Initialize distributed training
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    
    parser = argparse.ArgumentParser(description="Collect draft training data from Coconut model")
    parser.add_argument("config_file", help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    
    configs = Config(config_dict)
    set_seed(configs.seed)
    
    if rank == 0:
        print("Config:", config_dict)
    
    checkpoint_path = configs.load_model_path
    output_filename = getattr(configs, "output_filename", "draft_training_data.json")
    max_samples = getattr(configs, "max_samples", None)
    data_path = getattr(configs, "data_path", getattr(configs, "val_path", "data/gsm_valid.json"))
    max_latent_stage = configs.max_latent_stage
    c_thought = configs.c_thought
    
    output_dir = os.path.join(configs.save_path, "draft_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Load model and tokenizer
    if rank == 0:
        print("Loading model...")
    
    base_model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    # Resize token embeddings
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Wrap in Coconut
    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # Load checkpoint
    if checkpoint_path != "None":
        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{local_rank}")
        model.load_state_dict(checkpoint, strict=False)
    
    # Move to device
    model = model.to(local_rank)
    model.eval()
    
    # Prepare dataset
    if rank == 0:
        print(f"Loading dataset from {data_path}...")
    base_dataset = get_dataset(data_path, tokenizer)
    
    # Create dataset with latent tokens
    configs_dataset = SimpleNamespace(
        max_latent_stage=max_latent_stage,
        c_thought=c_thought,
        pad_latent_to_max=True,
        no_cot=False,
        uniform_prob=0.0,
    )
    
    scheduled_stage = max_latent_stage
    dataset = get_cot_latent_dataset(
        scheduled_stage,
        base_dataset,
        configs_dataset,
        start_id,
        latent_id,
        end_id,
        no_special_marker=False,
        shuffle=False,
    )
    
    # Create distributed sampler and dataloader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Must be 1 for now
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # Collect data
    if rank == 0:
        print("Collecting data...")
    
    collect_data_distributed(model, tokenizer, dataloader, output_path, rank, world_size, max_samples, total_dataset_size=len(dataset))
    
    if rank == 0:
        print("Done!")


if __name__ == "__main__":
    main()
