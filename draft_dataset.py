"""
Dataset for loading draft training data (JSON metadata + NPZ vector files).
Loads data from Modal volumes or local paths.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class DraftDataset(Dataset):
    """
    Dataset that loads draft training data.
    
    Each sample contains:
    - input_ids: Input sequence with latent tokens
    - target_tokens: Target token IDs to predict
    - target_logits: Target logits from Coconut (teacher)
    - latent_thoughts: Target latent thought vectors from Coconut (teacher)
    - latent_positions: Positions of latent tokens in input_ids
    - target_positions: Positions where we predict tokens
    """
    
    def __init__(
        self,
        json_path: str,
        data_dir: Optional[str] = None,
    ):
        """
        Args:
            json_path: Path to JSON metadata file
            data_dir: Directory containing NPZ files (if None, uses same dir as JSON)
        """
        self.json_path = json_path
        if data_dir is None:
            self.data_dir = os.path.dirname(json_path)
        else:
            self.data_dir = data_dir
        
        # Load metadata
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} samples from {json_path}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        
        # Load NPZ file
        npz_path = os.path.join(self.data_dir, sample['npz_file'])
        npz_data = np.load(npz_path)
        
        # Extract data
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        latent_positions = sample['latent_positions']
        target_positions = sample['target_positions']
        target_tokens = torch.tensor(sample['target_tokens'], dtype=torch.long)
        
        # Load vectors from NPZ
        latent_thoughts = torch.tensor(
            npz_data['latent_thoughts'], dtype=torch.float32
        )  # [num_latent_tokens, 768]
        
        target_logits = torch.tensor(
            npz_data['token_logits'], dtype=torch.float32
        )  # [num_target_tokens, vocab_size]
        
        return {
            'input_ids': input_ids,
            'latent_positions': latent_positions,
            'target_positions': target_positions,
            'target_tokens': target_tokens,
            'latent_thoughts': latent_thoughts,  # Teacher latent thoughts (768 dim)
            'target_logits': target_logits,  # Teacher logits
        }


class DraftCollator:
    """
    Custom collator for batching draft training data.
    Handles variable-length sequences and aligns latent thoughts with positions.
    Uses Coconut-style left-padding to align latent token positions across batch.
    """
    
    def __init__(self, pad_token_id=-100, latent_token_id=None):
        self.pad_token_id = pad_token_id
        self.latent_token_id = latent_token_id
    
    def __call__(self, batch):
        """
        Collate a batch of samples.
        
        Aligns latent token positions by left-padding (like Coconut's collator)
        so all samples have their first latent token at the same position.
        
        Returns batched tensors and metadata for loss computation.
        """
        batch_size = len(batch)
        
        # Find earliest latent position in each sample (if latent_token_id is provided)
        earliest_latent_positions = []
        if self.latent_token_id is not None:
            for item in batch:
                input_ids = item['input_ids']
                latent_pos = item.get('latent_positions', [])
                if len(latent_pos) > 0:
                    earliest_latent_positions.append(min(latent_pos))
                else:
                    earliest_latent_positions.append(None)
        
        # Find the latest earliest latent position (like Coconut)
        if earliest_latent_positions:
            latest_earliest_latent = max([p for p in earliest_latent_positions if p is not None] or [0])
        else:
            latest_earliest_latent = 0
        
        # First pass: calculate all left paddings
        left_paddings = []
        for item in batch:
            latent_positions = item.get('latent_positions', [])
            if self.latent_token_id is not None and len(latent_positions) > 0:
                earliest_latent = min(latent_positions)
                left_padding = latest_earliest_latent - earliest_latent
            else:
                left_padding = 0
            left_paddings.append(left_padding)
        
        # Calculate max total length after left padding
        max_input_len = max(len(item['input_ids']) + left_paddings[i] for i, item in enumerate(batch))
        
        input_ids_batch = []
        attention_mask_batch = []
        position_ids_batch = []
        updated_latent_positions_batch = []
        
        for item_idx, item in enumerate(batch):
            input_ids = item['input_ids']
            latent_positions = item.get('latent_positions', [])
            left_padding = left_paddings[item_idx]
            
            # Calculate right padding needed for max length
            right_padding = max_input_len - len(input_ids) - left_padding
            
            # Add left padding (align latent positions)
            input_ids_padded = torch.cat([
                torch.full((left_padding,), self.pad_token_id, dtype=torch.long),
                input_ids,
                torch.full((right_padding,), self.pad_token_id, dtype=torch.long)
            ])
            
            attention_mask = torch.cat([
                torch.zeros(left_padding, dtype=torch.long),
                torch.ones(len(input_ids), dtype=torch.long),
                torch.zeros(right_padding, dtype=torch.long)
            ])
            
            # Update position_ids and latent_positions to account for left padding
            # Coconut's approach: left-padded positions get position 0, then sequence continues
            position_ids = torch.zeros(max_input_len, dtype=torch.long)
            if left_padding > 0:
                # Left-padded positions stay at 0
                # Actual sequence starts at left_padding and continues
                position_ids[left_padding:] = torch.arange(
                    len(input_ids) + right_padding, dtype=torch.long
                )
            else:
                # No left padding, just regular positions
                position_ids = torch.arange(max_input_len, dtype=torch.long)
            
            # Update latent_positions to account for left padding
            updated_latent_positions = [pos + left_padding for pos in latent_positions]
            
            input_ids_batch.append(input_ids_padded)
            attention_mask_batch.append(attention_mask)
            position_ids_batch.append(position_ids)
            updated_latent_positions_batch.append(updated_latent_positions)
        
        input_ids_batch = torch.stack(input_ids_batch)
        attention_mask_batch = torch.stack(attention_mask_batch)
        position_ids_batch = torch.stack(position_ids_batch)
        
        # Collect target data (not batched, handled per-sample in loss)
        target_tokens_list = [item['target_tokens'] for item in batch]
        target_logits_list = [item['target_logits'] for item in batch]
        latent_thoughts_list = [item['latent_thoughts'] for item in batch]
        # Use updated latent positions (accounting for left padding)
        latent_positions_list = updated_latent_positions_batch
        # Adjust target_positions by the same left padding so indices line up with padded tensors
        target_positions_list = [
            [pos + left_paddings[idx] for pos in item['target_positions']]
            for idx, item in enumerate(batch)
        ]
        
        return {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'position_ids': position_ids_batch,
            'target_tokens': target_tokens_list,  # List of tensors
            'target_logits': target_logits_list,  # List of tensors
            'latent_thoughts': latent_thoughts_list,  # List of tensors
            'latent_positions': latent_positions_list,  # List of lists (updated for alignment)
            'target_positions': target_positions_list,  # List of lists
        }

