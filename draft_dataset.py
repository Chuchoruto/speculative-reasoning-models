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
    """
    
    def __init__(self, pad_token_id=-100):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        """
        Collate a batch of samples.
        
        Returns batched tensors and metadata for loss computation.
        """
        batch_size = len(batch)
        
        # Pad input_ids to same length
        max_input_len = max(len(item['input_ids']) for item in batch)
        input_ids_batch = []
        attention_mask_batch = []
        
        for item in batch:
            input_ids = item['input_ids']
            padding_len = max_input_len - len(input_ids)
            
            input_ids_padded = torch.cat([
                input_ids,
                torch.full((padding_len,), self.pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                torch.ones(len(input_ids), dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ])
            
            input_ids_batch.append(input_ids_padded)
            attention_mask_batch.append(attention_mask)
        
        input_ids_batch = torch.stack(input_ids_batch)
        attention_mask_batch = torch.stack(attention_mask_batch)
        
        # Create position_ids
        position_ids_batch = torch.arange(
            max_input_len, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Collect target data (not batched, handled per-sample in loss)
        target_tokens_list = [item['target_tokens'] for item in batch]
        target_logits_list = [item['target_logits'] for item in batch]
        latent_thoughts_list = [item['latent_thoughts'] for item in batch]
        latent_positions_list = [item['latent_positions'] for item in batch]
        target_positions_list = [item['target_positions'] for item in batch]
        
        return {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'position_ids': position_ids_batch,
            'target_tokens': target_tokens_list,  # List of tensors
            'target_logits': target_logits_list,  # List of tensors
            'latent_thoughts': latent_thoughts_list,  # List of tensors
            'latent_positions': latent_positions_list,  # List of lists
            'target_positions': target_positions_list,  # List of lists
        }

