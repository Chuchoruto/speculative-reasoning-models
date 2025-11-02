"""
Draft model that learns to mimic Coconut's latent thoughts and logits.
Uses a smaller model (GPT2-mini, 512 hidden dim) with projection to match Coconut (768 hidden dim).
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2LMHeadModel
from collections import namedtuple

Outputs = namedtuple("Outputs", ["loss", "logits", "latent_thoughts"])


class DraftModel(nn.Module):
    """
    Draft model that learns Coconut's generation style.
    
    Architecture:
    - Base model: GPT2-mini (512 hidden dim)
    - Projection layer: 512 -> 768 (for latent thoughts to match Coconut)
    - Coconut-style forward pass with latent token replacement
    """
    
    def __init__(
        self,
        base_model,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        hidden_dim_base=512,
        hidden_dim_target=768,
    ):
        super(DraftModel, self).__init__()
        
        self.base_model = base_model
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.hidden_dim_base = hidden_dim_base
        self.hidden_dim_target = hidden_dim_target
        
        # Projection layer: 512 -> 768 for latent thoughts
        self.latent_projection = nn.Linear(hidden_dim_base, hidden_dim_target)
        
        # Get embedding layer
        if isinstance(self.base_model, GPT2LMHeadModel):
            self.embedding = self.base_model.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_model.get_input_embeddings()
    
    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        collect_latent_thoughts=True,
        **kwargs
    ):
        """
        Forward pass with Coconut-style latent token processing.
        
        Returns:
            Outputs with logits and latent thoughts (projected to 768 dim)
        """
        logits = []
        # Track latent thoughts per sample: List[List[Tensor]] - outer list is samples, inner is latent thoughts
        latent_thoughts_collected = [[] for _ in range(input_ids.shape[0])]
        
        # Find latent token positions (only in non-padding positions)
        # Filter out padding positions using attention_mask if provided
        if attention_mask is not None:
            valid_mask = (input_ids == self.latent_token_id) & (attention_mask == 1)
        else:
            valid_mask = (input_ids == self.latent_token_id)
        
        latent_indices = valid_mask.nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists and max([len(l) for l in latent_lists]) > 0 else 0
        
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
        
        kv_cache = None
        
        # Process latent tokens (similar to Coconut)
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # First forward pass
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                # Use KV cache
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]
            
            logits.append(outputs.logits)
            
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )
            
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            kv_cache = outputs.past_key_values
            
            # Feedback continuous thoughts to input_embeds
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            
            # Convert to list for in-place operations
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            
            # Replace latent tokens with hidden states
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                
                # Calculate the index in hidden_states
                hidden_state_idx = token_idx - 1 - hidden_states_offset
                
                # Bounds check: ensure the index is valid
                if hidden_state_idx < 0 or hidden_state_idx >= hidden_states.shape[1]:
                    # Skip if index is out of bounds (e.g., due to padding or chunk boundaries)
                    continue
                
                # Also check if this is a padding position (shouldn't have latent tokens, but be safe)
                if token_idx >= inputs_embeds.shape[1]:
                    continue
                
                # Get hidden state that will replace latent token (512 dim)
                thought_vector_512 = hidden_states[
                    batch_idx, hidden_state_idx, :
                ]
                
                # Project to target dimension (768) for collection
                if collect_latent_thoughts:
                    thought_vector_768 = self.latent_projection(thought_vector_512)
                    # Store in the correct sample's list
                    latent_thoughts_collected[batch_idx].append(thought_vector_768.clone())
                
                # Use original 512-dim vector for replacement (model works in its own space)
                tensor_list[batch_idx][token_idx] = thought_vector_512
            
            # Reassemble inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )
        
        # Final pass
        outputs = self.base_model(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )
        
        logits.append(outputs.logits)
        logits = torch.cat(logits, dim=-2)
        
        # Placeholder loss (will be computed in training loop)
        loss = None
        
        # Return latent thoughts organized by sample, or None if not collected
        return_latent_thoughts = latent_thoughts_collected if collect_latent_thoughts else None
        
        return Outputs(
            loss=loss,
            logits=logits,
            latent_thoughts=return_latent_thoughts
        )
    
    def train(self, mode=True):
        """Set training mode. Compatible with DDP."""
        self.base_model.train(mode)
        self.latent_projection.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.base_model.eval()
        self.latent_projection.eval()
        return self

