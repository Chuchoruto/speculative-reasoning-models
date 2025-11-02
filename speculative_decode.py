"""
Speculative decoding with draft model and target (Coconut) model.
Implements latent thought verification and token acceptance criteria.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()


def accept_latent_thought(
    draft_vector: torch.Tensor,
    target_vector: torch.Tensor,
    threshold: float = 0.9
) -> bool:
    """
    Accept latent thought if cosine similarity > threshold.
    
    Args:
        draft_vector: Draft model's latent thought [768]
        target_vector: Target model's latent thought [768]
        threshold: Cosine similarity threshold (default 0.9)
    
    Returns:
        True if accepted, False otherwise
    """
    similarity = cosine_similarity(draft_vector, target_vector)
    return similarity > threshold


def accept_token_speculative(
    draft_prob: float,
    target_prob: float,
    rng: Optional[np.random.Generator] = None
) -> Tuple[bool, float]:
    """
    Speculative decoding token acceptance.
    
    If p(x) >= q(x): always accept
    If p(x) < q(x): accept with probability 1 - (p(x)/q(x))
    
    Args:
        draft_prob: Probability from draft model q(x)
        target_prob: Probability from target model p(x)
        rng: Random number generator
    
    Returns:
        (accepted: bool, adjusted_prob: float)
        adjusted_prob is the rejection sampling probability if rejected
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if target_prob >= draft_prob:
        # Always accept
        return True, target_prob
    else:
        # Accept with probability 1 - (p(x)/q(x))
        accept_prob = 1.0 - (target_prob / draft_prob)
        if rng.random() < accept_prob:
            return True, target_prob
        else:
            # Reject: sample from adjusted distribution
            # The adjusted prob is max(0, p(x) - q(x)) normalized
            # We'll return the rejection sampling probability
            adjusted_prob = max(0.0, target_prob - draft_prob)
            return False, adjusted_prob


def speculative_decode_latent_thoughts(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    latent_positions: List[int],
    num_latent_thoughts: int = 6,
    similarity_threshold: float = 0.9,
    device: torch.device = None,
) -> Tuple[List[bool], List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate and verify latent thoughts between draft and target models.
    
    Args:
        draft_model: Draft model instance
        target_model: Target (Coconut) model instance
        input_ids: Input sequence [seq_len]
        attention_mask: Attention mask [seq_len]
        position_ids: Position IDs [seq_len]
        latent_positions: Positions of latent tokens in sequence
        num_latent_thoughts: Number of latent thoughts to generate (default 6)
        similarity_threshold: Cosine similarity threshold for acceptance
        device: Device to run on
    
    Returns:
        (acceptance_list, draft_thoughts, target_thoughts)
    """
    if device is None:
        device = input_ids.device
    
    # Expand to batch dimension
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    position_ids = position_ids.unsqueeze(0).to(device)
    
    # Generate draft latent thoughts
    draft_model.eval()
    with torch.no_grad():
        draft_outputs = draft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            collect_latent_thoughts=True,
            latent_positions=[latent_positions],
        )
        # Draft model returns latent thoughts on GPU (from the model), convert to device
        draft_thoughts = [
            thought.to(device) if isinstance(thought, torch.Tensor) else thought
            for thought in (draft_outputs.latent_thoughts[0] if draft_outputs.latent_thoughts else [])
        ]
    
    # Generate target latent thoughts
    target_model.eval()
    with torch.no_grad():
        # Coconut model requires labels (can be dummy)
        dummy_labels = torch.full(
            (input_ids.shape[0], input_ids.shape[1]),
            -100,
            dtype=torch.long,
            device=device
        )
        target_outputs = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=dummy_labels,
            collect_latent_thoughts=True,
        )
        # Coconut model returns latent thoughts on CPU (via .clone().cpu()), convert to device
        target_thoughts = [
            thought.to(device) if isinstance(thought, torch.Tensor) else thought
            for thought in (target_outputs.latent_thoughts if target_outputs.latent_thoughts else [])
        ]
    
    # Verify each latent thought
    acceptance_list = []
    min_len = min(len(draft_thoughts), len(target_thoughts), num_latent_thoughts)
    
    for i in range(min_len):
        # Both tensors should already be on device, but ensure they are
        draft_vec = draft_thoughts[i] if isinstance(draft_thoughts[i], torch.Tensor) else torch.tensor(draft_thoughts[i], device=device)
        target_vec = target_thoughts[i] if isinstance(target_thoughts[i], torch.Tensor) else torch.tensor(target_thoughts[i], device=device)
        
        accepted = accept_latent_thought(
            draft_vec,
            target_vec,
            threshold=similarity_threshold
        )
        acceptance_list.append(accepted)
    
    return acceptance_list, draft_thoughts[:min_len], target_thoughts[:min_len]


def speculative_decode_tokens(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    gamma: int = 4,
    max_new_tokens: int = 50,
    eos_token_id: int = None,
    device: torch.device = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], int, int]:
    """
    Speculative decoding for token generation.
    
    Args:
        draft_model: Draft model instance
        target_model: Target (Coconut) model instance
        input_ids: Current input sequence [seq_len]
        attention_mask: Attention mask [seq_len]
        position_ids: Position IDs [seq_len]
        gamma: Number of draft tokens to generate per round
        max_new_tokens: Maximum total new tokens to generate
        eos_token_id: EOS token ID to stop generation
        device: Device to run on
        rng: Random number generator
    
    Returns:
        (generated_tokens, num_draft_calls, num_target_calls)
    """
    if device is None:
        device = input_ids.device
    
    if rng is None:
        rng = np.random.default_rng()
    
    draft_model.eval()
    target_model.eval()
    
    generated_tokens = []
    current_input_ids = input_ids.clone().to(device)
    current_attention_mask = attention_mask.clone().to(device)
    current_position_ids = position_ids.clone().to(device)
    
    num_draft_calls = 0
    num_target_calls = 0
    
    for _ in range(max_new_tokens):
        # Draft model generates gamma tokens
        draft_tokens = []
        draft_logits_list = []
        
        draft_input = current_input_ids.unsqueeze(0)
        draft_attn = current_attention_mask.unsqueeze(0)
        draft_pos = current_position_ids.unsqueeze(0)
        
        for _ in range(gamma):
            num_draft_calls += 1
            with torch.no_grad():
                # Get draft logits
                draft_outputs = draft_model(
                    input_ids=draft_input,
                    attention_mask=draft_attn,
                    position_ids=draft_pos,
                    collect_latent_thoughts=False,
                )
                draft_logits = draft_outputs.logits[0, -1, :]  # Last position logits
                draft_logits_list.append(draft_logits)
                
                # Sample from draft distribution
                draft_probs = F.softmax(draft_logits, dim=-1)
                draft_token = torch.multinomial(draft_probs, 1).item()
                draft_tokens.append(draft_token)
                
                # Append to input for next draft token
                draft_input = torch.cat([
                    draft_input,
                    torch.tensor([[draft_token]], device=device)
                ], dim=1)
                draft_attn = torch.cat([
                    draft_attn,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                ], dim=1)
                draft_pos = torch.cat([
                    draft_pos,
                    torch.tensor([[draft_pos[0, -1].item() + 1]], device=device)
                ], dim=1)
        
        # Verify draft tokens with target model
        verified_tokens = []
        target_input = current_input_ids.unsqueeze(0)
        target_attn = current_attention_mask.unsqueeze(0)
        target_pos = current_position_ids.unsqueeze(0)
        
        all_accepted = True
        
        for i, draft_token in enumerate(draft_tokens):
            num_target_calls += 1
            with torch.no_grad():
                # Get target logits
                # Note: Coconut model requires labels parameter (can be None or dummy)
                dummy_labels = torch.full(
                    (target_input.shape[0], target_input.shape[1]),
                    -100,
                    dtype=torch.long,
                    device=device
                )
                target_outputs = target_model(
                    input_ids=target_input,
                    attention_mask=target_attn,
                    position_ids=target_pos,
                    labels=dummy_labels,
                    collect_latent_thoughts=False,
                )
                target_logits = target_outputs.logits[0, -1, :]
                
                # Get probabilities
                target_probs = F.softmax(target_logits, dim=-1)
                draft_probs = F.softmax(draft_logits_list[i], dim=-1)
                
                draft_prob = draft_probs[draft_token].item()
                target_prob = target_probs[draft_token].item()
                
                # Speculative decoding acceptance
                accepted, _ = accept_token_speculative(
                    draft_prob,
                    target_prob,
                    rng=rng
                )
                
                if accepted:
                    verified_tokens.append(draft_token)
                    # Append to target input for next verification
                    target_input = torch.cat([
                        target_input,
                        torch.tensor([[draft_token]], device=device)
                    ], dim=1)
                    target_attn = torch.cat([
                        target_attn,
                        torch.ones((1, 1), device=device, dtype=torch.long)
                    ], dim=1)
                    target_pos = torch.cat([
                        target_pos,
                        torch.tensor([[target_pos[0, -1].item() + 1]], device=device)
                    ], dim=1)
                else:
                    # Rejected: sample from adjusted distribution
                    # Adjusted distribution: norm(max(0, p(x) - q(x)))
                    all_accepted = False
                    adjusted_probs = torch.clamp(target_probs - draft_probs, min=0.0)
                    adjusted_sum = adjusted_probs.sum()
                    
                    if adjusted_sum > 0:
                        adjusted_probs = adjusted_probs / adjusted_sum
                        sampled_token = torch.multinomial(adjusted_probs, 1).item()
                    else:
                        # Fallback: sample from target distribution
                        sampled_token = torch.multinomial(target_probs, 1).item()
                    
                    verified_tokens.append(sampled_token)
                    
                    # Update current state with the sampled token and break
                    target_input = torch.cat([
                        target_input,
                        torch.tensor([[sampled_token]], device=device)
                    ], dim=1)
                    current_input_ids = target_input[0]
                    current_attention_mask = torch.cat([
                        target_attn,
                        torch.ones((1, 1), device=device, dtype=torch.long)
                    ], dim=1)[0]
                    current_position_ids = torch.cat([
                        target_pos,
                        torch.tensor([[target_pos[0, -1].item() + 1]], device=device)
                    ], dim=1)[0]
                    break
        
        # Add verified tokens to output
        generated_tokens.extend(verified_tokens)
        
        # Check for EOS
        if eos_token_id is not None and eos_token_id in verified_tokens:
            break
        
        # If all accepted, update current state and continue drafting
        if all_accepted:
            current_input_ids = target_input[0]
            current_attention_mask = target_attn[0]
            current_position_ids = target_pos[0]
    
    return generated_tokens, num_draft_calls, num_target_calls


def speculative_decode(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    latent_positions: List[int],
    num_latent_thoughts: int = 6,
    gamma: int = 4,
    max_new_tokens: int = 50,
    eos_token_id: int = None,
    similarity_threshold: float = 0.9,
    device: torch.device = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], dict]:
    """
    Full speculative decoding pipeline: latent thoughts + tokens.
    
    Args:
        draft_model: Draft model instance
        target_model: Target (Coconut) model instance
        input_ids: Input sequence [seq_len]
        attention_mask: Attention mask [seq_len]
        position_ids: Position IDs [seq_len]
        latent_positions: Positions of latent tokens
        num_latent_thoughts: Number of latent thoughts (default 6)
        gamma: Number of draft tokens per round
        max_new_tokens: Maximum new tokens to generate
        eos_token_id: EOS token ID
        similarity_threshold: Cosine similarity threshold for latent thoughts
        device: Device to run on
        rng: Random number generator
    
    Returns:
        (generated_tokens, stats_dict)
    """
    if device is None:
        device = input_ids.device
    
    if rng is None:
        rng = np.random.default_rng()
    
    stats = {
        'latent_accepted': 0,
        'latent_total': 0,
        'num_draft_calls': 0,
        'num_target_calls': 0,
    }
    
    # Step 1: Generate and verify latent thoughts
    acceptance_list, draft_thoughts, target_thoughts = speculative_decode_latent_thoughts(
        draft_model,
        target_model,
        input_ids,
        attention_mask,
        position_ids,
        latent_positions,
        num_latent_thoughts=num_latent_thoughts,
        similarity_threshold=similarity_threshold,
        device=device,
    )
    
    stats['latent_accepted'] = sum(acceptance_list)
    stats['latent_total'] = len(acceptance_list)
    
    # If not all latent thoughts accepted, we might need to handle this
    # For now, continue with token generation regardless
    
    # Step 2: Generate tokens after latent thoughts
    # Find the position after latent tokens (after <end-latent> if present, or after last latent)
    if len(latent_positions) > 0:
        # Get position after latent tokens
        # Assuming sequence structure: ... <start-latent> <latent>*6 <end-latent> ...
        # We want to start from after <end-latent> or after the last latent token
        last_latent_pos = max(latent_positions)
        start_pos = last_latent_pos + 1
    else:
        start_pos = len(input_ids)
    
    # Create sequence from start of input to position after latent tokens
    if start_pos < len(input_ids):
        token_input_ids = input_ids[:start_pos].clone()
        token_attention_mask = attention_mask[:start_pos].clone()
        token_position_ids = position_ids[:start_pos].clone()
    else:
        # If we're already at the end, use full sequence
        token_input_ids = input_ids.clone()
        token_attention_mask = attention_mask.clone()
        token_position_ids = position_ids.clone()
    
    # Generate tokens with speculative decoding
    # This will generate new tokens after the latent thought section
    generated_tokens, num_draft, num_target = speculative_decode_tokens(
        draft_model,
        target_model,
        token_input_ids,
        token_attention_mask,
        token_position_ids,
        gamma=gamma,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        device=device,
        rng=rng,
    )
    
    stats['num_draft_calls'] = num_draft
    stats['num_target_calls'] = num_target
    
    return generated_tokens, stats

