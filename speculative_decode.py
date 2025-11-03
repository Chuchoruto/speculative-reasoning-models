"""
Speculative decoding with draft model and target (Coconut) model.
Implements latent thought verification and token acceptance criteria.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
import time


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
    gamma: int = 6,  # Unused, kept for API compatibility
    similarity_threshold: float = 0.9,
    device: torch.device = None,
) -> Tuple[List[bool], List[torch.Tensor], List[torch.Tensor], float, float]:
    """
    Generate draft latent thoughts sequentially (6 autoregressive calls),
    then verify all with target model in one parallel call.
    
    Process:
    1. Draft model generates latent thoughts sequentially (one per position)
    2. Construct input sequence with draft latent thoughts embedded
    3. Target model processes this sequence in ONE call, generating target latent thoughts
    4. Compare each draft latent with corresponding target latent
    
    Args:
        draft_model: Draft model instance
        target_model: Target (Coconut) model instance
        input_ids: Input sequence [seq_len]
        attention_mask: Attention mask [seq_len]
        position_ids: Position IDs [seq_len]
        latent_positions: Positions of latent tokens in sequence (sorted)
        num_latent_thoughts: Total number of latent thoughts to generate (default 6)
        gamma: Unused (kept for API compatibility)
        similarity_threshold: Cosine similarity threshold for acceptance
        device: Device to run on
    
    Returns:
        (acceptance_list, draft_thoughts, target_thoughts, draft_time, verify_time)
        - draft_time: Time taken to generate draft latent thoughts
        - verify_time: Time taken to verify with target model
    """
    if device is None:
        device = input_ids.device
    
    # Expand to batch dimension
    input_ids_batch = input_ids.unsqueeze(0).to(device)
    attention_mask_batch = attention_mask.unsqueeze(0).to(device)
    position_ids_batch = position_ids.unsqueeze(0).to(device)
    
    draft_model.eval()
    target_model.eval()
    
    # Sort latent positions to ensure sequential processing
    sorted_latent_positions = sorted(latent_positions)[:num_latent_thoughts]
    
    # Step 1: Generate draft latent thoughts SEQUENTIALLY
    # The draft model processes latent tokens sequentially in its forward pass,
    # so each latent thought depends on previous ones (draft_latent 1 -> draft_latent 2, etc.)
    # We call it once with all positions - it handles sequential processing internally
    all_draft_thoughts = []
    
    # Time the draft generation
    draft_start_time = time.perf_counter()
    with torch.no_grad():
        # Call draft model once - it processes all latent positions sequentially internally
        # Each latent thought is generated autoregressively (depends on previous ones)
        draft_outputs = draft_model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            position_ids=position_ids_batch,
            collect_latent_thoughts=True,
            latent_positions=[sorted_latent_positions],  # All positions at once
        )
        
        # Extract all draft latent thoughts (already projected to 768-dim)
        if draft_outputs.latent_thoughts and len(draft_outputs.latent_thoughts[0]) > 0:
            all_draft_thoughts = [
                thought.to(device) if isinstance(thought, torch.Tensor) else torch.tensor(thought, device=device)
                for thought in draft_outputs.latent_thoughts[0]
            ]
    draft_end_time = time.perf_counter()
    draft_time = draft_end_time - draft_start_time
    
    # Step 2: Construct input sequence with ALL draft latent thoughts embedded
    # Get base embeddings from target model
    with torch.no_grad():
        base_embeddings = target_model.embedding(input_ids_batch)
        
        # Replace latent token positions with draft latent thoughts (768-dim)
        for i, latent_pos in enumerate(sorted_latent_positions):
            if i < len(all_draft_thoughts):
                # Target model expects 768-dim embeddings
                base_embeddings[0, latent_pos, :] = all_draft_thoughts[i]
    
    # Step 3: Target model verifies in ONE batched call using 6 parallel sequences
    # Build a batch where sample k has first k-1 latent tokens replaced by draft thoughts
    # and only the k-th position remains a latent token to collect one thought per batch row
    verify_start_time = time.perf_counter()
    with torch.no_grad():
        batch_input_ids = []
        batch_attention = []
        batch_positions = []
        batch_embeddings = []
        # Use model-known special ids
        latent_id = getattr(target_model, 'latent_token_id', None)
        end_latent_id = getattr(target_model, 'end_latent_id', None)
        # Fallback: keep the same id if not available
        for k, pos_k in enumerate(sorted_latent_positions, start=1):
            # Start from device tensors
            ids_k = input_ids_batch[0].clone()  # [L] on device
            # Ensure exactly ONE latent token remains at pos_k.
            # All other latent positions are set to a non-latent token (end_latent or eos) to avoid multiple latents.
            neutral_id = None
            if hasattr(target_model, 'end_latent_id') and target_model.end_latent_id is not None:
                neutral_id = target_model.end_latent_id
            elif hasattr(target_model, 'eos_token_id') and target_model.eos_token_id is not None:
                neutral_id = target_model.eos_token_id
            else:
                # Fallback: if we don't have a neutral id, keep original id (worst case)
                neutral_id = ids_k[pos_k].item()
            for p in sorted_latent_positions:
                if p != pos_k:
                    ids_k[p] = neutral_id
            # Build embeddings and overwrite previous positions with draft thoughts
            emb_k = target_model.embedding(ids_k.unsqueeze(0))  # [1, L, 768] on device
            for i_thought, prev_pos in enumerate(sorted_latent_positions[:k-1]):
                if i_thought < len(all_draft_thoughts):
                    emb_k[0, prev_pos, :] = all_draft_thoughts[i_thought]
            batch_input_ids.append(ids_k.unsqueeze(0))
            batch_attention.append(attention_mask_batch[0].unsqueeze(0))
            batch_positions.append(position_ids_batch[0].unsqueeze(0))
            batch_embeddings.append(emb_k)
        # Stack batch
        batch_input_ids = torch.cat(batch_input_ids, dim=0).to(device)
        batch_attention = torch.cat(batch_attention, dim=0).to(device)
        batch_positions = torch.cat(batch_positions, dim=0).to(device)
        batch_embeddings = torch.cat(batch_embeddings, dim=0).to(device)
        dummy_labels = torch.full(
            (batch_input_ids.shape[0], batch_input_ids.shape[1]),
            -100,
            dtype=torch.long,
            device=device,
        )
        # One forward pass for all k
        target_outputs = target_model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention,
            position_ids=batch_positions,
            labels=dummy_labels,
            collect_latent_thoughts=True,
            inputs_embeds=batch_embeddings,
        )
        # With one latent per batch row, the model will collect exactly one thought per row in batch order
        raw_thoughts = target_outputs.latent_thoughts if target_outputs.latent_thoughts else []
        all_target_thoughts = [
            t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)
            for t in raw_thoughts
        ]
    verify_end_time = time.perf_counter()
    verify_time = verify_end_time - verify_start_time
    
    # Step 4: Verify all latent thoughts in parallel
    all_acceptance = []
    min_len = min(len(all_draft_thoughts), len(all_target_thoughts), num_latent_thoughts)
    
    for i in range(min_len):
        draft_vec = all_draft_thoughts[i]
        target_vec = all_target_thoughts[i]
        
        accepted = accept_latent_thought(
            draft_vec,
            target_vec,
            threshold=similarity_threshold
        )
        all_acceptance.append(accepted)
    
    # Return only the verified thoughts (slice to min_len)
    return all_acceptance, all_draft_thoughts[:min_len], all_target_thoughts[:min_len], draft_time, verify_time


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
) -> Tuple[List[int], int, int, int, int]:
    """
    Speculative decoding for token generation with parallel verification.
    
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
        (generated_tokens, num_draft_calls, num_target_calls, tokens_accepted, tokens_total)
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
    tokens_accepted = 0
    tokens_total = 0
    
    while len(generated_tokens) < max_new_tokens:
        # Draft model generates gamma tokens sequentially
        draft_tokens = []
        draft_logits_list = []
        
        draft_input = current_input_ids.unsqueeze(0)
        draft_attn = current_attention_mask.unsqueeze(0)
        draft_pos = current_position_ids.unsqueeze(0)
        
        # Generate all gamma draft tokens
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
                
                # Greedy decoding: use argmax
                draft_token = torch.argmax(draft_logits, dim=-1).item()
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
        
        # PARALLEL VERIFICATION: Build full sequence with all draft tokens
        # and call target model ONCE to get logits for all positions
        target_input = current_input_ids.unsqueeze(0)
        target_attn = current_attention_mask.unsqueeze(0)
        target_pos = current_position_ids.unsqueeze(0)
        
        # Append all draft tokens to target input for parallel verification
        draft_tokens_tensor = torch.tensor([draft_tokens], device=device)
        target_input_full = torch.cat([target_input, draft_tokens_tensor], dim=1)
        target_attn_full = torch.cat([
            target_attn,
            torch.ones((1, len(draft_tokens)), device=device, dtype=torch.long)
        ], dim=1)
        target_pos_full = torch.cat([
            target_pos,
            torch.arange(
                target_pos[0, -1].item() + 1,
                target_pos[0, -1].item() + 1 + len(draft_tokens),
                device=device
            ).unsqueeze(0)
        ], dim=1)
        
        # Single forward pass to get logits for all draft token positions
        num_target_calls += 1
        with torch.no_grad():
            dummy_labels = torch.full(
                (target_input_full.shape[0], target_input_full.shape[1]),
                -100,
                dtype=torch.long,
                device=device
            )
            target_outputs = target_model(
                input_ids=target_input_full,
                attention_mask=target_attn_full,
                position_ids=target_pos_full,
                labels=dummy_labels,
                collect_latent_thoughts=False,
            )
            
            # Extract logits for each draft token position
            # Logits shape: [batch=1, seq_len, vocab_size]
            # Note: logits[i] are the logits for predicting token at position i+1
            # So for draft token at position base_len+i, we need logits[base_len+i-1]
            # For draft token 0 (at base_len), we need logits[base_len-1]
            # For draft token i (at base_len+i), we need logits[base_len+i-1]
            base_len = target_input.shape[1]
            # Extract logits at positions [base_len-1, base_len, ..., base_len+len(draft_tokens)-2]
            # These correspond to predictions for draft tokens at positions [base_len, base_len+1, ..., base_len+len(draft_tokens)-1]
            start_idx = base_len - 1
            end_idx = base_len + len(draft_tokens) - 1
            target_logits_all = target_outputs.logits[0, start_idx:end_idx, :]
        
        # Verify all tokens in parallel using the extracted logits
        verified_tokens = []
        all_accepted = True
        first_rejection_idx = None
        
        for i, draft_token in enumerate(draft_tokens):
            tokens_total += 1
            
            # Get logits for this position
            target_logits_i = target_logits_all[i, :]  # [vocab_size]
            draft_logits_i = draft_logits_list[i]  # [vocab_size]
            
            # Greedy decoding: get argmax for both models
            draft_argmax = torch.argmax(draft_logits_i, dim=-1).item()
            target_argmax = torch.argmax(target_logits_i, dim=-1).item()
            
            # Strict acceptance: only accept if argmax matches
            accepted = (draft_argmax == target_argmax)
            
            if accepted:
                # Use the target model's argmax (which matches draft)
                verified_tokens.append(target_argmax)
                tokens_accepted += 1
            else:
                # Rejected: use target model's argmax and stop
                all_accepted = False
                first_rejection_idx = i
                verified_tokens.append(target_argmax)  # Use target's argmax
                break  # Stop after first rejection
        
        # Add verified tokens to output
        generated_tokens.extend(verified_tokens)
        
        # Check for EOS
        if eos_token_id is not None and eos_token_id in verified_tokens:
            break
        
        # Update current state for next iteration
        if all_accepted:
            # All tokens accepted: continue from end of verified sequence
            current_input_ids = target_input_full[0]
            current_attention_mask = target_attn_full[0]
            current_position_ids = target_pos_full[0]
        else:
            # Rejection occurred: restart from the token after the rejection
            # (the rejected token was replaced, so we continue from there)
            verified_len = len(verified_tokens)
            current_input_ids = target_input_full[0, :base_len + verified_len]
            current_attention_mask = target_attn_full[0, :base_len + verified_len]
            current_position_ids = target_pos_full[0, :base_len + verified_len]
        
        # Break if we've generated enough tokens
        if len(generated_tokens) >= max_new_tokens:
            break
    
    return generated_tokens, num_draft_calls, num_target_calls, tokens_accepted, tokens_total


def autoregressive_token_generation(
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: int = None,
    device: torch.device = None,
) -> List[int]:
    """
    Generate tokens autoregressively using only the target model.
    
    Args:
        target_model: Target (Coconut) model instance
        input_ids: Input sequence [seq_len]
        attention_mask: Attention mask [seq_len]
        position_ids: Position IDs [seq_len]
        max_new_tokens: Maximum new tokens to generate
        eos_token_id: EOS token ID
        device: Device to run on
    
    Returns:
        generated_tokens: List of generated token IDs
    """
    if device is None:
        device = input_ids.device
    
    target_model.eval()
    
    # Expand to batch dimension
    current_input_ids = input_ids.unsqueeze(0).to(device)
    current_attention_mask = attention_mask.unsqueeze(0).to(device)
    current_position_ids = position_ids.unsqueeze(0).to(device)
    
    generated_tokens = []
    
    # Autoregressive generation
    for _ in range(max_new_tokens):
        with torch.no_grad():
            dummy_labels = torch.full(
                (current_input_ids.shape[0], current_input_ids.shape[1]),
                -100,
                dtype=torch.long,
                device=device
            )
            outputs = target_model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                labels=dummy_labels,
                collect_latent_thoughts=False,
            )
            
            # Get next token (greedy decoding)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated_tokens.append(next_token)
            
            # Check for EOS
            if eos_token_id is not None and next_token == eos_token_id:
                break
            
            # Append to input for next iteration
            current_input_ids = torch.cat([
                current_input_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((1, 1), device=device, dtype=torch.long)
            ], dim=1)
            current_position_ids = torch.cat([
                current_position_ids,
                torch.tensor([[current_position_ids[0, -1].item() + 1]], device=device)
            ], dim=1)
    
    return generated_tokens


def speculative_decode(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    latent_positions: List[int],
    num_latent_thoughts: int = 6,
    gamma: int = 6,
    max_new_tokens: int = 50,
    eos_token_id: int = None,
    similarity_threshold: float = 0.9,
    device: torch.device = None,
    rng: Optional[np.random.Generator] = None,
    tokens_speculative: bool = False,
    skip_latent_verification: bool = False,
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
        gamma: Number of draft tokens per round (for tokens). Note: latent thoughts are now processed all at once (no batching)
        max_new_tokens: Maximum new tokens to generate
        eos_token_id: EOS token ID
        similarity_threshold: Cosine similarity threshold for latent thoughts
        device: Device to run on
        rng: Random number generator
        tokens_speculative: If True, use speculative decoding for tokens. If False, use autoregressive generation with target model only.
        skip_latent_verification: If True, skip target model verification of latent thoughts (use draft only). Much faster but less accurate.
    
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
    
    # Track wallclock time for speculative decoding
    start_time = time.perf_counter()
    
    # Step 1: Generate latent thoughts
    latent_start_time = time.perf_counter()
    draft_latent_time = 0.0
    verify_latent_time = 0.0
    
    if skip_latent_verification:
        # Only use draft model - skip target verification for speed
        input_ids_batch = input_ids.unsqueeze(0).to(device)
        attention_mask_batch = attention_mask.unsqueeze(0).to(device)
        position_ids_batch = position_ids.unsqueeze(0).to(device)
        
        draft_model.eval()
        draft_gen_start = time.perf_counter()
        with torch.no_grad():
            draft_outputs = draft_model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                position_ids=position_ids_batch,
                collect_latent_thoughts=True,
                latent_positions=[latent_positions],
            )
            draft_thoughts = [
                thought.to(device) if isinstance(thought, torch.Tensor) else torch.tensor(thought, device=device)
                for thought in (draft_outputs.latent_thoughts[0] if draft_outputs.latent_thoughts else [])
            ]
        draft_gen_end = time.perf_counter()
        
        # Accept all draft thoughts without verification
        acceptance_list = [True] * min(len(draft_thoughts), num_latent_thoughts)
        target_thoughts = draft_thoughts[:num_latent_thoughts] if len(draft_thoughts) >= num_latent_thoughts else draft_thoughts
        
        stats['num_draft_calls'] = 1
        stats['num_target_calls'] = 0  # No target call for latent thoughts!
        
        # Time draft generation
        draft_latent_time = draft_gen_end - draft_gen_start
        verify_latent_time = 0.0  # No verification
    else:
        # Generate and verify with both models
        acceptance_list, draft_thoughts, target_thoughts, draft_latent_time, verify_latent_time = speculative_decode_latent_thoughts(
            draft_model,
            target_model,
            input_ids,
            attention_mask,
            position_ids,
            latent_positions,
            num_latent_thoughts=num_latent_thoughts,
            gamma=gamma,
            similarity_threshold=similarity_threshold,
            device=device,
        )
        # Count draft calls for latent thoughts (now just one call for all thoughts)
        stats['num_draft_calls'] = 1  # Draft model called once for all latent thoughts
        stats['num_target_calls'] = 1  # Target model called once for all latent thoughts (parallel verification)
    
    latent_end_time = time.perf_counter()
    latent_time = latent_end_time - latent_start_time
    
    stats['latent_accepted'] = sum(acceptance_list)
    stats['latent_total'] = len(acceptance_list)
    stats['latent_thought_time'] = latent_time
    stats['draft_latent_time'] = draft_latent_time
    stats['verify_latent_time'] = verify_latent_time
    
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
    
    # Generate tokens: either speculative decoding or normal autoregressive
    token_start_time = time.perf_counter()
    if tokens_speculative:
        # Use speculative decoding for tokens
        generated_tokens, num_draft_tokens, num_target_tokens, tokens_accepted, tokens_total = speculative_decode_tokens(
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
        stats['num_draft_calls'] += num_draft_tokens
        stats['num_target_calls'] += num_target_tokens
        stats['tokens_accepted'] = tokens_accepted
        stats['tokens_total'] = tokens_total
    else:
        # Use normal autoregressive generation with target model only
        generated_tokens = autoregressive_token_generation(
            target_model,
            token_input_ids,
            token_attention_mask,
            token_position_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
        )
        # Count target model calls: one per token generated
        stats['num_target_calls'] += len(generated_tokens)
        stats['tokens_accepted'] = len(generated_tokens)  # All tokens are "accepted" (no rejection in autoregressive)
        stats['tokens_total'] = len(generated_tokens)
    token_end_time = time.perf_counter()
    token_time = token_end_time - token_start_time
    stats['token_generation_time'] = token_time
    
    # Record total wallclock time
    end_time = time.perf_counter()
    stats['wallclock_time'] = end_time - start_time
    
    return generated_tokens, stats


def baseline_autoregressive_decode(
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    latent_positions: List[int],
    num_latent_thoughts: int = 6,
    max_new_tokens: int = 50,
    eos_token_id: int = None,
    device: torch.device = None,
) -> Tuple[List[int], float, float]:
    """
    Baseline autoregressive generation using only the target model.
    Generates latent thought vectors first, then generates tokens autoregressively.
    Tracks separate wall clock times for each phase.
    
    Args:
        target_model: Target (Coconut) model instance
        input_ids: Input sequence [seq_len]
        attention_mask: Attention mask [seq_len]
        position_ids: Position IDs [seq_len]
        latent_positions: Positions of latent tokens
        num_latent_thoughts: Number of latent thoughts (default 6)
        max_new_tokens: Maximum new tokens to generate
        eos_token_id: EOS token ID
        device: Device to run on
    
    Returns:
        (generated_tokens, latent_thought_time, token_generation_time)
        - latent_thought_time: Time to generate latent thought vectors
        - token_generation_time: Time to generate tokens autoregressively
    """
    if device is None:
        device = input_ids.device
    
    target_model.eval()
    
    # Expand to batch dimension
    input_ids_batch = input_ids.unsqueeze(0).to(device)
    attention_mask_batch = attention_mask.unsqueeze(0).to(device)
    position_ids_batch = position_ids.unsqueeze(0).to(device)
    
    # Step 1: Generate latent thought vectors and time it
    latent_start_time = time.perf_counter()
    
    dummy_labels = torch.full(
        (input_ids_batch.shape[0], input_ids_batch.shape[1]),
        -100,
        dtype=torch.long,
        device=device
    )
    
    # Forward pass through Coconut model to generate latent thoughts
    with torch.no_grad():
        outputs = target_model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            position_ids=position_ids_batch,
            labels=dummy_labels,
            collect_latent_thoughts=True,  # Collect latent thoughts
        )
    
    latent_end_time = time.perf_counter()
    latent_thought_time = latent_end_time - latent_start_time
    
    # Get the position after latent tokens
    if len(latent_positions) > 0:
        last_latent_pos = max(latent_positions)
        start_pos = last_latent_pos + 1
    else:
        start_pos = len(input_ids)
    
    # Create sequence from start of input to position after latent tokens
    if start_pos < len(input_ids):
        current_input_ids = input_ids[:start_pos].clone().to(device)
        current_attention_mask = attention_mask[:start_pos].clone().to(device)
        current_position_ids = position_ids[:start_pos].clone().to(device)
    else:
        current_input_ids = input_ids.clone().to(device)
        current_attention_mask = attention_mask.clone().to(device)
        current_position_ids = position_ids.clone().to(device)
    
    # Expand to batch dimension for generation
    current_input_ids = current_input_ids.unsqueeze(0)
    current_attention_mask = current_attention_mask.unsqueeze(0)
    current_position_ids = current_position_ids.unsqueeze(0)
    
    # Step 2: Generate tokens autoregressively and time it
    token_start_time = time.perf_counter()
    
    generated_tokens = []
    
    # Autoregressive generation
    for _ in range(max_new_tokens):
        with torch.no_grad():
            dummy_labels_gen = torch.full(
                (current_input_ids.shape[0], current_input_ids.shape[1]),
                -100,
                dtype=torch.long,
                device=device
            )
            outputs = target_model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                labels=dummy_labels_gen,
                collect_latent_thoughts=False,
            )
            
            # Get next token (greedy decoding)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated_tokens.append(next_token)
            
            # Check for EOS
            if eos_token_id is not None and next_token == eos_token_id:
                break
            
            # Append to input for next iteration
            current_input_ids = torch.cat([
                current_input_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((1, 1), device=device, dtype=torch.long)
            ], dim=1)
            current_position_ids = torch.cat([
                current_position_ids,
                torch.tensor([[current_position_ids[0, -1].item() + 1]], device=device)
            ], dim=1)
    
    token_end_time = time.perf_counter()
    token_generation_time = token_end_time - token_start_time
    
    return generated_tokens, latent_thought_time, token_generation_time
