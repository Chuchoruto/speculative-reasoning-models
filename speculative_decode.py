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
        draft_vector: Draft model's latent thought [hidden_dim]
        target_vector: Target model's latent thought [hidden_dim]
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
    target_hidden_dim: int = None,  # Auto-detect if None
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
        target_hidden_dim: Target model hidden dimension (auto-detect if None)
    
    Returns:
        (acceptance_list, draft_thoughts, target_thoughts, draft_time, verify_time, draft_time_per_thought, verify_time_per_thought, logits)
        - draft_time: Total time taken to generate all draft latent thoughts (6 sequential passes)
        - verify_time: Total time taken to verify all draft thoughts with target model (1 parallel pass) - NOT divided
        - draft_time_per_thought: Average time per individual latent thought (draft model)
        - verify_time_per_thought: Total time for target model verification (same as verify_time, kept for consistency)
        - logits: Logits from target model verification [batch_size, seq_len, vocab_size] or None
    """
    if device is None:
        device = input_ids.device
    
    # Get target model's hidden dimension (auto-detect if not provided)
    if target_hidden_dim is None:
        # Try to get from model config
        if hasattr(target_model, 'base_causallm') and hasattr(target_model.base_causallm, 'config'):
            target_hidden_dim = target_model.base_causallm.config.n_embd
        elif hasattr(target_model, 'config'):
            target_hidden_dim = target_model.config.n_embd
        else:
            # Fallback: infer from embedding dimension
            target_hidden_dim = target_model.embedding.weight.shape[1]
    
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
    # NOTE: Internally, the draft model does 6 sequential forward passes (one per latent token)
    # This is slower than a single parallel pass, even with a smaller model
    all_draft_thoughts = []
    
    # Time the draft generation
    # IMPORTANT: This should ONLY call the draft model (GPT2-mini), NOT the target model
    # The draft model forward pass should be the same speed regardless of target model size
    # This timing includes ALL 6 sequential forward passes through the draft model
    draft_start_time = time.perf_counter()
    with torch.no_grad():
        # Call draft model once - it processes all latent positions sequentially internally
        # Each latent thought is generated autoregressively (depends on previous ones)
        # This ONLY uses the draft model (GPT2-mini base), not the target model
        # Internally: Does 6 sequential forward passes (one per latent token position)
        draft_outputs = draft_model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            position_ids=position_ids_batch,
            collect_latent_thoughts=True,
            latent_positions=[sorted_latent_positions],  # All positions at once
        )
        
        # Ensure GPU synchronization for accurate timing
        # This sync happens AFTER draft model forward pass, BEFORE any target model calls
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        
        # Extract all draft latent thoughts (already projected to target_hidden_dim)
        # This extraction happens within the draft timing window (before verify_start_time)
        if draft_outputs.latent_thoughts and len(draft_outputs.latent_thoughts[0]) > 0:
            all_draft_thoughts = [
                thought.to(device) if isinstance(thought, torch.Tensor) else torch.tensor(thought, device=device)
                for thought in draft_outputs.latent_thoughts[0]
            ]
    draft_end_time = time.perf_counter()
    draft_time = draft_end_time - draft_start_time
    # draft_time includes 6 sequential forward passes through the draft model (GPT2-mini)
    # This is slower than a single parallel pass, even with a larger model
    # Calculate per-unit time: divide by number of latent thoughts generated
    num_draft_thoughts_generated = len(all_draft_thoughts) if all_draft_thoughts else num_latent_thoughts
    draft_time_per_thought = draft_time / num_draft_thoughts_generated if num_draft_thoughts_generated > 0 else 0.0
    
    # Step 2: Construct input sequence with ALL draft latent thoughts embedded
    # Replace latent token positions with draft latent thoughts for verification
    # NOTE: Verification uses the TARGET model (target_model.base_causallm), NOT the draft model
    # This is a single parallel forward pass, which is faster than 6 sequential passes
    verify_start_time = time.perf_counter()
    with torch.no_grad():
        # Get base embeddings from target model
        base_embeddings = target_model.embedding(input_ids_batch)
        
        # Replace latent token positions with draft latent thoughts (target_hidden_dim-dim)
        for i, latent_pos in enumerate(sorted_latent_positions):
            if i < len(all_draft_thoughts):
                # Ensure draft thought has correct dimension
                draft_thought = all_draft_thoughts[i]
                if isinstance(draft_thought, torch.Tensor):
                    draft_thought = draft_thought.to(device)
                    if draft_thought.dim() > 1:
                        draft_thought = draft_thought.flatten()
                    # Ensure correct dimension
                    if draft_thought.shape[0] != target_hidden_dim:
                        if draft_thought.shape[0] > target_hidden_dim:
                            draft_thought = draft_thought[:target_hidden_dim]
                        else:
                            padding = torch.zeros(target_hidden_dim - draft_thought.shape[0], device=device)
                            draft_thought = torch.cat([draft_thought, padding])
                else:
                    draft_thought = torch.tensor(draft_thought, device=device)
                    if draft_thought.shape[0] != target_hidden_dim:
                        if draft_thought.shape[0] > target_hidden_dim:
                            draft_thought = draft_thought[:target_hidden_dim]
                        else:
                            padding = torch.zeros(target_hidden_dim - draft_thought.shape[0], device=device)
                            draft_thought = torch.cat([draft_thought, padding])
                base_embeddings[0, latent_pos, :] = draft_thought
        
        # Step 3: Call TARGET model's base_causallm in ONE forward pass to get all hidden states
        # This processes: question_input_seq -> draft_thought_1 -> draft_thought_2 -> ... -> draft_thought_6
        # We extract hidden states at positions right BEFORE each draft thought (after processing up to that point)
        # The hidden state at position i-1 represents the "verified thought" for draft_thought at position i
        # IMPORTANT: This uses the TARGET model (larger), but in a SINGLE parallel pass
        # This is why verification can be faster than draft (1 pass vs 6 sequential passes)
        
        # Get the base causal LM from Coconut (TARGET model, not draft)
        base_model = target_model.base_causallm  # This is the TARGET model (GPT2 or GPT2-medium)
        
        # Process entire sequence in ONE forward pass with output_hidden_states=True
        # This is a single parallel pass, processing all draft thoughts at once
        outputs = base_model(
            inputs_embeds=base_embeddings,
            attention_mask=attention_mask_batch,
            position_ids=position_ids_batch,
            output_hidden_states=True,  # Get all hidden states
        )
        
        # Ensure GPU synchronization for accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        
        # Extract hidden states from the last layer: [batch_size, seq_len, hidden_dim]
        # hidden_states[-1] is the last layer's hidden states
        all_hidden_states = outputs.hidden_states[-1]  # [1, seq_len, target_hidden_dim]
        
        # Extract verified thought vectors right BEFORE each draft thought position
        # The hidden state at position latent_pos - 1 represents the model's output AFTER processing
        # everything up to (but not including) the draft thought at latent_pos
        # This is the "verified thought" that represents what the model thinks should come at position latent_pos
        # We compare this with the draft thought embedded at position latent_pos
        all_target_thoughts = []
        for i, latent_pos in enumerate(sorted_latent_positions):
            # The verified thought is the hidden state at position latent_pos - 1
            # (after processing sequence up to latent_pos, before processing draft_thought at latent_pos)
            # For the first draft thought (latent_pos=0), use position 0
            extract_pos = max(0, latent_pos - 1)
            
            if extract_pos < all_hidden_states.shape[1]:
                # Extract hidden state: [target_hidden_dim]
                verified_thought = all_hidden_states[0, extract_pos, :].clone()
                all_target_thoughts.append(verified_thought)
            else:
                # Fallback: use last hidden state if position is out of bounds
                verified_thought = all_hidden_states[0, -1, :].clone()
                all_target_thoughts.append(verified_thought)
        
        # Ensure we have the right number of thoughts
        if len(all_target_thoughts) != len(sorted_latent_positions):
            # Pad or truncate to match expected count
            while len(all_target_thoughts) < len(sorted_latent_positions):
                all_target_thoughts.append(torch.zeros(target_hidden_dim, device=device))
            all_target_thoughts = all_target_thoughts[:len(sorted_latent_positions)]
    verify_end_time = time.perf_counter()
    verify_time = verify_end_time - verify_start_time
    # verify_time is for a single parallel forward pass that verifies all draft thoughts
    # This is the total time for the parallel verification pass (not divided)
    verify_time_per_thought = verify_time  # Keep as total time, not per-unit
    
    # Step 4: Verify all latent thoughts in parallel (vectorized)
    min_len = min(len(all_draft_thoughts), len(all_target_thoughts), num_latent_thoughts)
    
    if min_len == 0:
        return [], [], [], draft_time, verify_time, None
    
    # Stack all draft and target thoughts into tensors for vectorized comparison
    # Shape: [min_len, target_hidden_dim]
    draft_stack = []
    target_stack = []
    
    for i in range(min_len):
        draft_vec = all_draft_thoughts[i]
        target_vec = all_target_thoughts[i]
        
        # Ensure both are tensors on the same device with correct shape [target_hidden_dim]
        if isinstance(draft_vec, torch.Tensor):
            draft_vec = draft_vec.to(device)
            if draft_vec.dim() > 1:
                draft_vec = draft_vec.flatten()
        else:
            draft_vec = torch.tensor(draft_vec, device=device)
        
        if isinstance(target_vec, torch.Tensor):
            target_vec = target_vec.to(device)
            if target_vec.dim() > 1:
                target_vec = target_vec.flatten()
        else:
            target_vec = torch.tensor(target_vec, device=device)
        
        # Ensure both are exactly [target_hidden_dim] for comparison
        if draft_vec.shape[0] != target_hidden_dim:
            if draft_vec.shape[0] > target_hidden_dim:
                draft_vec = draft_vec[:target_hidden_dim]
            else:
                padding = torch.zeros(target_hidden_dim - draft_vec.shape[0], device=device)
                draft_vec = torch.cat([draft_vec, padding])
        
        if target_vec.shape[0] != target_hidden_dim:
            if target_vec.shape[0] > target_hidden_dim:
                target_vec = target_vec[:target_hidden_dim]
            else:
                padding = torch.zeros(target_hidden_dim - target_vec.shape[0], device=device)
                target_vec = torch.cat([target_vec, padding])
        
        draft_stack.append(draft_vec)
        target_stack.append(target_vec)
    
    # Stack into tensors: [min_len, target_hidden_dim]
    draft_tensor = torch.stack(draft_stack, dim=0)  # [min_len, target_hidden_dim]
    target_tensor = torch.stack(target_stack, dim=0)  # [min_len, target_hidden_dim]
    
    # Vectorized cosine similarity computation
    # Normalize both tensors
    draft_norm = torch.nn.functional.normalize(draft_tensor, p=2, dim=1)  # [min_len, target_hidden_dim]
    target_norm = torch.nn.functional.normalize(target_tensor, p=2, dim=1)  # [min_len, target_hidden_dim]
    
    # Compute cosine similarities: [min_len]
    cosine_similarities = (draft_norm * target_norm).sum(dim=1)  # [min_len]
    
    # Vectorized acceptance check
    all_acceptance = (cosine_similarities >= similarity_threshold).cpu().tolist()
    
    # Return only the verified thoughts (slice to min_len)
    # Also return logits from the target model for extracting first token
    target_logits = outputs.logits if hasattr(outputs, 'logits') else None
    return all_acceptance, all_draft_thoughts[:min_len], all_target_thoughts[:min_len], draft_time, verify_time, draft_time_per_thought, verify_time_per_thought, target_logits


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
        (generated_tokens, num_draft_calls, num_target_calls, tokens_accepted, tokens_total, draft_token_time, target_verify_time, draft_time_per_token, target_time_per_token)
        - draft_token_time: Total time spent in draft model token generation
        - target_verify_time: Total time spent in target model token verification (NOT divided, total time for parallel verification)
        - draft_time_per_token: Average time per individual token (draft model)
        - target_time_per_token: Total time for target model token verification (same as target_verify_time, kept for consistency)
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
    
    # Track timing separately for draft and target
    total_draft_token_time = 0.0
    total_target_verify_time = 0.0
    
    while len(generated_tokens) < max_new_tokens:
        # Draft model generates gamma tokens sequentially
        draft_tokens = []
        draft_logits_list = []
        
        draft_input = current_input_ids.unsqueeze(0)
        draft_attn = current_attention_mask.unsqueeze(0)
        draft_pos = current_position_ids.unsqueeze(0)
        
        # Time draft token generation - time each individual token
        draft_token_times_this_round = []
        # Generate all gamma draft tokens
        # Track whether we stopped early due to drafting EOS
        draft_reached_eos = False

        for _ in range(gamma):
            num_draft_calls += 1
            draft_token_start = time.perf_counter()
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
            
            # Ensure GPU synchronization for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            draft_token_end = time.perf_counter()
            draft_token_times_this_round.append(draft_token_end - draft_token_start)

            # Stop drafting further tokens in this round if EOS is generated.
            # Once <|endoftext|> is proposed, any additional draft tokens would be discarded
            # regardless of whether the EOS token is ultimately accepted or rejected.
            if eos_token_id is not None and draft_token == eos_token_id:
                draft_reached_eos = True
                break
        
        total_draft_token_time += sum(draft_token_times_this_round)
        
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
        
        # Time target verification
        target_verify_start = time.perf_counter()
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
            
            # Ensure GPU synchronization for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
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
        target_verify_end = time.perf_counter()
        target_verify_time_this_round = target_verify_end - target_verify_start
        total_target_verify_time += target_verify_time_this_round
        
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
        
        # Calculate per-token time for this round: divide by number of tokens verified
        # Note: target verification is parallel, so we divide the total time by number of tokens
        num_tokens_verified_this_round = len(verified_tokens)
        
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
    
    # Calculate per-token average for draft model only
    # Target verification time is kept as total (not divided) since it's parallel verification
    draft_time_per_token = total_draft_token_time / tokens_total if tokens_total > 0 else 0.0
    target_time_per_token = total_target_verify_time  # Keep as total time, not per-unit
    
    return generated_tokens, num_draft_calls, num_target_calls, tokens_accepted, tokens_total, total_draft_token_time, total_target_verify_time, draft_time_per_token, target_time_per_token


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
    gamma_tokens: int = None,  # If None, uses gamma value
    max_new_tokens: int = 50,
    eos_token_id: int = None,
    similarity_threshold: float = 0.9,
    device: torch.device = None,
    rng: Optional[np.random.Generator] = None,
    tokens_speculative: bool = False,
    skip_latent_verification: bool = False,
    target_hidden_dim: int = None,  # Auto-detect if None
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
        gamma_tokens: Number of draft tokens per round for token generation. If None, uses gamma value (for backward compatibility)
        max_new_tokens: Maximum new tokens to generate
        eos_token_id: EOS token ID
        similarity_threshold: Cosine similarity threshold for latent thoughts
        device: Device to run on
        rng: Random number generator
        tokens_speculative: If True, use speculative decoding for tokens. If False, use autoregressive generation with target model only.
        skip_latent_verification: If True, skip target model verification of latent thoughts (use draft only). Much faster but less accurate.
        target_hidden_dim: Target model hidden dimension (auto-detect if None)
    
    Returns:
        (generated_tokens, stats_dict)
    """
    # Use gamma_tokens if provided, otherwise fall back to gamma for backward compatibility
    if gamma_tokens is None:
        gamma_tokens = gamma
    if device is None:
        device = input_ids.device
    
    # Get target model's hidden dimension (auto-detect if not provided)
    if target_hidden_dim is None:
        # Try to get from model config
        if hasattr(target_model, 'base_causallm') and hasattr(target_model.base_causallm, 'config'):
            target_hidden_dim = target_model.base_causallm.config.n_embd
        elif hasattr(target_model, 'config'):
            target_hidden_dim = target_model.config.n_embd
        else:
            # Fallback: infer from embedding dimension
            target_hidden_dim = target_model.embedding.weight.shape[1]
    
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
        verification_logits = None  # No verification, so no logits
    else:
        # Generate and verify with both models
        acceptance_list, draft_thoughts, target_thoughts, draft_latent_time, verify_latent_time, draft_time_per_thought, verify_time_per_thought, verification_logits = speculative_decode_latent_thoughts(
            draft_model,
            target_model,
            input_ids,
            attention_mask,
            position_ids,
            latent_positions,
            num_latent_thoughts=num_latent_thoughts,
            gamma=gamma_tokens,
            similarity_threshold=similarity_threshold,
            device=device,
            target_hidden_dim=target_hidden_dim,
        )
        # Count draft calls for latent thoughts (now just one call for all thoughts)
        stats['num_draft_calls'] = 1  # Draft model called once for all latent thoughts
        stats['num_target_calls'] = 1  # Target model called once for all latent thoughts (parallel verification)
        # Note: This counts as 1 call per sample, even though Coconut processes 6 latent tokens internally.
        # The parallel verification strategy allows us to verify all 6 latent thoughts in a single forward pass.
    
    latent_end_time = time.perf_counter()
    # Use the sum of draft and verify times for consistency (more accurate than wallclock diff)
    # The wallclock diff might include tiny overhead, but draft+verify is what we actually measured
    latent_time = draft_latent_time + verify_latent_time
    
    stats['latent_accepted'] = sum(acceptance_list)
    stats['latent_total'] = len(acceptance_list)
    stats['latent_thought_time'] = latent_time
    stats['draft_latent_time'] = draft_latent_time
    stats['verify_latent_time'] = verify_latent_time
    stats['draft_time_per_thought'] = draft_time_per_thought
    stats['verify_time_per_thought'] = verify_time_per_thought
    
    # If not all latent thoughts accepted, we might need to handle this
    # For now, continue with token generation regardless
    
    # Step 2: Generate tokens after latent thoughts
    # Find the position after latent tokens (after <end-latent> if present, or after last latent)
    if len(latent_positions) > 0:
        # Get position after latent tokens
        # Assuming sequence structure: ... <start-latent> <latent>*6 <end-latent> ...
        # We want to start from after <end-latent> or after the last latent token
        last_latent_pos = max(latent_positions)
        start_pos = last_latent_pos + 1  # Position of <end-latent> token
    else:
        start_pos = len(input_ids)
    
    # Extract the first token from latent verification logits (if available)
    # This token was generated "for free" during latent verification
    generated_tokens = []
    if not skip_latent_verification and verification_logits is not None and start_pos < verification_logits.shape[1]:
        # Extract logits for the first token after <end-latent>
        # logits[start_pos] predicts the token at position start_pos + 1
        first_token_logits = verification_logits[0, start_pos, :]  # [vocab_size]
        first_token = torch.argmax(first_token_logits).item()
        generated_tokens.append(first_token)
    
    # Create sequence from start of input to position after latent tokens (or after first token if extracted)
    if len(generated_tokens) > 0:
        # We already extracted the first token, so start from after it
        if start_pos + 1 < len(input_ids):
            token_input_ids = input_ids[:start_pos + 2].clone()  # Include first token if in input
            token_attention_mask = attention_mask[:start_pos + 2].clone()
            token_position_ids = position_ids[:start_pos + 2].clone()
        else:
            # First token is beyond input_ids, append it
            token_input_ids = torch.cat([input_ids.clone(), torch.tensor([generated_tokens[0]], device=input_ids.device)], dim=0)
            token_attention_mask = torch.cat([attention_mask.clone(), torch.ones(1, device=input_ids.device, dtype=torch.long)], dim=0)
            token_position_ids = torch.cat([position_ids.clone(), torch.tensor([position_ids[-1].item() + 1], device=input_ids.device)], dim=0)
    else:
        # No first token extracted, start from <end-latent>
        if start_pos < len(input_ids):
            token_input_ids = input_ids[:start_pos + 1].clone()  # Include <end-latent>
            token_attention_mask = attention_mask[:start_pos + 1].clone()
            token_position_ids = position_ids[:start_pos + 1].clone()
        else:
            # If we're already at the end, use full sequence
            token_input_ids = input_ids.clone()
            token_attention_mask = attention_mask.clone()
            token_position_ids = position_ids.clone()
    
    # Generate tokens: either speculative decoding or normal autoregressive
    token_start_time = time.perf_counter()
    if tokens_speculative:
        # Use speculative decoding for tokens
        additional_tokens, num_draft_tokens, num_target_tokens, tokens_accepted, tokens_total, draft_token_time, target_verify_time, draft_time_per_token, target_time_per_token = speculative_decode_tokens(
            draft_model,
            target_model,
            token_input_ids,
            token_attention_mask,
            token_position_ids,
            gamma=gamma_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
            rng=rng,
        )
        # Prepend the first token (if extracted) to the generated tokens
        generated_tokens = generated_tokens + additional_tokens
        stats['num_draft_calls'] += num_draft_tokens
        stats['num_target_calls'] += num_target_tokens
        stats['tokens_accepted'] = tokens_accepted
        stats['tokens_total'] = tokens_total
        stats['draft_token_time'] = draft_token_time
        stats['target_verify_token_time'] = target_verify_time
        stats['draft_time_per_token'] = draft_time_per_token
        stats['target_time_per_token'] = target_time_per_token
    else:
        # Use normal autoregressive generation with target model only
        additional_tokens = autoregressive_token_generation(
            target_model,
            token_input_ids,
            token_attention_mask,
            token_position_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
        )
        # Prepend the first token (if extracted) to the generated tokens
        generated_tokens = generated_tokens + additional_tokens
        # Count target model calls: one per token generated
        # Note: stats['num_target_calls'] already includes 1 call from latent verification (if not skipped)
        # Here we add the token generation calls (one per token)
        stats['num_target_calls'] += len(additional_tokens)
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
) -> Tuple[List[int], float, float, float, float]:
    """
    Baseline autoregressive generation using only the target model.
    Generates latent thought vectors first, then generates tokens autoregressively.
    Tracks separate wall clock times for each phase and per-unit times.
    
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
        (generated_tokens, latent_thought_time, token_generation_time, latent_time_per_thought, token_time_per_token)
        - latent_thought_time: Total time to generate latent thought vectors (single parallel pass)
        - token_generation_time: Total time to generate tokens autoregressively
        - latent_time_per_thought: Average time per individual latent thought (latent_thought_time / num_latent_thoughts)
        - token_time_per_token: Average time per individual token (token_generation_time / num_tokens_generated)
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
        
        # Ensure GPU synchronization for accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    
    latent_end_time = time.perf_counter()
    latent_thought_time = latent_end_time - latent_start_time
    # Calculate per-thought time: divide total time by number of latent thoughts
    # Note: This is a single parallel pass, so we divide to get per-unit time
    latent_time_per_thought = latent_thought_time / num_latent_thoughts if num_latent_thoughts > 0 else 0.0
    
    # Get the position after latent tokens
    if len(latent_positions) > 0:
        last_latent_pos = max(latent_positions)
        start_pos = last_latent_pos + 1  # Position of <end-latent> token
        # The first token after <end-latent> is at position start_pos + 1
        # Logits[i] predicts token[i+1], so logits[start_pos] predicts token at start_pos+1
        first_token_pos = start_pos + 1
    else:
        start_pos = len(input_ids)
        first_token_pos = len(input_ids)
    
    # Extract the first token from the logits returned during latent thought processing
    # This token was generated "for free" during latent processing
    generated_tokens = []
    if first_token_pos < outputs.logits.shape[1]:
        # Extract logits for the first token after <end-latent>
        # logits[start_pos] predicts the token at position start_pos + 1
        first_token_logits = outputs.logits[0, start_pos, :]  # [vocab_size]
        first_token = torch.argmax(first_token_logits).item()
        generated_tokens.append(first_token)
        
        # Create sequence from start of input to position after first token
        # This includes the first token we just extracted
        if first_token_pos < len(input_ids):
            current_input_ids = input_ids[:first_token_pos + 1].clone().to(device)
            current_attention_mask = attention_mask[:first_token_pos + 1].clone().to(device)
            current_position_ids = position_ids[:first_token_pos + 1].clone().to(device)
        else:
            # First token is beyond input_ids, append it
            current_input_ids = torch.cat([input_ids.clone().to(device), torch.tensor([first_token], device=device)], dim=0)
            current_attention_mask = torch.cat([attention_mask.clone().to(device), torch.ones(1, device=device, dtype=torch.long)], dim=0)
            current_position_ids = torch.cat([position_ids.clone().to(device), torch.tensor([position_ids[-1].item() + 1], device=device)], dim=0)
    else:
        # No token generated during latent processing, start from <end-latent>
        if start_pos < len(input_ids):
            current_input_ids = input_ids[:start_pos + 1].clone().to(device)
            current_attention_mask = attention_mask[:start_pos + 1].clone().to(device)
            current_position_ids = position_ids[:start_pos + 1].clone().to(device)
        else:
            current_input_ids = input_ids.clone().to(device)
            current_attention_mask = attention_mask.clone().to(device)
            current_position_ids = position_ids.clone().to(device)
    
    # Expand to batch dimension for generation
    current_input_ids = current_input_ids.unsqueeze(0)
    current_attention_mask = current_attention_mask.unsqueeze(0)
    current_position_ids = current_position_ids.unsqueeze(0)
    
    # Step 2: Generate remaining tokens autoregressively and time it
    token_start_time = time.perf_counter()
    
    # Track per-token times
    token_times = []
    
    # Autoregressive generation
    for _ in range(max_new_tokens):
        token_iter_start = time.perf_counter()
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
            
            # Ensure GPU synchronization for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
            # Get next token (greedy decoding)
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            generated_tokens.append(next_token)
            
            token_iter_end = time.perf_counter()
            token_times.append(token_iter_end - token_iter_start)
            
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
    
    # Calculate per-token time: average of individual token times
    num_tokens_generated = len(token_times)
    token_time_per_token = sum(token_times) / num_tokens_generated if num_tokens_generated > 0 else 0.0
    
    return generated_tokens, latent_thought_time, token_generation_time, latent_time_per_thought, token_time_per_token
