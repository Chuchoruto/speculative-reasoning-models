"""
Modal script for running speculative decoding evaluation.
Loads draft model and target (Coconut) model from checkpoints and evaluates on test set.
"""

import modal
import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = modal.App("speculative-decoding-eval")

# Create image with dependencies
image = (
    modal.Image
    .debian_slim()
    .pip_install([
        "torch==2.5.1",
        "numpy==2.1.3",
        "transformers==4.46.2",
        "datasets==3.1.0",
        "tqdm==4.67.0",
        "pyyaml",
    ])
    .env({
        "CUDA_VISIBLE_DEVICES": "0"
    })
    .add_local_file("speculative_decode.py", "/workspace/speculative_decode.py")
    .add_local_file("draft_model.py", "/workspace/draft_model.py")
    .add_local_file("coconut.py", "/workspace/coconut.py")
    .add_local_file("dataset.py", "/workspace/dataset.py")
    .add_local_dir("data", "/workspace/data")
)

# Use the same persistent volume
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100:1",
    timeout=60 * 60 * 4,  # 4 hours
    volumes={"/checkpoints": checkpoint_volume},
)
def evaluate_speculative_decoding(
    draft_checkpoint_path: str,
    target_checkpoint_path: str,
    data_path: str = "data/gsm_test.json",
    num_latent_thoughts: int = 6,
    gamma: int = 6,
    max_new_tokens: int = 50,
    similarity_threshold: float = 0.9,
    max_samples: int = 100,
    clock_run: str = "False",
    record_output: str = "False",
    baseline_only: str = "False",
    tokens_speculative: str = "False",
    skip_latent_verification: str = "False",
):
    """
    Evaluate speculative decoding with draft and target models.
    
    Args:
        draft_checkpoint_path: Path to draft model checkpoint in Modal volume
        target_checkpoint_path: Path to target (Coconut) model checkpoint
        data_path: Path to test data JSON file
        num_latent_thoughts: Number of latent thoughts to generate (default 6)
        gamma: Number of draft tokens per round (default 4)
        max_new_tokens: Maximum new tokens to generate (default 50)
        similarity_threshold: Cosine similarity threshold for latent thoughts (default 0.9)
        max_samples: Maximum number of samples to evaluate (default 100)
        clock_run: Enable wallclock timing. Accepts "True", "true", "1", "False", "false", "0" (default "False")
        record_output: Record generated outputs for comparison. Requires clock_run=True. Accepts "True", "true", "1", etc. (default "False")
        baseline_only: Only run baseline (target model) decoding, skip speculative decoding. Accepts "True", "true", "1", etc. (default "False")
        tokens_speculative: If True, use speculative decoding for tokens. If False, use autoregressive generation with target model only after latent thoughts. Accepts "True", "true", "1", etc. (default "False")
        skip_latent_verification: If True, skip target model verification of latent thoughts (use draft only). Much faster. Accepts "True", "true", "1", etc. (default "False")
    """
    import json
    import sys
    import numpy as np
    import os
    from tqdm import tqdm
    
    # Parse string parameters to boolean
    clock_run_bool = clock_run.lower() in ("true", "1", "yes", "on")
    record_output_bool = record_output.lower() in ("true", "1", "yes", "on")
    baseline_only_bool = baseline_only.lower() in ("true", "1", "yes", "on")
    tokens_speculative_bool = tokens_speculative.lower() in ("true", "1", "yes", "on")
    skip_latent_verification_bool = skip_latent_verification.lower() in ("true", "1", "yes", "on")
    
    # If baseline_only is True, clock_run must also be True
    if baseline_only_bool:
        clock_run_bool = True
        print("⚠️  Note: baseline_only=True implies clock_run=True")
    
    # If tokens_speculative is False, we need baseline comparison, so enable clock_run
    if not tokens_speculative_bool:
        clock_run_bool = True
        print("⚠️  Note: tokens_speculative=False implies clock_run=True (for baseline comparison)")
    
    if record_output_bool and not clock_run_bool:
        print("⚠️  Warning: record_output requires clock_run=True. Disabling record_output.")
        record_output_bool = False

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    
    from dataset import get_dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add Coconut tokens
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    eos_token_id = tokenizer.eos_token_id
    
    # Load target (Coconut) model
    print(f"Loading target model from {target_checkpoint_path}...")
    target_base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    target_base_model.resize_token_embeddings(len(tokenizer))
    
    from coconut import Coconut
    target_model = Coconut(
        target_base_model,
        latent_id,
        start_id,
        end_id,
        eos_token_id,
    )
    
    # Load target checkpoint
    checkpoint = torch.load(target_checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        target_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        target_model.load_state_dict(checkpoint)
    target_model = target_model.to(device)
    print("Target model loaded.")
    
    # Load draft model (only if not baseline_only)
    draft_model = None
    if not baseline_only_bool:
        print(f"Loading draft model from {draft_checkpoint_path}...")
        draft_base_model = AutoModelForCausalLM.from_pretrained("erwanf/gpt2-mini")
        draft_base_model.resize_token_embeddings(len(tokenizer))
        
        from draft_model import DraftModel
        draft_model = DraftModel(
            draft_base_model,
            latent_id,
            start_id,
            end_id,
            eos_token_id,
            hidden_dim_base=512,
            hidden_dim_target=768,
        )
        
        # Load draft checkpoint
        checkpoint = torch.load(draft_checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            draft_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            draft_model.load_state_dict(checkpoint)
        draft_model = draft_model.to(device)
        print("Draft model loaded.")
    else:
        print("⚠️  Skipping draft model loading (baseline_only=True)")
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = get_dataset(data_path, tokenizer, max_size=max_samples)
    print(f"Loaded {len(dataset)} samples")
    
    # Import speculative decoding
    from speculative_decode import speculative_decode, baseline_autoregressive_decode
    
    # Evaluation metrics
    total_latent_accepted = 0
    total_latent_total = 0
    total_draft_calls = 0
    total_target_calls = 0
    total_tokens_accepted = 0
    total_tokens_total = 0
    total_generated_tokens = 0
    num_samples = 0
    
    # Timing metrics
    total_speculative_time = 0.0
    total_baseline_time = 0.0
    total_speculative_latent_time = 0.0
    total_draft_latent_time = 0.0
    total_verify_latent_time = 0.0
    total_speculative_token_time = 0.0
    total_baseline_latent_time = 0.0
    total_baseline_token_time = 0.0
    per_sample_speculative_times = []
    per_sample_baseline_times = []
    per_sample_speculative_latent_times = []
    per_sample_speculative_token_times = []
    per_sample_baseline_latent_times = []
    per_sample_baseline_token_times = []
    exact_matches = 0
    total_comparisons = 0
    
    # Output recording (only used if record_output=True and clock_run=True)
    output_records = []
    if record_output_bool:
        # Load original data to get question text
        original_data = json.load(open(data_path))[:max_samples]
    
    # Evaluate each sample
    if baseline_only_bool:
        print("\nStarting baseline (target-only) evaluation...")
    else:
        print("\nStarting speculative decoding evaluation...")
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Convert to list or iterate correctly
    # HuggingFace Dataset iteration returns dicts directly
    dataset_samples = [dataset[i] for i in range(min(len(dataset), max_samples))]
    
    for idx, sample in enumerate(tqdm(dataset_samples, desc="Evaluating")):
        # Construct input with latent tokens
        # sample is a dict from HuggingFace Dataset
        question_tokens = sample["question_tokenized"]
        
        # Create input with latent thoughts
        input_tokens = (
            question_tokens
            + [start_id]
            + [latent_id] * num_latent_thoughts
            + [end_id]
        )
        
        # Convert to tensors
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        
        # Get latent positions
        latent_positions = [
            i for i, token_id in enumerate(input_tokens) 
            if token_id == latent_id
        ]
        
        try:
            generated_tokens = None
            
            # Run speculative decoding (unless baseline_only)
            if not baseline_only_bool:
                # Run speculative decoding (latent thoughts always speculative, tokens conditional)
                generated_tokens, stats = speculative_decode(
                    draft_model,
                    target_model,
                    input_ids,
                    attention_mask,
                    position_ids,
                    latent_positions,
                    num_latent_thoughts=num_latent_thoughts,
                    gamma=gamma,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                    similarity_threshold=similarity_threshold,
                    device=device,
                    rng=rng,
                    tokens_speculative=tokens_speculative_bool,
                    skip_latent_verification=skip_latent_verification_bool,
                )
                
                # Accumulate statistics
                total_latent_accepted += stats['latent_accepted']
                total_latent_total += stats['latent_total']
                total_draft_calls += stats['num_draft_calls']
                total_target_calls += stats['num_target_calls']
                total_tokens_accepted += stats.get('tokens_accepted', 0)
                total_tokens_total += stats.get('tokens_total', 0)
                total_generated_tokens += len(generated_tokens)
                
                # Track timing (always track if we have timing data)
                speculative_time = stats.get('wallclock_time', 0.0)
                speculative_latent_time = stats.get('latent_thought_time', 0.0)
                draft_latent_time = stats.get('draft_latent_time', 0.0)
                verify_latent_time = stats.get('verify_latent_time', 0.0)
                speculative_token_time = stats.get('token_generation_time', 0.0)
                total_speculative_time += speculative_time
                total_speculative_latent_time += speculative_latent_time
                total_draft_latent_time += draft_latent_time
                total_verify_latent_time += verify_latent_time
                total_speculative_token_time += speculative_token_time
                per_sample_speculative_times.append(speculative_time)
                per_sample_speculative_latent_times.append(speculative_latent_time)
                per_sample_speculative_token_times.append(speculative_token_time)
                
                # Increment sample count for speculative decoding
                num_samples += 1
            
            # Always run baseline autoregressive decoding (for baseline_only or for comparison)
            # If tokens_speculative=False, we need baseline to compare outputs and timing
            if clock_run_bool or baseline_only_bool or not tokens_speculative_bool:
                baseline_tokens, latent_thought_time, token_generation_time = baseline_autoregressive_decode(
                    target_model,
                    input_ids,
                    attention_mask,
                    position_ids,
                    latent_positions,
                    num_latent_thoughts=num_latent_thoughts,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                    device=device,
                )
                baseline_time = latent_thought_time + token_generation_time
                total_baseline_time += baseline_time
                total_baseline_latent_time += latent_thought_time
                total_baseline_token_time += token_generation_time
                per_sample_baseline_times.append(baseline_time)
                per_sample_baseline_latent_times.append(latent_thought_time)
                per_sample_baseline_token_times.append(token_generation_time)
                
                # Compare outputs if we have both
                if generated_tokens is not None and baseline_tokens is not None:
                    total_comparisons += 1
                    if generated_tokens == baseline_tokens:
                        exact_matches += 1
                
                # For baseline_only, use baseline_tokens as the generated tokens
                if baseline_only_bool:
                    generated_tokens = baseline_tokens
                    total_generated_tokens += len(generated_tokens)
                    num_samples += 1
                else:
                    # For speculative decoding, increment happens below
                    pass
                
                # Record outputs if requested
                if record_output_bool and generated_tokens is not None:
                    # Decode tokens to text
                    spec_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
                    
                    # Get original question text from dataset
                    original_idx = sample.get("idx", idx)
                    if original_idx < len(original_data):
                        question_text = original_data[original_idx].get("question", "N/A")
                    else:
                        question_text = f"Sample {idx}"
                    
                    output_records.append({
                        "sample_idx": idx,
                        "question": question_text,
                        "output_spec_decode": spec_text,
                        "output_standard_decode": baseline_text,
                        "tokens_spec_decode": generated_tokens,
                        "tokens_standard_decode": baseline_tokens,
                        "exact_match": generated_tokens == baseline_tokens,
                    })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Print results
    print("\n" + "=" * 60)
    if baseline_only_bool:
        print("BASELINE (TARGET MODEL ONLY) EVALUATION RESULTS")
    else:
        print("SPECULATIVE DECODING EVALUATION RESULTS")
    print("=" * 60)
    print(f"Evaluated {num_samples} samples")
    
    # Latent thought acceptance (only if not baseline_only)
    if not baseline_only_bool:
        print(f"\nLatent Thought Acceptance:")
        if total_latent_total > 0:
            latent_accept_rate = total_latent_accepted / total_latent_total
            print(f"  Accepted: {total_latent_accepted}/{total_latent_total} ({latent_accept_rate:.2%})")
        else:
            latent_accept_rate = 0.0
        
        # Token acceptance
        print(f"\nToken Acceptance:")
        if total_tokens_total > 0:
            token_accept_rate = total_tokens_accepted / total_tokens_total
            print(f"  Accepted: {total_tokens_accepted}/{total_tokens_total} ({token_accept_rate:.2%})")
        else:
            token_accept_rate = 0.0
        
        # Model calls and speedup
        print(f"\nModel Calls:")
        print(f"  Draft model calls: {total_draft_calls}")
        print(f"  Target model calls: {total_target_calls}")
        print(f"  Total calls: {total_draft_calls + total_target_calls}")
        
        # Estimated speedup based on model calls
        baseline_target_calls = total_generated_tokens
        print(f"  Baseline (target-only) calls: {baseline_target_calls}")
        
        if baseline_target_calls > 0:
            # Estimated speedup = baseline_target_calls / total_target_calls
            # (with parallel verification, we should have far fewer target calls)
            estimated_speedup = baseline_target_calls / total_target_calls
            print(f"  Estimated speedup (based on calls): {estimated_speedup:.2f}x")
        else:
            estimated_speedup = 0.0
    else:
        latent_accept_rate = 0.0
        token_accept_rate = 0.0
        estimated_speedup = 0.0
    
    # Output comparison (if we ran both)
    if total_comparisons > 0:
        print(f"\nOutput Comparison:")
        match_rate = exact_matches / total_comparisons
        print(f"  Exact token matches: {exact_matches}/{total_comparisons} ({match_rate:.2%})")
        if match_rate < 1.0:
            print(f"  ⚠️  Warning: {total_comparisons - exact_matches} samples have mismatched outputs!")
    
    # Actual wallclock speedup (if baseline was run)
    if total_baseline_time > 0 and num_samples > 0:
        print(f"\nWallclock Time Comparison:")
        avg_speculative_time = total_speculative_time / num_samples if num_samples > 0 else 0.0
        avg_baseline_time = total_baseline_time / num_samples if num_samples > 0 else 0.0
        avg_spec_latent_time = total_speculative_latent_time / num_samples if num_samples > 0 else 0.0
        avg_spec_token_time = total_speculative_token_time / num_samples if num_samples > 0 else 0.0
        avg_base_latent_time = total_baseline_latent_time / num_samples if num_samples > 0 else 0.0
        avg_base_token_time = total_baseline_token_time / num_samples if num_samples > 0 else 0.0
        
        print(f"  Average speculative decoding time: {avg_speculative_time:.4f}s per sample")
        print(f"  Average baseline decoding time: {avg_baseline_time:.4f}s per sample")
        print(f"  Total speculative time: {total_speculative_time:.4f}s")
        print(f"  Total baseline time: {total_baseline_time:.4f}s")
        
        print(f"\n  Speculative Time Breakdown:")
        print(f"    Average latent thought generation time: {avg_spec_latent_time:.4f}s per sample")
        avg_draft_latent_time = total_draft_latent_time / num_samples if num_samples > 0 else 0.0
        avg_verify_latent_time = total_verify_latent_time / num_samples if num_samples > 0 else 0.0
        print(f"      - Draft latent thought time: {avg_draft_latent_time:.4f}s per sample ({total_draft_latent_time:.4f}s total)")
        print(f"      - Verify latent thought time: {avg_verify_latent_time:.4f}s per sample ({total_verify_latent_time:.4f}s total)")
        print(f"    Average token generation time: {avg_spec_token_time:.4f}s per sample")
        print(f"    Total latent thought time: {total_speculative_latent_time:.4f}s")
        print(f"    Total token generation time: {total_speculative_token_time:.4f}s")
        
        print(f"\n  Baseline Time Breakdown:")
        print(f"    Average latent thought generation time: {avg_base_latent_time:.4f}s per sample")
        print(f"    Average token generation time: {avg_base_token_time:.4f}s per sample")
        print(f"    Total latent thought time: {total_baseline_latent_time:.4f}s")
        print(f"    Total token generation time: {total_baseline_token_time:.4f}s")
        
        if avg_baseline_time > 0:
            actual_speedup = avg_baseline_time / avg_speculative_time
            latent_speedup = avg_base_latent_time / avg_spec_latent_time if avg_spec_latent_time > 0 else 0.0
            token_speedup = avg_base_token_time / avg_spec_token_time if avg_spec_token_time > 0 else 0.0
            print(f"\n  Speedup Analysis:")
            print(f"    Overall speedup: {actual_speedup:.2f}x")
            print(f"    Latent thought speedup: {latent_speedup:.2f}x")
            print(f"    Token generation speedup: {token_speedup:.2f}x")
            
            # Calculate min/max speedup per sample
            speedups_per_sample = [
                baseline / spec if spec > 0 else 0.0
                for baseline, spec in zip(per_sample_baseline_times, per_sample_speculative_times)
            ]
            if speedups_per_sample:
                print(f"    Min overall speedup: {min(speedups_per_sample):.2f}x")
                print(f"    Max overall speedup: {max(speedups_per_sample):.2f}x")
        else:
            actual_speedup = 0.0
            latent_speedup = 0.0
            token_speedup = 0.0
    else:
        actual_speedup = 0.0
        latent_speedup = 0.0
        token_speedup = 0.0
    
    print(f"\nGeneration:")
    print(f"  Total tokens generated: {total_generated_tokens}")
    if num_samples > 0:
        avg_tokens = total_generated_tokens / num_samples
        print(f"  Average tokens per sample: {avg_tokens:.1f}")
    
    print("=" * 60)
    
    # Save results to file
    results = {
        "num_samples": num_samples,
        "baseline_only": baseline_only_bool,
        "total_tokens": total_generated_tokens,
        "avg_tokens_per_sample": total_generated_tokens / num_samples if num_samples > 0 else 0.0,
    }
    
    # Add speculative decoding stats only if not baseline_only
    if not baseline_only_bool:
        results.update({
            "latent_accepted": total_latent_accepted,
            "latent_total": total_latent_total,
            "latent_accept_rate": latent_accept_rate,
            "tokens_accepted": total_tokens_accepted,
            "tokens_total": total_tokens_total,
            "token_accept_rate": token_accept_rate,
            "draft_calls": total_draft_calls,
            "target_calls": total_target_calls,
            "total_calls": total_draft_calls + total_target_calls,
            "baseline_calls": total_generated_tokens,
            "estimated_speedup": estimated_speedup,
        })
    
    # Add timing and comparison results
    if total_baseline_time > 0:
        results.update({
            "baseline_comparison": True,
            "total_speculative_time": total_speculative_time,
            "total_baseline_time": total_baseline_time,
            "total_speculative_latent_time": total_speculative_latent_time,
            "total_draft_latent_time": total_draft_latent_time,
            "total_verify_latent_time": total_verify_latent_time,
            "total_speculative_token_time": total_speculative_token_time,
            "total_baseline_latent_time": total_baseline_latent_time,
            "total_baseline_token_time": total_baseline_token_time,
            "avg_speculative_time": total_speculative_time / num_samples if num_samples > 0 else 0.0,
            "avg_baseline_time": total_baseline_time / num_samples if num_samples > 0 else 0.0,
            "avg_speculative_latent_time": total_speculative_latent_time / num_samples if num_samples > 0 else 0.0,
            "avg_draft_latent_time": total_draft_latent_time / num_samples if num_samples > 0 else 0.0,
            "avg_verify_latent_time": total_verify_latent_time / num_samples if num_samples > 0 else 0.0,
            "avg_speculative_token_time": total_speculative_token_time / num_samples if num_samples > 0 else 0.0,
            "avg_baseline_latent_time": total_baseline_latent_time / num_samples if num_samples > 0 else 0.0,
            "avg_baseline_token_time": total_baseline_token_time / num_samples if num_samples > 0 else 0.0,
            "overall_speedup": actual_speedup,
            "latent_speedup": latent_speedup,
            "token_speedup": token_speedup,
            "exact_matches": exact_matches,
            "total_comparisons": total_comparisons,
            "match_rate": exact_matches / total_comparisons if total_comparisons > 0 else 0.0,
            "per_sample_speculative_times": per_sample_speculative_times,
            "per_sample_baseline_times": per_sample_baseline_times,
            "per_sample_speculative_latent_times": per_sample_speculative_latent_times,
            "per_sample_speculative_token_times": per_sample_speculative_token_times,
            "per_sample_baseline_latent_times": per_sample_baseline_latent_times,
            "per_sample_baseline_token_times": per_sample_baseline_token_times,
        })
    else:
        results["baseline_comparison"] = False
    
    results_path = "/checkpoints/speculative_decoding_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Save output records if requested
    if record_output_bool and output_records:
        output_dir = "/checkpoints/output_verification"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output_comparison.json")
        with open(output_path, "w") as f:
            json.dump(output_records, f, indent=2)
        print(f"\nOutput comparison saved to: {output_path}")
        print(f"  Total samples recorded: {len(output_records)}")
        
        # Count exact matches
        exact_matches = sum(1 for record in output_records if record["exact_match"])
        print(f"  Exact token matches: {exact_matches}/{len(output_records)} ({exact_matches/len(output_records):.1%})")
    
    checkpoint_volume.commit()


@app.local_entrypoint()
def main():
    print("Speculative Decoding Evaluation")
    print("=" * 60)
    print()
    print("Usage:")
    print("  modal run modal_speculative_decode.py::evaluate_speculative_decoding \\")
    print("    --draft-checkpoint-path '/checkpoints/draft_model/draft_model_epoch_10.pt' \\")
    print("    --target-checkpoint-path '/checkpoints/gsm-coconut/checkpoint_4' \\")
    print("    --data-path 'data/gsm_test.json' \\")
    print("    --num-latent-thoughts 6 \\")
    print("    --gamma 4 \\")
    print("    --max-new-tokens 50 \\")
    print("    --similarity-threshold 0.9 \\")
    print("    --max-samples 100 \\")
    print("    --clock-run True \\")
    print("    --record-output True")
    print()
    print("Required Parameters:")
    print("  - draft-checkpoint-path: Path to draft model checkpoint in Modal volume")
    print("  - target-checkpoint-path: Path to target (Coconut) model checkpoint")
    print()
    print("Optional Parameters (with defaults):")
    print("  - data-path: Path to test data (default: 'data/gsm_test.json')")
    print("  - num-latent-thoughts: Number of latent thoughts to generate (default: 6)")
    print("  - gamma: Number of draft tokens per round (default: 4)")
    print("           ⚠️  Higher gamma = more parallel verification, potentially better speedup")
    print("  - max-new-tokens: Maximum tokens to generate per sample (default: 50)")
    print("  - similarity-threshold: Cosine similarity threshold for latent thought acceptance (default: 0.9)")
    print("                            ⚠️  Range: 0.0-1.0. Higher = stricter acceptance criteria")
    print("  - max-samples: Maximum number of samples to evaluate (default: 100)")
    print("  - clock-run: Enable wallclock time comparison with baseline autoregressive decoding (default: \"False\")")
    print("              ⚠️  Accepts: \"True\", \"true\", \"1\", \"False\", \"false\", \"0\"")
    print("              When enabled, also runs baseline decoding and reports actual speedup based on time")
    print("  - record-output: Record generated outputs for comparison (default: \"False\")")
    print("                  ⚠️  Requires clock-run=True. Accepts: \"True\", \"true\", \"1\", etc.")
    print("                  Saves outputs to /checkpoints/output_verification/output_comparison.json")
    print("  - baseline-only: Only run baseline (target model) decoding, skip speculative decoding (default: \"False\")")
    print("                  ⚠️  Accepts: \"True\", \"true\", \"1\", etc.")
    print("                  When enabled, only runs target model and reports timing breakdown (latent + tokens)")
    print("                  Note: baseline_only=True automatically enables clock_run=True")
    print("  - tokens-speculative: Use speculative decoding for tokens (default: \"False\")")
    print("                       ⚠️  Accepts: \"True\", \"true\", \"1\", etc.")
    print("                       If False, uses autoregressive generation with target model only after latent thoughts")
    print("                       If True, uses speculative decoding for tokens (draft + verify)")
    print("                       Note: Latent thoughts always use speculative decoding (draft + verify)")
    print()
    print("Examples:")
    print("  # Use higher gamma for better parallelization:")
    print("  --gamma 8 --similarity-threshold 0.9")
    print()
    print("  # Use stricter latent thought acceptance:")
    print("  --gamma 4 --similarity-threshold 0.95")
    print()
    print("  # Get actual wallclock speedup comparison:")
    print("  --gamma 4 --similarity-threshold 0.9 --clock-run True")
    print("  # Or:")
    print("  --gamma 4 --similarity-threshold 0.9 --clock-run \"True\"")
    print()
    print("The script will evaluate speculative decoding and report:")
    print("  - Latent thought acceptance rate")
    print("  - Token acceptance rate")
    print("  - Model call counts (draft vs target)")
    print("  - Estimated speedup vs baseline (based on model calls)")
    if True:  # Always show this option
        print("  - Actual wallclock speedup (if --clock-run True)")
    print("  - Generation statistics")
