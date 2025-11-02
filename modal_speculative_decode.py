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
    gamma: int = 4,
    max_new_tokens: int = 50,
    similarity_threshold: float = 0.9,
    max_samples: int = 100,
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
    """
    import json
    import sys
    import numpy as np
    from tqdm import tqdm
    
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
    
    # Load draft model
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
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = get_dataset(data_path, tokenizer, max_size=max_samples)
    print(f"Loaded {len(dataset)} samples")
    
    # Import speculative decoding
    from speculative_decode import speculative_decode
    
    # Evaluation metrics
    total_latent_accepted = 0
    total_latent_total = 0
    total_draft_calls = 0
    total_target_calls = 0
    total_tokens_accepted = 0
    total_tokens_total = 0
    total_generated_tokens = 0
    num_samples = 0
    
    # Evaluate each sample
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
            # Run speculative decoding
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
            )
            
            # Accumulate statistics
            total_latent_accepted += stats['latent_accepted']
            total_latent_total += stats['latent_total']
            total_draft_calls += stats['num_draft_calls']
            total_target_calls += stats['num_target_calls']
            total_tokens_accepted += stats.get('tokens_accepted', 0)
            total_tokens_total += stats.get('tokens_total', 0)
            total_generated_tokens += len(generated_tokens)
            num_samples += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Print results
    print("\n" + "=" * 60)
    print("SPECULATIVE DECODING EVALUATION RESULTS")
    print("=" * 60)
    print(f"Evaluated {num_samples} samples")
    
    # Latent thought acceptance
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
    
    # Baseline: target-only would need one call per generated token
    baseline_target_calls = total_generated_tokens
    print(f"  Baseline (target-only) calls: {baseline_target_calls}")
    
    if baseline_target_calls > 0:
        # Speedup = baseline_target_calls / total_target_calls
        # (with parallel verification, we should have far fewer target calls)
        speedup = baseline_target_calls / total_target_calls
        print(f"  Estimated speedup: {speedup:.2f}x")
    else:
        speedup = 0.0
    
    print(f"\nGeneration:")
    print(f"  Total tokens generated: {total_generated_tokens}")
    if num_samples > 0:
        avg_tokens = total_generated_tokens / num_samples
        print(f"  Average tokens per sample: {avg_tokens:.1f}")
    
    print("=" * 60)
    
    # Save results to file
    results = {
        "num_samples": num_samples,
        "latent_accepted": total_latent_accepted,
        "latent_total": total_latent_total,
        "latent_accept_rate": latent_accept_rate,
        "tokens_accepted": total_tokens_accepted,
        "tokens_total": total_tokens_total,
        "token_accept_rate": token_accept_rate,
        "draft_calls": total_draft_calls,
        "target_calls": total_target_calls,
        "total_calls": total_draft_calls + total_target_calls,
        "baseline_calls": baseline_target_calls,
        "estimated_speedup": speedup,
        "total_tokens": total_generated_tokens,
        "avg_tokens_per_sample": total_generated_tokens / num_samples if num_samples > 0 else 0.0,
    }
    
    results_path = "/checkpoints/speculative_decoding_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
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
    print("    --max-samples 100")
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
    print()
    print("Examples:")
    print("  # Use higher gamma for better parallelization:")
    print("  --gamma 8 --similarity-threshold 0.9")
    print()
    print("  # Use stricter latent thought acceptance:")
    print("  --gamma 4 --similarity-threshold 0.95")
    print()
    print("The script will evaluate speculative decoding and report:")
    print("  - Latent thought acceptance rate")
    print("  - Token acceptance rate")
    print("  - Model call counts (draft vs target)")
    print("  - Estimated speedup vs baseline")
    print("  - Generation statistics")

