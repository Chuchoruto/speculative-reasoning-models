import modal
import os
import yaml

# Define the Modal app
app = modal.App("coconut-data-collection")

# Create the image with all dependencies and include project files needed remotely
image = (
    modal.Image
        .debian_slim()
        .pip_install([
            "torch==2.5.1",
            "numpy==2.1.3", 
            "transformers==4.46.2",
            "wandb==0.18.7",
            "datasets==3.1.0",
            "tqdm==4.67.0",
            "pyyaml"
        ])
        .env({
            "NCCL_DEBUG": "INFO",
            "CUDA_VISIBLE_DEVICES": "0"
        })
        .add_local_file("run.py", "/workspace/run.py")
        .add_local_file("coconut.py", "/workspace/coconut.py")
        .add_local_file("dataset.py", "/workspace/dataset.py")
        .add_local_file("utils.py", "/workspace/utils.py")
        .add_local_file("collect_draft_training_data.py", "/workspace/collect_draft_training_data.py")
        .add_local_dir("data", "/workspace/data")
        .add_local_dir("args", "/workspace/args")
)

# Use the same persistent volume (gsm/gpt2 small)
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)
# Separate volume for gpt2-medium
checkpoint_volume_medium = modal.Volume.from_name("coconut-checkpoints-gpt2-medium", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100:1",
    timeout=60 * 60 * 6,  # 6 hour timeout for data collection
    volumes={"/checkpoints": checkpoint_volume},  # Mount standard volume
    secrets=[modal.Secret.from_name("wandb")],
)
def collect_draft_training_data(
    checkpoint_path: str,
    data_path: str,
    output_filename: str = None,
    max_samples: int = None,
    max_latent_stage: int = None,
    c_thought: int = None,
    model_id: str = None,
):
    """
    Collect latent thought vectors and logits from Coconut model for draft model training.
    Uses torchrun for data collection with 1 GPU.
    
    Large vectors (latent thoughts and logits) are stored in compressed NPZ files,
    while metadata is stored in JSON with references to the NPZ files.
    
    Auto-detects model type and sets defaults based on checkpoint path:
    - If checkpoint contains "gpt2-medium" or "medium", uses gpt2-medium defaults
    - Otherwise uses standard gpt2 defaults
    - For ProntoQA: defaults to max_latent_stage=6, c_thought=1
    
    Args:
        checkpoint_path: Path to Coconut model checkpoint (required)
        data_path: Path to dataset JSON file (required)
        output_filename: Name of the output JSON file (auto-generated if None)
        max_samples: Maximum number of samples to collect (None for all)
        max_latent_stage: Maximum latent stage to use (auto-detected if None)
        c_thought: Number of latent tokens per stage (auto-detected if None)
        model_id: Base model ID (auto-detected if None)
    """
    import subprocess
    
    os.chdir("/workspace")
    
    # Auto-detect model type from checkpoint path
    is_medium = "gpt2-medium" in checkpoint_path or "medium" in checkpoint_path.lower()
    
    # Set defaults based on model type and dataset
    if model_id is None:
        model_id = "openai-community/gpt2-medium" if is_medium else "openai-community/gpt2"
    
    # ProntoQA defaults
    if "prontoqa" in data_path.lower():
        if max_latent_stage is None:
            max_latent_stage = 6
        if c_thought is None:
            c_thought = 1
    else:
        # GSM8K defaults
        if max_latent_stage is None:
            max_latent_stage = 3
        if c_thought is None:
            c_thought = 2
    
    # Auto-generate output filename if not provided
    if output_filename is None:
        # Extract split from data_path (train/valid/test)
        if "train" in data_path.lower():
            split = "train"
        elif "valid" in data_path.lower() or "val" in data_path.lower():
            split = "valid"
        elif "test" in data_path.lower():
            split = "test"
        else:
            split = "data"
        
        model_type = "medium" if is_medium else "standard"
        output_filename = f"prontoqa_{split}_draft_training_data_{model_type}.json"
    
    print(f"Collecting draft training data from checkpoint: {checkpoint_path}")
    print(f"Model: {model_id}")
    print(f"Data path: {data_path}")
    print(f"Max latent stage: {max_latent_stage}, C_thought: {c_thought}")
    print(f"Output will be saved to: /checkpoints/draft_data_pqa/{output_filename}")
    
    # Create output directory in volume
    os.makedirs("/checkpoints/draft_data_pqa", exist_ok=True)
    
    # Create config for data collection
    config = {
        "project": "Speculative-Reasoning",
        "save_path": "/checkpoints",
        "model_id": model_id,
        "load_model_path": checkpoint_path,
        "c_thought": c_thought,
        "max_latent_stage": max_latent_stage,
        "seed": 0,
        "output_filename": output_filename,
        "output_dir": "draft_data_pqa",  # Override default "draft_data" directory
        "max_samples": max_samples,
        "data_path": data_path,
    }
    
    with open("draft_collection_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("Starting data collection with 1x A100 GPU...")
    
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "collect_draft_training_data.py",
        "draft_collection_config.yaml"
    ], check=True)
    
    print("✅ Data collection completed!")
    print(f"Collected data saved to: /checkpoints/draft_data_pqa/{output_filename}")
    
    # Commit the volume to persist changes
    checkpoint_volume.commit()


@app.function(
    image=image,
    gpu="A100:1",
    timeout=60 * 60 * 6,
    volumes={"/checkpoints": checkpoint_volume_medium},  # Mount medium volume
    secrets=[modal.Secret.from_name("wandb")],
)
def collect_draft_training_data_medium(
    checkpoint_path: str,
    data_path: str,
    output_filename: str = None,
    max_samples: int = None,
    max_latent_stage: int = None,
    c_thought: int = None,
    model_id: str = "openai-community/gpt2-medium",
):
    """
    Collect draft training data for GPT2-medium models (uses medium volume).
    Same interface as collect_draft_training_data but uses the medium volume.
    
    Args:
        checkpoint_path: Path to Coconut model checkpoint (required)
        data_path: Path to dataset JSON file (required)
        output_filename: Name of the output JSON file (auto-generated if None)
        max_samples: Maximum number of samples to collect (None for all)
        max_latent_stage: Maximum latent stage to use (defaults to 6 for ProntoQA)
        c_thought: Number of latent tokens per stage (defaults to 1 for ProntoQA)
        model_id: Base model ID (defaults to gpt2-medium)
    """
    import subprocess
    
    os.chdir("/workspace")
    
    # ProntoQA defaults
    if "prontoqa" in data_path.lower():
        if max_latent_stage is None:
            max_latent_stage = 6
        if c_thought is None:
            c_thought = 1
    else:
        # GSM8K defaults
        if max_latent_stage is None:
            max_latent_stage = 3
        if c_thought is None:
            c_thought = 2
    
    # Auto-generate output filename if not provided
    if output_filename is None:
        # Extract split from data_path (train/valid/test)
        if "train" in data_path.lower():
            split = "train"
        elif "valid" in data_path.lower() or "val" in data_path.lower():
            split = "valid"
        elif "test" in data_path.lower():
            split = "test"
        else:
            split = "data"
        
        output_filename = f"prontoqa_{split}_draft_training_data.json"
    
    # Base directory for gpt2-medium data (matches training script expectation)
    base_dir = "/checkpoints/gpt2medium-prontoqa-checkpoints"
    draft_data_dir = f"{base_dir}/draft_data_pqa"
    
    print(f"Collecting draft training data from checkpoint: {checkpoint_path}")
    print(f"Model: {model_id}")
    print(f"Data path: {data_path}")
    print(f"Max latent stage: {max_latent_stage}, C_thought: {c_thought}")
    print(f"Output will be saved to: {draft_data_dir}/{output_filename}")
    
    # Create output directory in volume
    os.makedirs(draft_data_dir, exist_ok=True)
    
    # Create config for data collection
    # Note: output_dir should be relative to save_path, so we need to construct full path
    config = {
        "project": "Speculative-Reasoning",
        "save_path": base_dir,  # Save to base directory
        "model_id": model_id,
        "load_model_path": checkpoint_path,
        "c_thought": c_thought,
        "max_latent_stage": max_latent_stage,
        "seed": 0,
        "output_filename": output_filename,
        "output_dir": "draft_data_pqa",  # Relative to base_dir
        "max_samples": max_samples,
        "data_path": data_path,
    }
    
    with open("draft_collection_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("Starting data collection with 1x A100 GPU...")
    
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "collect_draft_training_data.py",
        "draft_collection_config.yaml"
    ], check=True)
    
    print("✅ Data collection completed!")
    print(f"Collected data saved to: {draft_data_dir}/{output_filename}")
    
    # Commit the volume to persist changes
    checkpoint_volume_medium.commit()


@app.local_entrypoint()
def main():
    print("🚀 Coconut Draft Training Data Collection")
    print("=" * 60)
    print()
    print("This script collects latent thought vectors and logits from a trained Coconut model")
    print("for use in training a draft model for speculative decoding.")
    print()
    print("Standard GPT2 (ProntoQA-Final):")
    print("  modal run modal_data_collection.py::collect_draft_training_data \\")
    print("    --checkpoint-path '/checkpoints/prontoqa-coconut-final/checkpoint_50' \\")
    print("    --data-path 'data/prontoqa_train.json'")
    print()
    print("  modal run modal_data_collection.py::collect_draft_training_data \\")
    print("    --checkpoint-path '/checkpoints/prontoqa-coconut-final/checkpoint_50' \\")
    print("    --data-path 'data/prontoqa_valid.json'")
    print()
    print("GPT2-Medium (ProntoQA):")
    print("  modal run modal_data_collection.py::collect_draft_training_data_medium \\")
    print("    --checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_50' \\")
    print("    --data-path 'data/prontoqa_train.json'")
    print()
    print("  modal run modal_data_collection.py::collect_draft_training_data_medium \\")
    print("    --checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_50' \\")
    print("    --data-path 'data/prontoqa_valid.json'")
    print()
    print("Parameters:")
    print("  --checkpoint-path: Path to Coconut model checkpoint (required)")
    print("  --data-path: Path to dataset JSON file (required)")
    print("  --output-filename: Name of output JSON file (auto-generated if not provided)")
    print("  --max-samples: Maximum number of samples to collect (default: None, collect all)")
    print("  --max-latent-stage: Maximum latent stage (default: 6 for ProntoQA, 3 for GSM8K)")
    print("  --c-thought: Number of latent tokens per stage (default: 1 for ProntoQA, 2 for GSM8K)")
    print()
    print("Default values are auto-detected based on:")
    print("  - Model type (gpt2 vs gpt2-medium) from checkpoint path")
    print("  - Dataset type (ProntoQA vs GSM8K) from data path")
    print()
    print("The collected data will be saved to /checkpoints/draft_data_pqa/ in the Modal volume.")
    print("You can download it using modal_download.py::download_draft_training_data")

