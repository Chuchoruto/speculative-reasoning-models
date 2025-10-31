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
            "CUDA_VISIBLE_DEVICES": "0,1,2,3"
        })
        .add_local_file("run.py", "/workspace/run.py")
        .add_local_file("coconut.py", "/workspace/coconut.py")
        .add_local_file("dataset.py", "/workspace/dataset.py")
        .add_local_file("utils.py", "/workspace/utils.py")
        .add_local_file("collect_draft_training_data.py", "/workspace/collect_draft_training_data.py")
        .add_local_dir("data", "/workspace/data")
        .add_local_dir("args", "/workspace/args")
)

# Use the same persistent volume
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100:4",
    timeout=60 * 60 * 3,  # 3 hour timeout for data collection
    volumes={"/checkpoints": checkpoint_volume},  # Mount same volume
    secrets=[modal.Secret.from_name("wandb")],
)
def collect_draft_training_data(
    checkpoint_path: str,
    output_filename: str = "draft_training_data.json",
    max_samples: int = None,
    data_path: str = "data/gsm_valid.json",
    max_latent_stage: int = 3,
    c_thought: int = 2,
    model_id: str = "openai-community/gpt2",
):
    """
    Collect latent thought vectors and logits from Coconut model for draft model training.
    Uses torchrun for distributed parallel data collection across 4 GPUs.
    
    Args:
        checkpoint_path: Path to Coconut model checkpoint
        output_filename: Name of the output JSON file
        max_samples: Maximum number of samples to collect (None for all)
        data_path: Path to dataset JSON file
        max_latent_stage: Maximum latent stage to use
        c_thought: Number of latent tokens per stage
        model_id: Base model ID
    """
    import subprocess
    
    os.chdir("/workspace")
    
    print(f"Collecting draft training data from checkpoint: {checkpoint_path}")
    print(f"Output will be saved to: /checkpoints/draft_data/{output_filename}")
    
    # Create output directory in volume
    os.makedirs("/checkpoints/draft_data", exist_ok=True)
    
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
        "max_samples": max_samples,
        "data_path": data_path,
    }
    
    with open("draft_collection_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("Starting distributed data collection with 4x A100 GPUs...")
    
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "collect_draft_training_data.py",
        "draft_collection_config.yaml"
    ], check=True)
    
    print("✅ Data collection completed!")
    print(f"Collected data saved to: /checkpoints/draft_data/{output_filename}")
    
    # Commit the volume to persist changes
    checkpoint_volume.commit()


@app.local_entrypoint()
def main():
    print("🚀 Coconut Draft Training Data Collection")
    print("=" * 50)
    print()
    print("This script collects latent thought vectors and logits from a trained Coconut model")
    print("for use in training a draft model for speculative decoding.")
    print()
    print("Usage:")
    print("modal run modal_data_collection.py::collect_draft_training_data \\")
    print("  --checkpoint-path '/checkpoints/gsm-coconut/checkpoint_4' \\")
    print("  --output-filename 'draft_training_data.json' \\")
    print("  --max-samples 1000 \\")
    print("  --data-path 'data/gsm_valid.json' \\")
    print("  --max-latent-stage 3 \\")
    print("  --c-thought 2")
    print()
    print("Parameters:")
    print("  --checkpoint-path: Path to Coconut model checkpoint (required)")
    print("  --output-filename: Name of output JSON file (default: 'draft_training_data.json')")
    print("  --max-samples: Maximum number of samples to collect (default: None, collect all)")
    print("  --data-path: Path to dataset JSON file (default: 'data/gsm_valid.json')")
    print("  --max-latent-stage: Maximum latent stage to use (default: 3)")
    print("  --c-thought: Number of latent tokens per stage (default: 2)")
    print()
    print("The collected data will be saved to /checkpoints/draft_data/ in the Modal volume.")
    print("You can download it using modal_download.py::download_draft_training_data")

