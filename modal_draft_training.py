"""
Modal script for training draft model.
Loads training data from Modal volumes.
Includes WandB integration.
"""

import modal
import os
import yaml

app = modal.App("draft-model-training")

# Create image with dependencies
image = (
    modal.Image
    .debian_slim()
    .pip_install([
        "torch==2.5.1",
        "numpy==2.1.3",
        "transformers==4.46.2",
        "tqdm==4.67.0",
        "pyyaml",
        "wandb==0.18.7",
    ])
    .env({
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0,1"
    })
    .add_local_file("train_draft_model.py", "/workspace/train_draft_model.py")
    .add_local_file("draft_model.py", "/workspace/draft_model.py")
    .add_local_file("draft_dataset.py", "/workspace/draft_dataset.py")
)

# Use the same persistent volume
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100:2",  # 2 GPUs for parallel training
    timeout=60 * 60 * 24,  # 24 hours
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("wandb")],  # WandB secret
)
def train_draft_model(
    data_json_filename: str = "draft_training_data.json",
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-4,
    kl_weight: float = 1.0,
    mse_weight: float = 1.0,
    temperature: float = 1.0,
    wandb_project: str = "draft-model-training",
    wandb_run_name: str = None,
):
    """
    Train draft model using data from Modal volume.
    
    Args:
        data_json_filename: Name of JSON metadata file in /checkpoints/draft_data/
        batch_size: Training batch size per GPU (effective batch size = batch_size * 2)
        num_epochs: Number of training epochs
        lr: Learning rate
        kl_weight: Weight for KL divergence loss
        mse_weight: Weight for MSE loss
        temperature: Temperature for softmax in KL divergence
        wandb_project: WandB project name
        wandb_run_name: WandB run name (auto-generated if None)
    """
    import subprocess
    import datetime
    
    os.chdir("/workspace")
    
    # Paths in Modal volume
    data_json_path = f"/checkpoints/draft_data/{data_json_filename}"
    data_dir = "/checkpoints/draft_data"  # Directory with NPZ files
    save_path = "/checkpoints/draft_model"
    
    os.makedirs(save_path, exist_ok=True)
    
    # Verify data exists
    if not os.path.exists(data_json_path):
        available_files = os.listdir("/checkpoints/draft_data/") if os.path.exists("/checkpoints/draft_data/") else []
        raise FileNotFoundError(
            f"Data file not found: {data_json_path}\n"
            f"Available files in /checkpoints/draft_data/: {available_files[:20]}"
        )
    
    print(f"Loading training data from: {data_json_path}")
    print(f"NPZ files directory: {data_dir}")
    
    # Generate run name if not provided
    if wandb_run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run_name = f"draft_model_{timestamp}"
    
    # Create config
    config = {
        "model_id": "erwanf/gpt2-mini",
        "teacher_hidden_dim": 768,  # Coconut model hidden dim
        "data_json_path": data_json_path,
        "data_dir": data_dir,
        "save_path": save_path,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": 0.01,
        "num_workers": 4,
        "kl_weight": kl_weight,
        "mse_weight": mse_weight,
        "temperature": temperature,
        "use_wandb": True,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }
    
    config_path = "draft_training_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print("Starting draft model training with 2x A100 GPUs...")
    print(f"Config: {config}")
    
    # Run training with torchrun for distributed training
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "2",
        "train_draft_model.py",
        config_path
    ], check=True)
    
    print("Draft model training completed!")
    print(f"Checkpoints saved to: {save_path}")
    
    # Commit volume to persist checkpoints
    checkpoint_volume.commit()


@app.local_entrypoint()
def main():
    print("Draft Model Training on Modal")
    print("=" * 60)
    print()
    print("Usage:")
    print("  modal run modal_draft_training.py::train_draft_model \\")
    print("    --data-json-filename 'draft_training_data.json' \\")
    print("    --batch-size 32 \\")
    print("    --num-epochs 10 \\")
    print("    --lr 1e-4 \\")
    print("    --kl-weight 1.0 \\")
    print("    --mse-weight 1.0 \\")
    print("    --wandb-project 'draft-model-training' \\")
    print("    --wandb-run-name 'my-run-name'")
    print()
    print("Default parameters:")
    print("  - data_json_filename: 'draft_training_data.json'")
    print("  - batch_size: 32 (per GPU, effective: 64 with 2 GPUs)")
    print("  - num_epochs: 10")
    print("  - lr: 1e-4")
    print("  - kl_weight: 1.0")
    print("  - mse_weight: 1.0")
    print("  - temperature: 1.0")
    print("  - wandb_project: 'draft-model-training'")
    print()
    print("Note: Training uses 2 GPUs with torchrun for distributed training.")
    print("      Batch size is per GPU, so effective batch size = batch_size * 2")

