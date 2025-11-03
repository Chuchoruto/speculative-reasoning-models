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
    val_json_filename: str = None,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    ce_weight: float = 1.0,
    kl_weight: float = 1.0,
    cosine_weight: float = 1.0,
    temperature: float = 1.0,
    wandb_project: str = "draft-model-training",
    wandb_run_name: str = None,
):
    """
    Train draft model using data from Modal volume.
    
    Args:
        data_json_filename: Name of JSON metadata file in /checkpoints/draft_data/
        val_json_filename: Optional name of validation JSON metadata file in /checkpoints/draft_data/
        batch_size: Training batch size per GPU (effective batch size = batch_size * 2)
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer (L2 regularization)
        ce_weight: Weight for causal LM (CrossEntropy) loss
        kl_weight: Weight for KL divergence loss
        cosine_weight: Weight for cosine similarity loss (latent thoughts)
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
    
    # Verify training data exists
    if not os.path.exists(data_json_path):
        available_files = os.listdir("/checkpoints/draft_data/") if os.path.exists("/checkpoints/draft_data/") else []
        raise FileNotFoundError(
            f"Training data file not found: {data_json_path}\n"
            f"Available files in /checkpoints/draft_data/: {available_files[:20]}"
        )
    
    print(f"Loading training data from: {data_json_path}")
    print(f"NPZ files directory: {data_dir}")
    
    # Check validation data if provided
    val_json_path = None
    if val_json_filename:
        val_json_path = f"/checkpoints/draft_data/{val_json_filename}"
        if not os.path.exists(val_json_path):
            print(f"Warning: Validation data file not found: {val_json_path}")
            print("Continuing without validation...")
            val_json_path = None
        else:
            print(f"Validation data will be loaded from: {val_json_path}")
    
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
        "val_json_path": val_json_path,  # None if not provided
        "val_data_dir": data_dir,  # Use same directory for validation NPZ files
        "save_path": save_path,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "num_workers": 4,
        "ce_weight": ce_weight,
        "kl_weight": kl_weight,
        "cosine_weight": cosine_weight,
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
    print("    --val-json-filename 'draft_validation_data.json' \\")
    print("    --batch-size 32 \\")
    print("    --num-epochs 10 \\")
    print("    --lr 0.001 \\")
    print("    --weight-decay 0.01 \\")
    print("    --ce-weight 1.0 \\")
    print("    --kl-weight 1.0 \\")
    print("    --cosine-weight 1.0 \\")
    print("    --temperature 3.0 \\")
    print("    --wandb-project 'draft-model-training'")
    print()
    print("Default parameters:")
    print("  - data_json_filename: 'draft_training_data.json'")
    print("  - val_json_filename: None (optional, enable validation if provided)")
    print("  - batch_size: 32 (per GPU, effective: 64 with 2 GPUs)")
    print("  - num_epochs: 10")
    print("  - lr: 1e-4 (recommended: 1e-3)")
    print("  - weight_decay: 0.01")
    print("  - ce_weight: 1.0 (causal LM loss weight)")
    print("  - kl_weight: 1.0 (KL divergence loss weight)")
    print("  - cosine_weight: 1.0 (cosine similarity loss weight for latent thoughts)")
    print("  - temperature: 1.0 (recommended: 3.0 for KL divergence)")
    print("  - wandb_project: 'draft-model-training'")
    print()
    print("Note: Validation runs after each epoch and logs metrics to WandB.")
    print()
    print("Note: Training uses 2 GPUs with torchrun for distributed training.")
    print("      Batch size is per GPU, so effective batch size = batch_size * 2")

