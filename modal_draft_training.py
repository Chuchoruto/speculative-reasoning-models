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
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    })
    .add_local_file("train_draft_model.py", "/workspace/train_draft_model.py")
    .add_local_file("draft_model.py", "/workspace/draft_model.py")
    .add_local_file("draft_dataset.py", "/workspace/draft_dataset.py")
)

# Use the same persistent volume (gsm/small model)
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)
# Separate volume for gpt2-medium
checkpoint_volume_medium = modal.Volume.from_name("coconut-checkpoints-gpt2-medium", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100:4",  # 4 GPUs for parallel training
    timeout=60 * 60 * 24,  # 24 hours
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("wandb")],  # WandB secret
)
def train_draft_model(
    data_json_filename: str = "prontoqa_train_draft_training_data_standard.json",
    val_json_filename: str = "prontoqa_valid_draft_training_data_standard.json",
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    kl_weight: float = 1.0,
    cosine_weight: float = 1.0,
    temperature: float = 1.0,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 20,
    wandb_project: str = "draft-model-training",
    wandb_run_name: str = None,
    save_path: str = None,
):
    """
    Train draft model using data from Modal volume.
    
    Args:
        data_json_filename: Name of JSON metadata file in /checkpoints/draft_data_pqa/
        val_json_filename: Optional name of validation JSON metadata file in /checkpoints/draft_data_pqa/
        batch_size: Training batch size per GPU (effective batch size = batch_size * 4 with 4 GPUs)
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer (L2 regularization)
        kl_weight: Weight for KL divergence loss
        cosine_weight: Weight for cosine similarity loss (latent thoughts)
        temperature: Temperature for softmax in KL divergence
        gradient_accumulation_steps: Number of batches to accumulate gradients before updating (default: 4)
        warmup_steps: Number of warmup steps for learning rate (default: 20)
        wandb_project: WandB project name
        wandb_run_name: WandB run name (auto-generated if None)
        save_path: Path to save checkpoints (default: /checkpoints/draft_model_final)
    """
    import subprocess
    import datetime
    
    os.chdir("/workspace")
    
    # Paths in Modal volume
    data_json_path = f"/checkpoints/draft_data_pqa/{data_json_filename}"
    data_dir = "/checkpoints/draft_data_pqa"  # Directory with NPZ files
    if save_path is None:
        save_path = "/checkpoints/draft_model_final"
    
    os.makedirs(save_path, exist_ok=True)
    
    # Verify training data exists
    if not os.path.exists(data_json_path):
        available_files = os.listdir("/checkpoints/draft_data_pqa/") if os.path.exists("/checkpoints/draft_data_pqa/") else []
        raise FileNotFoundError(
            f"Training data file not found: {data_json_path}\n"
            f"Available files in /checkpoints/draft_data_pqa/: {available_files[:20]}"
        )
    
    print(f"Loading training data from: {data_json_path}")
    print(f"NPZ files directory: {data_dir}")
    
    # Check validation data if provided
    val_json_path = None
    if val_json_filename:
        val_json_path = f"/checkpoints/draft_data_pqa/{val_json_filename}"
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
        "kl_weight": kl_weight,
        "cosine_weight": cosine_weight,
        "temperature": temperature,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "peak_lr": 4e-4,  # Cap LR after warmup (consistent with medium model)
        "use_wandb": True,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }
    
    config_path = "draft_training_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print("Starting draft model training with 4x A100 GPUs...")
    print(f"Config: {config}")
    
    # Run training with torchrun for distributed training
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "train_draft_model.py",
        config_path
    ], check=True)
    
    print("Draft model training completed!")
    print(f"Checkpoints saved to: {save_path}")
    
    # Commit volume to persist checkpoints
    checkpoint_volume.commit()


@app.function(
    image=image,
    gpu="A100:4",  # 4 GPUs for parallel training
    timeout=60 * 60 * 24,  # 24 hours
    volumes={"/checkpoints": checkpoint_volume_medium},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_draft_model_medium(
    data_json_filename: str = "prontoqa_train_draft_training_data.json",
    val_json_filename: str = "prontoqa_valid_draft_training_data.json",
    batch_size: int = 32,
    num_epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    kl_weight: float = 1.0,
    cosine_weight: float = 0.5,
    temperature: float = 2.0,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 20,
    wandb_project: str = "draft-model-training-prontoqa-medium",
    wandb_run_name: str = None,
    save_path: str = None,
):
    """
    Train draft model using gpt2-medium ProntoQA Coconut data from Modal volume.
    Uses checkpoint_25 from gpt2medium-prontoqa-checkpoints.
    Saves checkpoints to /checkpoints/gpt2medium-prontoqa-checkpoints/draft_checkpoints/
    
    Args:
        data_json_filename: Name of JSON metadata file in /checkpoints/gpt2medium-prontoqa-checkpoints/draft_data/
        val_json_filename: Name of validation JSON metadata file
        batch_size: Training batch size per GPU (effective batch size = batch_size * 2)
        num_epochs: Number of training epochs
        lr: Learning rate (default: 1e-3)
        weight_decay: Weight decay for optimizer (L2 regularization)
        kl_weight: Weight for KL divergence loss
        cosine_weight: Weight for cosine similarity loss (latent thoughts) (default: 0.5, reduced from 1.0)
        temperature: Temperature for softmax in KL divergence (default: 2.0)
        gradient_accumulation_steps: Number of batches to accumulate gradients before updating (default: 4)
        warmup_steps: Number of warmup steps for learning rate (default: 20)
        wandb_project: WandB project name
        wandb_run_name: WandB run name (auto-generated if None)
        save_path: Path to save checkpoints (default: /checkpoints/gpt2medium-prontoqa-checkpoints/draft_model_final)
    """
    import subprocess
    import datetime
    
    os.chdir("/workspace")
    
    # Base directory in gpt2-medium volume
    base_dir = "/checkpoints/gpt2medium-prontoqa-checkpoints"
    draft_data_dir = f"{base_dir}/draft_data_pqa"
    
    # Paths in Modal volume
    data_json_path = f"{draft_data_dir}/{data_json_filename}"
    val_json_path = f"{draft_data_dir}/{val_json_filename}"
    if save_path is None:
        save_path = f"{base_dir}/draft_model_final"
    
    os.makedirs(save_path, exist_ok=True)
    
    # Verify training data exists
    if not os.path.exists(data_json_path):
        available_files = os.listdir(draft_data_dir) if os.path.exists(draft_data_dir) else []
        raise FileNotFoundError(
            f"Training data file not found: {data_json_path}\n"
            f"Available files in {draft_data_dir}: {available_files[:20]}"
        )
    
    print(f"Loading training data from: {data_json_path}")
    print(f"NPZ files directory: {draft_data_dir}")
    
    # Verify validation data
    if not os.path.exists(val_json_path):
        print(f"Warning: Validation data file not found: {val_json_path}")
        print("Continuing without validation...")
        val_json_path = None
    else:
        print(f"Validation data will be loaded from: {val_json_path}")
    
    # Generate run name if not provided
    if wandb_run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run_name = f"draft_model_medium_{timestamp}"
    
    # Create config
    config = {
        "model_id": "erwanf/gpt2-mini",
        "teacher_hidden_dim": 1024,  # gpt2-medium has 1024 hidden dim
        "data_json_path": data_json_path,
        "data_dir": draft_data_dir,
        "val_json_path": val_json_path,
        "val_data_dir": draft_data_dir,
        "save_path": save_path,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "peak_lr": 4e-4,
        "num_workers": 4,
        "kl_weight": kl_weight,
        "cosine_weight": cosine_weight,
        "temperature": temperature,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "use_wandb": True,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }
    
    config_path = "draft_training_config_medium.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print("Starting draft model training (gpt2-medium ProntoQA) with 4x A100 GPUs...")
    print(f"Config: {config}")
    
    # Run training with torchrun for distributed training
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "train_draft_model.py",
        config_path
    ], check=True)
    
    print("Draft model training completed!")
    print(f"Checkpoints saved to: {save_path}")
    
    # Commit volume to persist checkpoints
    checkpoint_volume_medium.commit()


@app.local_entrypoint()
def main():
    print("Draft Model Training on Modal")
    print("=" * 60)
    print()
    print("Usage:")
    print("  modal run modal_draft_training.py::train_draft_model \\")
    print("    --data-json-filename 'prontoqa_train_draft_training_data_standard.json' \\")
    print("    --val-json-filename 'prontoqa_valid_draft_training_data_standard.json' \\")
    print("    --batch-size 32 \\")
    print("    --num-epochs 40 \\")
    print("    --lr 0.001 \\")
    print("    --weight-decay 0.01 \\")
    print("    --kl-weight 1.0 \\")
    print("    --cosine-weight 0.5 \\")
    print("    --temperature 2.0 \\")
    print("    --gradient-accumulation-steps 4 \\")
    print("    --warmup-steps 20 \\")
    print("    --wandb-project 'draft-model-training-prontoqa-standard'")
    print()
    print("Default parameters:")
    print("  - data_json_filename: 'prontoqa_train_draft_training_data_standard.json'")
    print("  - val_json_filename: 'prontoqa_valid_draft_training_data_standard.json'")
    print("  - batch_size: 32 (per GPU, effective: 128 with 4 GPUs)")
    print("  - num_epochs: 10")
    print("  - lr: 1e-4")
    print("  - weight_decay: 0.01")
    print("  - kl_weight: 1.0 (KL divergence loss weight)")
    print("  - cosine_weight: 1.0 (cosine similarity loss weight for latent thoughts)")
    print("  - temperature: 1.0")
    print("  - gradient_accumulation_steps: 4 (accumulate gradients over 4 batches before updating)")
    print("  - warmup_steps: 20 (linear warmup for learning rate, then exponential decay per step)")
    print("  - wandb_project: 'draft-model-training'")
    print("  - save_path: '/checkpoints/draft_model_final' (path to save checkpoints)")
    print()
    print("Note: Validation runs after each epoch and logs metrics to WandB.")
    print()
    print("Note: Training uses 4 GPUs with torchrun for distributed training.")
    print("      Batch size is per GPU, so effective batch size = batch_size * 4")
    print()
    print("Note: Use --save-path to specify a different checkpoint directory (e.g., for reverse KL model):")
    print("      --save-path '/checkpoints/draft_model_reverse_kl'")
    print()
    print("=" * 60)
    print("GPT2-MEDIUM PRONTOQA TRAINING:")
    print("=" * 60)
    print("Usage:")
    print("  modal run modal_draft_training.py::train_draft_model_medium \\")
    print("    --data-json-filename 'prontoqa_train_draft_training_data.json' \\")
    print("    --val-json-filename 'prontoqa_valid_draft_training_data.json' \\")
    print("    --batch-size 32 \\")
    print("    --num-epochs 20 \\")
    print("    --lr 0.001 \\")
    print("    --weight-decay 0.01 \\")
    print("    --kl-weight 1.0 \\")
    print("    --cosine-weight 1.0 \\")
    print("    --temperature 3.0 \\")
    print("    --wandb-project 'draft-model-training-prontoqa-medium'")
    print()
    print("Default parameters:")
    print("  - data_json_filename: 'prontoqa_train_draft_training_data.json'")
    print("  - val_json_filename: 'prontoqa_valid_draft_training_data.json'")
    print("  - batch_size: 32 (per GPU, effective: 64 with 2 GPUs)")
    print("  - num_epochs: 10")
    print("  - lr: 1e-3 (recommended)")
    print("  - weight_decay: 0.01")
    print("")
    print("  - kl_weight: 1.0 (KL divergence loss weight)")
    print("  - cosine_weight: 0.5 (cosine similarity loss weight for latent thoughts, reduced)")
    print("  - temperature: 2.0 (for KL divergence)")
    print("  - gradient_accumulation_steps: 4 (accumulate gradients over 4 batches before updating)")
    print("  - warmup_steps: 20 (linear warmup for learning rate, then exponential decay per step)")
    print("  - wandb_project: 'draft-model-training-prontoqa-medium'")
    print("  - save_path: '/checkpoints/gpt2medium-prontoqa-checkpoints/draft_model_final' (path to save checkpoints)")
    print()
    print("Note: Checkpoints saved to /checkpoints/draft_model_final/ (standard) or")
    print("      /checkpoints/gpt2medium-prontoqa-checkpoints/draft_model_final/ (medium)")
    print("      Use --save-path to specify a different checkpoint directory (e.g., for reverse KL model)")
    print("Note: Validation runs after each epoch and logs metrics to WandB.")

