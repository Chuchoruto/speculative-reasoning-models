import modal
import os
import yaml

# Define the Modal app
app = modal.App("coconut-gsm8k-coconut")

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
        .add_local_dir("data", "/workspace/data")
        .add_local_dir("args", "/workspace/args")
)

# Use the same persistent volume
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100:4",
    timeout=60 * 60 * 24,
    volumes={"/checkpoints": checkpoint_volume},  # Mount same volume
    secrets=[modal.Secret.from_name("wandb")],
)
def train_coconut(cot_checkpoint_path: str):
    """Train Coconut model using CoT checkpoint with 4x A100 GPUs"""
    import subprocess
    import os
    
    os.chdir("/workspace")
    
    print(f"Loading CoT checkpoint from: {cot_checkpoint_path}")
    
    # Create config for Coconut training - save to volume
    config = {
        "project": "Speculative-Reasoning",
        "save_path": "/checkpoints",  # Save to volume
        "name": "gsm-coconut",
        "only_eval": False,
        "coconut": True,
        "cot": False,
        "no_thoughts": False,
        "no_cot": False,
        "c_thought": 2,
        "epochs_per_stage": 3,
        "max_latent_stage": 3,
        "pad_latent_to_max": True,
        "save_only_improve": False,
        "uniform_prob": 0.0,
        "model_id": "openai-community/gpt2",
        "load_model_path": cot_checkpoint_path,
        "seed": 0,
        "resume": 3,
        "bf16": True,
        "train_path": "data/gsm_train.json",
        "val_path": "data/gsm_valid.json",
        "reset_optimizer": True,
        "batch_size_training": 64,  # Per GPU batch size
        "debug": False,
        "gradient_accumulation_steps": 1,
        "num_epochs": 15,
        "lr": 1e-4,
        "weight_decay": 0.01
    }
    
    with open("gsm_coconut_modal.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("Starting Coconut training with 4x A100 GPUs...")
    print("Training stages: 0, 1, 2, 3 (with continuous latent reasoning)")
    
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "run.py",
        "gsm_coconut_modal.yaml"
    ], check=True)
    
    print("Coconut training completed!")
    print("Checkpoints saved to: /checkpoints/gsm-coconut/")
    print("Look for checkpoint with best validation accuracy")
    
    # Commit the volume to persist changes
    checkpoint_volume.commit()

@app.function(
    image=image,
    gpu="A100:4",
    timeout=60 * 60 * 24,
    volumes={"/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_prontoqa_coconut(cot_checkpoint_path: str = None):
    """
    Train ProntoQA Coconut model using existing args/prontoqa_coconut.yaml config.
    
    Args:
        cot_checkpoint_path: Optional CoT checkpoint path. If None, uses value from YAML (None means train from scratch).
    """
    import subprocess
    import os
    
    os.chdir("/workspace")
    
    # Load existing prontoqa config
    config_path = "args/prontoqa_coconut.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Update paths for Modal
    config["save_path"] = "/checkpoints"  # Override save_path to use volume
    if cot_checkpoint_path:
        config["load_model_path"] = cot_checkpoint_path
        print(f"Loading CoT checkpoint from: {cot_checkpoint_path}")
    else:
        print(f"Using load_model_path from config: {config.get('load_model_path', 'None')}")
    
    # Save updated config to workspace (temporary file)
    modal_config_path = "prontoqa_coconut_modal.yaml"
    with open(modal_config_path, "w") as f:
        yaml.dump(config, f)
    
    print("Starting ProntoQA Coconut training with 4x A100 GPUs...")
    print(f"Training config: {config_path}")
    print(f"Max latent stages: {config.get('max_latent_stage', 'N/A')}")
    print(f"C_thought: {config.get('c_thought', 'N/A')}")
    print(f"Training stages: 0 through {config.get('max_latent_stage', 'N/A')}")
    
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "run.py",
        modal_config_path
    ], check=True)
    
    print("ProntoQA Coconut training completed!")
    print("Checkpoints saved to: /checkpoints/prontoqa-coconut/")
    print("Look for checkpoint with best validation accuracy")
    
    # Commit the volume to persist changes
    checkpoint_volume.commit()


@app.local_entrypoint()
def main():
    print("Coconut Training on Modal")
    print("=" * 60)
    print()
    print("GSM8K Training:")
    print("  modal run modal_coconut_training.py::train_coconut \\")
    print("    --cot-checkpoint-path '/checkpoints/gsm-cot/checkpoint_25'")
    print()
    print("ProntoQA Training:")
    print("  modal run modal_coconut_training.py::train_prontoqa_coconut")
    print("    (or with CoT checkpoint:)")
    print("  modal run modal_coconut_training.py::train_prontoqa_coconut \\")
    print("    --cot-checkpoint-path '/checkpoints/prontoqa-cot/checkpoint_X'")
    print()
    print("Note: ProntoQA training uses args/prontoqa_coconut.yaml config")