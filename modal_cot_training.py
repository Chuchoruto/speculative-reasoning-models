import modal
import os
import yaml

# Define the Modal app
app = modal.App("coconut-gsm8k-cot")

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

# Create a persistent volume for checkpoints
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100:4",  # Request 4x A100 GPUs
    timeout=60 * 60 * 8,  # 8 hour timeout
    volumes={"/checkpoints": checkpoint_volume},  # Mount volume
    secrets=[modal.Secret.from_name("wandb")],
)
def train_cot():
    """Train CoT model (Stage 0) with 4x A100 GPUs"""
    import subprocess
    import os
    
    os.chdir("/workspace")
    
    # Create config for CoT training - save to volume
    config = {
        "project": "Speculative-Reasoning",
        "save_path": "/checkpoints",  # Save to volume instead of workspace
        "name": "gsm-cot",
        "only_eval": False,
        "coconut": False,
        "cot": True,
        "no_thoughts": False,
        "no_cot": False,
        "c_thought": 0,
        "epochs_per_stage": 1,
        "max_latent_stage": 0,
        "pad_latent_to_max": True,
        "save_only_improve": True,
        "uniform_prob": 0.0,
        "model_id": "openai-community/gpt2",
        "load_model_path": "None",
        "seed": 0,
        "resume": 0,
        "bf16": True,
        "train_path": "data/gsm_train.json",
        "val_path": "data/gsm_valid.json",
        "reset_optimizer": False,
        "batch_size_training": 32,  # Per GPU batch size
        "debug": False,
        "gradient_accumulation_steps": 1,
        "num_epochs": 25,
        "lr": 1e-4,
        "weight_decay": 0.01
    }
    
    with open("gsm_cot_modal.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("Starting CoT training with 4x A100 GPUs...")
    print("Expected validation accuracy: ~40%")
    
    # Run with torchrun for distributed training
    subprocess.run([
        "torchrun", 
        "--nnodes", "1", 
        "--nproc_per_node", "4",  # 4 GPUs
        "run.py", 
        "gsm_cot_modal.yaml"
    ], check=True)
    
    print("CoT training completed!")
    print("Checkpoints saved to: /checkpoints/gsm-cot/")
    print("Look for checkpoint with ~40% validation accuracy")
    
    # Commit the volume to persist changes
    checkpoint_volume.commit()

@app.local_entrypoint()
def main():
    print("Starting GSM8K CoT Training (Stage 0)...")
    train_cot.remote()