import modal
import os
import yaml

# Define the Modal app
app = modal.App("coconut-gsm8k-coconut")

# Create the image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "torch==2.5.1",
    "numpy==2.1.3", 
    "transformers==4.46.2",
    "wandb==0.18.7",
    "datasets==3.1.0",
    "tqdm==4.67.0",
    "pyyaml"
])

# Mount the code directory
code_mount = modal.Mount.from_local_dir(".", remote_path="/workspace")

# Use the same persistent volume
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100:4",
    timeout=60 * 60 * 24,
    mounts=[code_mount],
    volumes={"/checkpoints": checkpoint_volume},  # Mount same volume
    environment_variables={
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    }
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
        "bf16": False,
        "train_path": "data/gsm_train.json",
        "val_path": "data/gsm_valid.json",
        "reset_optimizer": True,
        "batch_size_training": 32,  # Per GPU batch size
        "debug": False,
        "gradient_accumulation_steps": 1,
        "num_epochs": 25,
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

@app.local_entrypoint()
def main():
    print("Starting GSM8K Coconut Training (Stage 1)...")
    print("Make sure you have completed CoT training first!")
    print("You need to specify the CoT checkpoint path.")
    print("Example usage:")
    print("modal run modal_coconut_training.py::train_coconut --cot-checkpoint-path '/checkpoints/gsm-cot/checkpoint_25'")