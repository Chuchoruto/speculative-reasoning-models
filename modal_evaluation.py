import modal
import os
import yaml

# Define the Modal app
app = modal.App("coconut-gsm8k-eval")

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
    timeout=60 * 60 * 2,  # 2 hour timeout for evaluation
    volumes={"/checkpoints": checkpoint_volume},  # Mount same volume
    secrets=[modal.Secret.from_name("wandb")],
)
def evaluate_model(checkpoint_path: str):
    """Evaluate the trained Coconut model with 4x A100 GPUs"""
    import subprocess
    import os
    
    os.chdir("/workspace")
    
    print(f"Evaluating model from: {checkpoint_path}")
    
    config = {
        "project": "Speculative-Reasoning",
        "save_path": "/checkpoints",
        "name": "gsm-coconut-eval",
        "only_eval": True,
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
        "load_model_path": checkpoint_path,
        "seed": 0,
        "resume": 0,
        "bf16": False,
        "train_path": "data/gsm_train.json",
        "val_path": "data/gsm_test.json",  # Use test set for final evaluation
        "reset_optimizer": False,
        "batch_size_training": 32,
        "debug": False,
        "gradient_accumulation_steps": 1,
        "num_epochs": 1,
        "lr": 1e-4,
        "weight_decay": 0.01
    }
    
    with open("gsm_eval_modal.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("Starting evaluation on GSM8K test set...")
    
    subprocess.run([
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", "4",
        "run.py",
        "gsm_eval_modal.yaml"
    ], check=True)
    
    print("Evaluation completed!")
    print("Check the logs for final test accuracy results")

@app.local_entrypoint()
def main():
    print("Starting GSM8K Model Evaluation...")
    print("Make sure you have completed Coconut training first!")
    print("You need to specify the best Coconut checkpoint path.")
    print("Example usage:")
    print("modal run modal_evaluation.py::evaluate_model --checkpoint-path '/checkpoints/gsm-coconut/checkpoint_25'")