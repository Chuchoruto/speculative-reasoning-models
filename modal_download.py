import modal
import os
import argparse

app = modal.App("coconut-download")

image = modal.Image.debian_slim().pip_install(["modal", "torch"])

# Use the same volumes as the training scripts
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)
checkpoint_volume_medium = modal.Volume.from_name("coconut-checkpoints-gpt2-medium", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 2
)
def download_cot_checkpoint(checkpoint_name: str, local_path: str = "./downloaded_checkpoints"):
    """Download a specific CoT checkpoint"""
    import os
    
    # Create local directory and local file path
    local_dir = f"{local_path}/gsm-cot"
    os.makedirs(local_dir, exist_ok=True)
    local_checkpoint_path = f"{local_dir}/{checkpoint_name}"
    
    # Download the checkpoint
    try:
        checkpoint_volume.download(f"/checkpoints/gsm-cot/{checkpoint_name}", local_checkpoint_path)
        print(f"✅ Successfully downloaded CoT checkpoint '{checkpoint_name}' to {local_checkpoint_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading CoT checkpoint '{checkpoint_name}': {e}")
        return False

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 2
)
def download_coconut_checkpoint(checkpoint_name: str, local_path: str = "./downloaded_checkpoints"):
    """Download a specific Coconut checkpoint"""
    import os
    
    # Create local directory and local file path
    local_dir = f"{local_path}/gsm-coconut"
    os.makedirs(local_dir, exist_ok=True)
    local_checkpoint_path = f"{local_dir}/{checkpoint_name}"
    
    # Download the checkpoint
    try:
        checkpoint_volume.download(f"/checkpoints/gsm-coconut/{checkpoint_name}", local_checkpoint_path)
        print(f"✅ Successfully downloaded Coconut checkpoint '{checkpoint_name}' to {local_checkpoint_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading Coconut checkpoint '{checkpoint_name}': {e}")
        return False

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 2
)
def download_all_cot_checkpoints(local_path: str = "./downloaded_checkpoints"):
    """Download all CoT checkpoints"""
    import os
    
    local_cot_path = f"{local_path}/gsm-cot"
    os.makedirs(local_cot_path, exist_ok=True)
    
    try:
        cot_path = "/checkpoints/gsm-cot"
        if not os.path.exists(cot_path):
            print("❌ CoT checkpoint directory not found")
            return False
        files = [f for f in os.listdir(cot_path) if f.startswith("checkpoint_")]
        if not files:
            print("❌ No CoT checkpoints found")
            return False
        for f in sorted(files, key=lambda x: int(x.split("_")[-1])):
            checkpoint_volume.download(f"{cot_path}/{f}", f"{local_cot_path}/{f}")
        print(f"✅ Successfully downloaded all CoT checkpoints to {local_cot_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading all CoT checkpoints: {e}")
        return False

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 2
)
def download_all_coconut_checkpoints(local_path: str = "./downloaded_checkpoints"):
    """Download all Coconut checkpoints"""
    import os
    
    local_coconut_path = f"{local_path}/gsm-coconut"
    os.makedirs(local_coconut_path, exist_ok=True)
    
    try:
        coco_path = "/checkpoints/gsm-coconut"
        if not os.path.exists(coco_path):
            print("❌ Coconut checkpoint directory not found")
            return False
        files = [f for f in os.listdir(coco_path) if f.startswith("checkpoint_")]
        if not files:
            print("❌ No Coconut checkpoints found")
            return False
        for f in sorted(files, key=lambda x: int(x.split("_")[-1])):
            checkpoint_volume.download(f"{coco_path}/{f}", f"{local_coconut_path}/{f}")
        print(f"✅ Successfully downloaded all Coconut checkpoints to {local_coconut_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading all Coconut checkpoints: {e}")
        return False

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 3
)
def download_all_checkpoints(local_path: str = "./downloaded_checkpoints"):
    """Download all checkpoints (both CoT and Coconut)"""
    import os
    
    os.makedirs(local_path, exist_ok=True)
    
    success_count = 0
    
    # Download CoT checkpoints
    try:
        cot_path = "/checkpoints/gsm-cot"
        os.makedirs(f"{local_path}/gsm-cot", exist_ok=True)
        if os.path.exists(cot_path):
            files = [f for f in os.listdir(cot_path) if f.startswith("checkpoint_")]
            for f in sorted(files, key=lambda x: int(x.split("_")[-1])):
                checkpoint_volume.download(f"{cot_path}/{f}", f"{local_path}/gsm-cot/{f}")
            if files:
                print(f"✅ Downloaded CoT checkpoints to {local_path}/gsm-cot")
                success_count += 1
            else:
                print("❌ No CoT checkpoints found")
        else:
            print("❌ CoT checkpoint directory not found")
    except Exception as e:
        print(f"❌ Error downloading CoT checkpoints: {e}")
    
    # Download Coconut checkpoints
    try:
        coco_path = "/checkpoints/gsm-coconut"
        os.makedirs(f"{local_path}/gsm-coconut", exist_ok=True)
        if os.path.exists(coco_path):
            files = [f for f in os.listdir(coco_path) if f.startswith("checkpoint_")]
            for f in sorted(files, key=lambda x: int(x.split("_")[-1])):
                checkpoint_volume.download(f"{coco_path}/{f}", f"{local_path}/gsm-coconut/{f}")
            if files:
                print(f"✅ Downloaded Coconut checkpoints to {local_path}/gsm-coconut")
                success_count += 1
            else:
                print("❌ No Coconut checkpoints found")
        else:
            print("❌ Coconut checkpoint directory not found")
    except Exception as e:
        print(f"❌ Error downloading Coconut checkpoints: {e}")
    
    print(f"\n📊 Download Summary: {success_count}/2 checkpoint types downloaded")
    return success_count > 0

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 1
)
def list_available_checkpoints():
    """List all available checkpoints in the volume"""
    import os
    
    print("🔍 Available checkpoints in the volume:")
    print("=" * 50)
    
    # List CoT checkpoints (files)
    try:
        cot_path = "/checkpoints/gsm-cot"
        if os.path.exists(cot_path):
            files = [f for f in os.listdir(cot_path) if f.startswith("checkpoint_")]
            if files:
                print("📁 CoT Checkpoints (files):")
                for f in sorted(files, key=lambda x: int(x.split("_")[-1])):
                    print(f"   - {f}")
            else:
                print("📁 CoT Checkpoints: None found")
        else:
            print("📁 CoT Checkpoints: Directory not found")
    except Exception as e:
        print(f"❌ Error listing CoT checkpoints: {e}")
    
    print()
    
    # List Coconut checkpoints (files)
    try:
        coconut_path = "/checkpoints/gsm-coconut"
        if os.path.exists(coconut_path):
            files = [f for f in os.listdir(coconut_path) if f.startswith("checkpoint_")]
            if files:
                print("📁 Coconut Checkpoints (files):")
                for f in sorted(files, key=lambda x: int(x.split("_")[-1])):
                    print(f"   - {f}")
            else:
                print("📁 Coconut Checkpoints: None found")
        else:
            print("📁 Coconut Checkpoints: Directory not found")
    except Exception as e:
        print(f"❌ Error listing Coconut checkpoints: {e}")


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 1
)
def list_checkpoints_in_path(path: str = "/checkpoints/gpt2medium-prontoqa-checkpoints"):
    """List checkpoints found under an arbitrary path (default: gpt2-medium prontoqa path)."""
    import os
    print(f"🔍 Listing checkpoints in: {path}")
    print("=" * 50)
    try:
        if not os.path.exists(path):
            print("❌ Directory not found")
            return False
        entries = sorted(os.listdir(path))
        # Show checkpoint_* directories or files
        ckpts = [e for e in entries if e.startswith("checkpoint_")]
        if ckpts:
            print("📁 Checkpoints:")
            for e in ckpts:
                print(f"   - {e}")
        else:
            print("📁 Checkpoints: None found")
        # Also list non-checkpoint entries for visibility
        others = [e for e in entries if not e.startswith("checkpoint_")]
        if others:
            print()
            print("📄 Other entries:")
            for e in others[:20]:
                print(f"   - {e}")
            if len(others) > 20:
                print(f"   ... and {len(others) - 20} more")
        return True
    except Exception as e:
        print(f"❌ Error listing '{path}': {e}")
        return False


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume_medium},
    timeout=60 * 60 * 1
)
def list_checkpoints_in_path_medium(path: str = "/checkpoints/gpt2medium-prontoqa-checkpoints"):
    """List checkpoints under path in the gpt2-medium volume."""
    import os
    print(f"🔍 Listing checkpoints (gpt2-medium volume) in: {path}")
    print("=" * 50)
    try:
        if not os.path.exists(path):
            print("❌ Directory not found")
            return False
        entries = sorted(os.listdir(path))
        ckpts = [e for e in entries if e.startswith("checkpoint_")]
        if ckpts:
            print("📁 Checkpoints:")
            for e in ckpts:
                print(f"   - {e}")
        else:
            print("📁 Checkpoints: None found")
        others = [e for e in entries if not e.startswith("checkpoint_")]
        if others:
            print()
            print("📄 Other entries:")
            for e in others[:20]:
                print(f"   - {e}")
            if len(others) > 20:
                print(f"   ... and {len(others) - 20} more")
        return True
    except Exception as e:
        print(f"❌ Error listing '{path}': {e}")
        return False

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume_medium},
    timeout=60 * 60 * 1
)
def inspect_draft_model_checkpoint(checkpoint_path: str):
    """Inspect a draft model checkpoint to show base model and configuration."""
    import torch
    import os
    
    print(f"🔍 Inspecting draft model checkpoint: {checkpoint_path}")
    print("=" * 50)
    
    try:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("📋 Configuration:")
            print(f"   Model ID: {config.get('model_id', 'Not specified')}")
            print(f"   Teacher Hidden Dim: {config.get('teacher_hidden_dim', 'Not specified')}")
            print(f"   Save Path: {config.get('save_path', 'Not specified')}")
            if 'loss_type' in config:
                print(f"   Loss Type: {config.get('loss_type', 'cosine')}")
        else:
            print("⚠️  No config found in checkpoint")
        
        # Check model state dict keys to infer structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\n📦 Model Structure:")
            
            # Check for base model keys
            base_model_keys = [k for k in state_dict.keys() if 'base_model' in k]
            if base_model_keys:
                print(f"   Found {len(base_model_keys)} base_model keys")
                # Show first few keys as examples
                print(f"   Example keys: {base_model_keys[:3]}")
            
            # Check for projection layer
            projection_keys = [k for k in state_dict.keys() if 'latent_projection' in k]
            if projection_keys:
                print(f"   Found projection layer: {projection_keys}")
                # Try to infer dimensions from weight shape
                if 'latent_projection.weight' in state_dict:
                    weight_shape = state_dict['latent_projection.weight'].shape
                    print(f"   Projection layer shape: {weight_shape}")
                    if len(weight_shape) == 2:
                        print(f"   Projection: {weight_shape[1]} -> {weight_shape[0]} dims")
            
            # Check epoch
            if 'epoch' in checkpoint:
                print(f"\n📅 Training Info:")
                print(f"   Epoch: {checkpoint['epoch']}")
        
        # Check if it's a simple state dict (no wrapper)
        elif isinstance(checkpoint, dict) and any('transformer' in k or 'base_model' in k for k in checkpoint.keys()):
            print("\n📦 Model Structure:")
            print("   Checkpoint appears to be a direct state dict")
            # Try to infer from key names
            if any('latent_projection' in k for k in checkpoint.keys()):
                print("   Contains projection layer")
        
        print("\n✅ Checkpoint inspection complete")
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 1
)
def inspect_draft_model_checkpoint_standard(checkpoint_path: str):
    """Inspect a draft model checkpoint to show base model and configuration (standard volume)."""
    import torch
    import os
    
    print(f"🔍 Inspecting draft model checkpoint: {checkpoint_path}")
    print("=" * 50)
    
    try:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("📋 Configuration:")
            print(f"   Model ID: {config.get('model_id', 'Not specified')}")
            print(f"   Teacher Hidden Dim: {config.get('teacher_hidden_dim', 'Not specified')}")
            print(f"   Save Path: {config.get('save_path', 'Not specified')}")
            if 'loss_type' in config:
                print(f"   Loss Type: {config.get('loss_type', 'cosine')}")
        else:
            print("⚠️  No config found in checkpoint")
        
        # Check model state dict keys to infer structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\n📦 Model Structure:")
            
            # Check for base model keys
            base_model_keys = [k for k in state_dict.keys() if 'base_model' in k]
            if base_model_keys:
                print(f"   Found {len(base_model_keys)} base_model keys")
                # Show first few keys as examples
                print(f"   Example keys: {base_model_keys[:3]}")
            
            # Check for projection layer
            projection_keys = [k for k in state_dict.keys() if 'latent_projection' in k]
            if projection_keys:
                print(f"   Found projection layer: {projection_keys}")
                # Try to infer dimensions from weight shape
                if 'latent_projection.weight' in state_dict:
                    weight_shape = state_dict['latent_projection.weight'].shape
                    print(f"   Projection layer shape: {weight_shape}")
                    if len(weight_shape) == 2:
                        print(f"   Projection: {weight_shape[1]} -> {weight_shape[0]} dims")
            
            # Check epoch
            if 'epoch' in checkpoint:
                print(f"\n📅 Training Info:")
                print(f"   Epoch: {checkpoint['epoch']}")
        
        # Check if it's a simple state dict (no wrapper)
        elif isinstance(checkpoint, dict) and any('transformer' in k or 'base_model' in k for k in checkpoint.keys()):
            print("\n📦 Model Structure:")
            print("   Checkpoint appears to be a direct state dict")
            # Try to infer from key names
            if any('latent_projection' in k for k in checkpoint.keys()):
                print("   Contains projection layer")
        
        print("\n✅ Checkpoint inspection complete")
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 1
)
def download_latest_checkpoint(model_type: str, local_path: str = "./downloaded_checkpoints"):
    """Download the latest checkpoint for a specific model type"""
    import os
    import glob
    
    if model_type.lower() not in ["cot", "coconut"]:
        print(f"❌ Invalid model type '{model_type}'. Must be 'cot' or 'coconut'")
        return False
    
    checkpoint_dir = f"/checkpoints/gsm-{model_type.lower()}"
    
    try:
        # Find all checkpoint directories
        checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_*")
        
        if not checkpoints:
            print(f"❌ No checkpoints found for {model_type}")
            return False
        
        # Get the latest checkpoint (highest number)
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1]))
        checkpoint_name = os.path.basename(latest_checkpoint)
        
        # Download it
        local_checkpoint_path = f"{local_path}/gsm-{model_type.lower()}/{checkpoint_name}"
        os.makedirs(local_checkpoint_path, exist_ok=True)
        
        checkpoint_volume.download(latest_checkpoint, local_checkpoint_path)
        print(f"✅ Downloaded latest {model_type} checkpoint '{checkpoint_name}' to {local_checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading latest {model_type} checkpoint: {e}")
        return False

# Note: Modal volumes don't support .download() method in functions.
# Use the local script download_draft_data.py instead, which uses Modal CLI.

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 1
)
def download_output_comparison():
    """Read and return the output comparison JSON file content from speculative decoding evaluation"""
    import json
    import os
    
    source_path = "/checkpoints/output_verification/output_comparison.json"
    
    try:
        if os.path.exists(source_path):
            with open(source_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Successfully read output comparison file")
            print(f"   Total samples: {len(data)}")
            print(f"\n📋 To download, use Modal CLI:")
            print(f"   modal volume get coconut-checkpoints /checkpoints/output_verification .")
            print(f"\n   Or download the entire /checkpoints directory:")
            print(f"   modal volume get coconut-checkpoints /checkpoints ./downloaded_checkpoints/")
            return data
        else:
            print(f"❌ File not found: {source_path}")
            # List what's in the directory
            output_dir = "/checkpoints/output_verification"
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"Available files in {output_dir}: {files}")
            return None
    except Exception as e:
        print(f"❌ Error reading output comparison: {e}")
        return None

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 1
)
def list_draft_training_data():
    """List all available draft training data files in the volume"""
    import os
    import glob
    
    print("🔍 Available draft training data files:")
    print("=" * 50)
    
    try:
        draft_data_path = "/checkpoints/draft_data"
        if os.path.exists(draft_data_path):
            files = os.listdir(draft_data_path)
            if files:
                json_files = [f for f in files if f.endswith('.json')]
                npz_files = [f for f in files if f.endswith('.npz')]
                
                print(f"📁 Draft Training Data:")
                print(f"   Base volume path: {draft_data_path}")
                print(f"   JSON metadata files: {len(json_files)}")
                print(f"   NPZ vector files: {len(npz_files)}")
                print()
                
                # List JSON files with details
                if json_files:
                    print("📄 JSON Metadata Files:")
                    print("   Full volume paths:")
                    for f in sorted(json_files):
                        file_path = os.path.join(draft_data_path, f)
                        full_volume_path = f"{draft_data_path}/{f}"
                        try:
                            size = os.path.getsize(file_path)
                            size_mb = size / (1024 * 1024)
                            # Count associated NPZ files
                            base_name = f.replace('.json', '')
                            npz_count = len(glob.glob(os.path.join(draft_data_path, f"{base_name}_sample_*.npz")))
                            print(f"   - {full_volume_path}")
                            print(f"     File: {f} ({size_mb:.2f} MB) → {npz_count} NPZ files")
                        except:
                            print(f"   - {full_volume_path}")
                            print(f"     File: {f}")
                
                # Show summary of NPZ files
                if npz_files:
                    print()
                    print("📦 NPZ Vector Files:")
                    print(f"   Total: {len(npz_files)} files")
                    print("   Full volume paths:")
                    if len(npz_files) <= 20:
                        # Show all if not too many
                        for f in sorted(npz_files):
                            file_path = os.path.join(draft_data_path, f)
                            full_volume_path = f"{draft_data_path}/{f}"
                            try:
                                size = os.path.getsize(file_path)
                                size_kb = size / 1024
                                print(f"   - {full_volume_path} ({size_kb:.1f} KB)")
                            except:
                                print(f"   - {full_volume_path}")
                    else:
                        # Show first and last few
                        for f in sorted(npz_files)[:5]:
                            file_path = os.path.join(draft_data_path, f)
                            full_volume_path = f"{draft_data_path}/{f}"
                            try:
                                size = os.path.getsize(file_path)
                                size_kb = size / 1024
                                print(f"   - {full_volume_path} ({size_kb:.1f} KB)")
                            except:
                                print(f"   - {full_volume_path}")
                        print(f"   ... ({len(npz_files) - 10} more files) ...")
                        for f in sorted(npz_files)[-5:]:
                            file_path = os.path.join(draft_data_path, f)
                            full_volume_path = f"{draft_data_path}/{f}"
                            try:
                                size = os.path.getsize(file_path)
                                size_kb = size / 1024
                                print(f"   - {full_volume_path} ({size_kb:.1f} KB)")
                            except:
                                print(f"   - {full_volume_path}")
            else:
                print("📁 Draft Training Data: None found")
        else:
            print("📁 Draft Training Data: Directory not found")
    except Exception as e:
        print(f"❌ Error listing draft training data: {e}")

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume_medium},
    timeout=60 * 60 * 1
)
def list_draft_training_data_medium():
    """List all available draft training data files in the gpt2-medium volume"""
    import os
    import glob
    
    print("🔍 Available draft training data files (gpt2-medium volume):")
    print("=" * 50)
    
    try:
        draft_data_path = "/checkpoints/gpt2medium-prontoqa-checkpoints/draft_data_pqa"
        if os.path.exists(draft_data_path):
            files = os.listdir(draft_data_path)
            if files:
                json_files = [f for f in files if f.endswith('.json')]
                npz_files = [f for f in files if f.endswith('.npz')]
                
                print(f"📁 Draft Training Data:")
                print(f"   Base volume path: {draft_data_path}")
                print(f"   JSON metadata files: {len(json_files)}")
                print(f"   NPZ vector files: {len(npz_files)}")
                print()
                
                # List JSON files with details
                if json_files:
                    print("📄 JSON Metadata Files:")
                    print("   Full volume paths:")
                    for f in sorted(json_files):
                        file_path = os.path.join(draft_data_path, f)
                        full_volume_path = f"{draft_data_path}/{f}"
                        try:
                            size = os.path.getsize(file_path)
                            size_mb = size / (1024 * 1024)
                            # Count associated NPZ files
                            base_name = f.replace('.json', '')
                            npz_count = len(glob.glob(os.path.join(draft_data_path, f"{base_name}_sample_*.npz")))
                            print(f"   - {full_volume_path}")
                            print(f"     File: {f} ({size_mb:.2f} MB) → {npz_count} NPZ files")
                        except:
                            print(f"   - {full_volume_path}")
                            print(f"     File: {f}")
                
                # Show summary of NPZ files
                if npz_files:
                    print()
                    print("📦 NPZ Vector Files:")
                    print(f"   Total: {len(npz_files)} files")
                    print("   Full volume paths:")
                    if len(npz_files) <= 20:
                        # Show all if not too many
                        for f in sorted(npz_files):
                            file_path = os.path.join(draft_data_path, f)
                            full_volume_path = f"{draft_data_path}/{f}"
                            try:
                                size = os.path.getsize(file_path)
                                size_kb = size / 1024
                                print(f"   - {full_volume_path} ({size_kb:.1f} KB)")
                            except:
                                print(f"   - {full_volume_path}")
                    else:
                        # Show first and last few
                        for f in sorted(npz_files)[:5]:
                            file_path = os.path.join(draft_data_path, f)
                            full_volume_path = f"{draft_data_path}/{f}"
                            try:
                                size = os.path.getsize(file_path)
                                size_kb = size / 1024
                                print(f"   - {full_volume_path} ({size_kb:.1f} KB)")
                            except:
                                print(f"   - {full_volume_path}")
                        print(f"   ... ({len(npz_files) - 10} more files) ...")
                        for f in sorted(npz_files)[-5:]:
                            file_path = os.path.join(draft_data_path, f)
                            full_volume_path = f"{draft_data_path}/{f}"
                            try:
                                size = os.path.getsize(file_path)
                                size_kb = size / 1024
                                print(f"   - {full_volume_path} ({size_kb:.1f} KB)")
                            except:
                                print(f"   - {full_volume_path}")
            else:
                print("📁 Draft Training Data: None found")
        else:
            print("📁 Draft Training Data: Directory not found")
    except Exception as e:
        print(f"❌ Error listing draft training data: {e}")

@app.local_entrypoint()
def main():
    print("🚀 Coconut Model Download Utility")
    print("=" * 50)
    print()
    print("Available commands:")
    print()
    print("📋 List available checkpoints:")
    print("   modal run modal_download.py::list_available_checkpoints")
    print("   modal run modal_download.py::list_checkpoints_in_path --path '/checkpoints/gpt2medium-prontoqa-checkpoints'")
    print("   modal run modal_download.py::list_checkpoints_in_path_medium --path '/checkpoints/gpt2medium-prontoqa-checkpoints'")
    print()
    print("📥 Download specific checkpoints:")
    print("   modal run modal_download.py::download_cot_checkpoint --checkpoint-name 'checkpoint_25' --local-path './my_checkpoints'")
    print("   modal run modal_download.py::download_coconut_checkpoint --checkpoint-name 'checkpoint_25' --local-path './my_checkpoints'")
    print()
    print("📥 Download latest checkpoints:")
    print("   modal run modal_download.py::download_latest_checkpoint --model-type 'cot' --local-path './my_checkpoints'")
    print("   modal run modal_download.py::download_latest_checkpoint --model-type 'coconut' --local-path './my_checkpoints'")
    print()
    print("📥 Download all checkpoints:")
    print("   modal run modal_download.py::download_all_cot_checkpoints --local-path './my_checkpoints'")
    print("   modal run modal_download.py::download_all_coconut_checkpoints --local-path './my_checkpoints'")
    print("   modal run modal_download.py::download_all_checkpoints --local-path './my_checkpoints'")
    print()
    print("📊 Draft Training Data:")
    print("   modal run modal_download.py::list_draft_training_data  # Standard volume")
    print("   modal run modal_download.py::list_draft_training_data_medium  # GPT2-medium volume")
    print()
    print("📥 Download Evaluation Results:")
    print("   modal run modal_download.py::download_output_comparison  # Reads and verifies file exists")
    print("   modal volume get coconut-checkpoints /checkpoints/output_verification .  # Download directory")
    print()
    print("   To download, use the local script (requires Modal CLI):")
    print("   python download_draft_data.py --filename 'draft_training_data.json' --local-path './my_checkpoints'")
    print("   python download_draft_data.py --filename 'draft_training_data.json' --local-path './my_checkpoints' --no-npz  # JSON only")
    print("   python download_draft_data.py --local-path './my_checkpoints'  # Downloads all files")
    print()
    print("   Or use Modal CLI directly:")
    print("   modal volume get coconut-checkpoints /checkpoints/draft_data/draft_training_data.json ./downloaded_checkpoints/draft_data/")
    print("   modal volume get coconut-checkpoints /checkpoints/draft_data ./downloaded_checkpoints/draft_data/  # Entire directory")
    print()
    print("💡 Tips:")
    print("   - Use 'list_available_checkpoints' first to see what's available")
    print("   - Default local path is './downloaded_checkpoints'")
    print("   - Checkpoint names are typically 'checkpoint_1', 'checkpoint_2', etc.")
    print("   - Draft training data is saved in '/checkpoints/draft_data/' in the volume")
