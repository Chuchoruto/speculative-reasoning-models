import modal
import os
import argparse

app = modal.App("coconut-download")

image = modal.Image.debian_slim().pip_install(["modal"])

# Use the same volume as the training scripts
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)

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

@app.local_entrypoint()
def main():
    print("🚀 Coconut Model Download Utility")
    print("=" * 50)
    print()
    print("Available commands:")
    print()
    print("📋 List available checkpoints:")
    print("   modal run modal_download.py::list_available_checkpoints")
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
    print("   modal run modal_download.py::list_draft_training_data")
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
