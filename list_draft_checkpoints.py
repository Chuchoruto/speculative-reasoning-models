"""
Script to list all draft model checkpoints in Modal volume.
Uses Modal CLI to list files, or provides a Modal function as fallback.
"""

import subprocess
import sys

def run_modal_cli(cmd: list):
    """Run Modal CLI command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**subprocess.os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError:
        return False, "", "Modal CLI not found. Install with: pip install modal"
    except Exception as e:
        return False, "", str(e)


def list_draft_model_checkpoints_cli(volume_name: str = "coconut-checkpoints"):
    """
    List draft model checkpoints using Modal CLI.
    
    Args:
        volume_name: Name of the Modal volume
    """
    print("=" * 60)
    print("DRAFT MODEL CHECKPOINTS")
    print("=" * 60)
    print(f"\nVolume: {volume_name}")
    print("Searching in: /checkpoints/draft_model/")
    print()
    
    # Try different path formats
    path_variants = [
        "draft_model",
        "/checkpoints/draft_model",
        "checkpoints/draft_model",
    ]
    
    files = []
    working_path = None
    
    for path_var in path_variants:
        cmd = ["modal", "volume", "ls", volume_name, path_var]
        success, stdout, stderr = run_modal_cli(cmd)
        
        if success and stdout.strip():
            lines = stdout.strip().split('\n')
            files = [line.strip() for line in lines if line.strip()]
            working_path = path_var
            break
    
    if not files:
        print("❌ No files found in draft_model directory.")
        print("\n💡 Tip: Use the Modal function instead:")
        print("   modal run list_draft_checkpoints.py::list_checkpoints")
        return
    
    # Filter for checkpoint files
    checkpoint_files = [
        f for f in files 
        if (f.endswith('.pt') or f.endswith('.pth')) and 
           ('checkpoint' in f.lower() or 'epoch' in f.lower() or 'draft' in f.lower())
    ]
    
    if not checkpoint_files:
        print(f"Found {len(files)} files, but none appear to be checkpoints:")
        for f in sorted(files)[:20]:
            print(f"  - {f}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return
    
    print(f"✅ Found {len(checkpoint_files)} draft model checkpoint(s):\n")
    
    # Sort by filename (should sort by epoch number if naming is consistent)
    checkpoint_files.sort()
    
    for i, checkpoint_file in enumerate(checkpoint_files, 1):
        # Extract epoch number if present
        epoch_num = None
        if 'epoch' in checkpoint_file.lower():
            try:
                # Extract epoch number from filename like "draft_model_epoch_10.pt"
                import re
                match = re.search(r'epoch[_\s]*(\d+)', checkpoint_file.lower())
                if match:
                    epoch_num = int(match.group(1))
            except:
                pass
        
        print(f"{i}. {checkpoint_file}")
        print(f"   📍 Full path: {working_path}/{checkpoint_file}")
        if epoch_num is not None:
            print(f"   📅 Epoch: {epoch_num}")
        print()
    
    # Find the most recent checkpoint (highest epoch number)
    latest_checkpoint = None
    latest_epoch = -1
    
    for checkpoint_file in checkpoint_files:
        if 'epoch' in checkpoint_file.lower():
            try:
                import re
                match = re.search(r'epoch[_\s]*(\d+)', checkpoint_file.lower())
                if match:
                    epoch_num = int(match.group(1))
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_checkpoint = checkpoint_file
            except:
                pass
    
    if latest_checkpoint:
        print("=" * 60)
        print(f"🆕 MOST RECENT CHECKPOINT: {latest_checkpoint}")
        print(f"   Epoch: {latest_epoch}")
        print(f"   Full path: /checkpoints/{working_path}/{latest_checkpoint}")
        print("=" * 60)
        print(f"\n💡 Use this in your speculative decoding command:")
        print(f"   --draft-checkpoint-path '/checkpoints/{working_path}/{latest_checkpoint}'")
        print("=" * 60)
    else:
        print("=" * 60)
        print("⚠️  Could not determine most recent checkpoint")
        print("=" * 60)


# Modal function for remote listing
import modal

app = modal.App("list-draft-checkpoints")

checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/checkpoints": checkpoint_volume},
)
def list_checkpoints():
    """List all draft model checkpoints in the volume."""
    import os
    
    draft_model_path = "/checkpoints/draft_model"
    
    if not os.path.exists(draft_model_path):
        print(f"❌ Directory {draft_model_path} does not exist.")
        return
    
    files = os.listdir(draft_model_path)
    
    # Filter for checkpoint files
    checkpoint_files = [
        f for f in files 
        if (f.endswith('.pt') or f.endswith('.pth')) and 
           ('checkpoint' in f.lower() or 'epoch' in f.lower() or 'draft' in f.lower())
    ]
    
    if not checkpoint_files:
        print(f"Found {len(files)} files in {draft_model_path}, but none are checkpoints:")
        for f in sorted(files)[:20]:
            print(f"  - {f}")
        return
    
    print(f"✅ Found {len(checkpoint_files)} draft model checkpoint(s):\n")
    
    checkpoint_files.sort()
    
    import re
    checkpoints_with_epochs = []
    
    for i, checkpoint_file in enumerate(checkpoint_files, 1):
        epoch_num = None
        if 'epoch' in checkpoint_file.lower():
            match = re.search(r'epoch[_\s]*(\d+)', checkpoint_file.lower())
            if match:
                epoch_num = int(match.group(1))
        
        print(f"{i}. {checkpoint_file}")
        print(f"   📍 Full path: /checkpoints/draft_model/{checkpoint_file}")
        if epoch_num is not None:
            print(f"   📅 Epoch: {epoch_num}")
            checkpoints_with_epochs.append((checkpoint_file, epoch_num))
        print()
    
    # Find latest
    if checkpoints_with_epochs:
        latest_checkpoint, latest_epoch = max(checkpoints_with_epochs, key=lambda x: x[1])
        print("=" * 60)
        print(f"🆕 MOST RECENT CHECKPOINT: {latest_checkpoint}")
        print(f"   Epoch: {latest_epoch}")
        print(f"   Full path: /checkpoints/draft_model/{latest_checkpoint}")
        print("=" * 60)


def main():
    """Main entry point - tries CLI first, then suggests Modal function."""
    volume_name = sys.argv[1] if len(sys.argv) > 1 else "coconut-checkpoints"
    
    # Try CLI first
    try:
        list_draft_model_checkpoints_cli(volume_name)
    except Exception as e:
        print(f"Error with CLI: {e}")
        print("\n💡 Use Modal function instead:")
        print("   modal run list_draft_checkpoints.py::list_checkpoints")


if __name__ == "__main__":
    main()

