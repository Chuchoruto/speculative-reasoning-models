"""
Local script to download draft training data from Modal volume using CLI commands.
Run this on your local machine (not in Modal).
"""

import subprocess
import os
import json
import glob
import argparse
from pathlib import Path


def run_modal_cli(command):
    """Run a modal CLI command and return the result"""
    import os as os_module
    # Set environment to use UTF-8 encoding
    env = os_module.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of failing
            env=env,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        # Handle encoding issues in error messages too
        try:
            # Try to decode stderr/stdout, but fall back to string representation
            error_msg = ""
            if e.stderr:
                try:
                    error_msg = e.stderr.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                except:
                    error_msg = str(e.stderr)
            elif e.stdout:
                try:
                    error_msg = e.stdout.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                except:
                    error_msg = str(e.stdout)
            else:
                error_msg = str(e)
        except Exception:
            error_msg = "Command failed (encoding error in error message)"
        return False, error_msg




def list_volume_files(volume_name, remote_path):
    """List files in a Modal volume directory. Returns (files_list, working_path)"""
    # Try different path formats - Modal CLI paths may differ from mount paths
    path_variants = [
        remote_path,  # Original path
        remote_path.lstrip('/'),  # Without leading slash
        remote_path.replace('/checkpoints/', ''),  # Remove checkpoints prefix
        remote_path.replace('/checkpoints/', 'checkpoints/'),  # Relative with checkpoints
    ]
    
    for path_variant in path_variants:
        # Skip empty paths
        if not path_variant or path_variant == '/':
            continue
            
        command = f'modal volume ls {volume_name} "{path_variant}"'
        success, output = run_modal_cli(command)
        if success:
            # Parse the output - Modal CLI returns file listings
            lines = output.strip().split('\n')
            files = []
            file_paths = {}  # Map filename -> full path for downloading
            for line in lines:
                line = line.strip()
                # Skip header lines and totals
                if line and not any(keyword in line.lower() for keyword in ['total', 'size', 'files', 'directories', '---']):
                    # Extract file path - could be at different positions
                    parts = line.split()
                    if parts:
                        # Try to find the file path - usually the last non-numeric part
                        for part in reversed(parts):
                            if '.' in part or not part.isdigit():
                                # Likely a file path
                                full_path = part
                                # Extract just the filename
                                filename = os.path.basename(part)
                                files.append(filename)
                                file_paths[filename] = full_path
                                break
            if files:
                print(f"   ✅ Found files using path: {path_variant}")
                return files, path_variant, file_paths
    
    # If all variants failed, return empty list
    print(f"   ⚠️  Could not list files with any path format")
    return [], None, {}


def download_file_from_volume_with_paths(volume_name, remote_path, local_path, verbose=False):
    """Download a single file from Modal volume, trying different path formats
    
    Args:
        volume_name: Name of the Modal volume
        remote_path: Remote path to the file (mount path format)
        local_path: Local path to save the file
        verbose: If True, print detailed progress messages
    
    Returns:
        (success: bool, working_path: str or None)
    """
    # Ensure local directory exists
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    
    # Try different path formats
    path_variants = [
        remote_path,  # Original path
        remote_path.lstrip('/'),  # Without leading slash
        remote_path.replace('/checkpoints/', ''),  # Remove checkpoints prefix
        remote_path.replace('/checkpoints/', 'checkpoints/'),  # Relative with checkpoints
    ]
    
    for i, path_variant in enumerate(path_variants):
        # Skip empty paths
        if not path_variant or path_variant == '/':
            continue
        
        # Normalize paths - use forward slashes for Modal paths
        path_variant = path_variant.replace('\\', '/')
        # Use forward slashes for local path too (works on both Windows and Unix)
        local_path_normalized = local_path.replace('\\', '/')
        
        # Add --force flag to overwrite existing files
        command = f'modal volume get --force {volume_name} "{path_variant}" "{local_path_normalized}"'
        if verbose or i == 0:
            print(f"   Trying: modal volume get --force {volume_name} \"{path_variant}\" ...")
        success, output = run_modal_cli(command)
        if success:
            # Verify file exists locally
            if os.path.exists(local_path):
                if verbose or i == 0:
                    print(f"   ✅ Success with path: {path_variant}")
                return True, path_variant
        else:
            # Only print error on last attempt, and suppress encoding errors
            if path_variant == path_variants[-1]:
                # Don't print encoding-related errors since they're not real failures
                if verbose and "charmap" not in output.lower() and "codec" not in output.lower():
                    print(f"   ❌ Modal CLI error: {output}")
                elif verbose and ("charmap" in output.lower() or "codec" in output.lower()):
                    # Encoding error - usually harmless, just continue
                    pass
    
    return False, None


def download_draft_training_data(
    filename: str = None,
    local_path: str = "./downloaded_checkpoints",
    include_npz: bool = True,
    volume_name: str = "coconut-checkpoints"
):
    """
    Download draft training data from Modal volume using CLI.
    
    Args:
        filename: If provided, download only this specific JSON file (and optionally its NPZ files)
        local_path: Local directory to download to
        include_npz: If True and filename is specified, also download all associated NPZ files
        volume_name: Name of the Modal volume
    """
    volume_path = "/checkpoints/draft_data"
    local_draft_path = os.path.join(local_path, "draft_data")
    os.makedirs(local_draft_path, exist_ok=True)
    
    print(f"📥 Downloading draft training data from volume '{volume_name}'")
    print(f"   Mount path (inside containers): {volume_path}")
    print(f"   Local path: {local_draft_path}")
    print()
    
    # First, verify the file exists by listing the directory
    print("🔍 Verifying files exist in volume...")
    files, working_list_path, file_paths = list_volume_files(volume_name, volume_path)
    
    # Determine working path format for downloads
    working_download_path_base = None
    if working_list_path:
        working_download_path_base = working_list_path
        # If it's a file path, get the directory
        if '.' in os.path.basename(working_list_path):
            working_download_path_base = os.path.dirname(working_list_path)
    else:
        # Try common patterns - default to path without leading slash
        working_download_path_base = volume_path.lstrip('/')
    
    if not files:
        print(f"⚠️  Could not list files in {volume_path}")
        print("   Will try to download with common path formats...")
    else:
        print(f"   Found {len(files)} files in volume")
        if filename and filename not in files:
            print(f"❌ File '{filename}' not found in volume!")
            print(f"   Available files (first 20): {', '.join(sorted(files)[:20])}")
            if len(files) > 20:
                print(f"   ... and {len(files) - 20} more files")
            return False
        elif filename:
            print(f"✅ Found '{filename}' in volume")
    print()
    
    # Download specific file (JSON) and optionally its NPZ files
    if filename:
        # Use the full path from listing if available, otherwise construct it
        if filename in file_paths:
            remote_file_path = file_paths[filename]
        else:
            # Fallback: construct path using working path base
            if working_download_path_base:
                remote_file_path = f"{working_download_path_base}/{filename}"
            else:
                remote_file_path = f"{volume_path}/{filename}"
        
        local_file_path = os.path.join(local_draft_path, filename)
        
        print(f"Downloading JSON metadata: {filename}...")
        print(f"   Remote path: {remote_file_path}")
        print(f"   Local path: {local_file_path}")
        print()
        
        success, working_path = download_file_from_volume_with_paths(volume_name, remote_file_path, local_file_path, verbose=True)
        if success:
            # Verify file was downloaded
            if os.path.exists(local_file_path):
                file_size = os.path.getsize(local_file_path)
                print(f"✅ Downloaded JSON: {filename} ({file_size / 1024:.1f} KB)")
                # Update working path base for subsequent downloads
                if working_path:
                    # Extract base path (directory part)
                    working_download_path_base = os.path.dirname(working_path)
            else:
                print(f"❌ Download appeared successful but file not found at {local_file_path}")
                return False
            
            # If include_npz, find and download associated NPZ files
            if include_npz and filename.endswith('.json'):
                # First, read the JSON to see how many samples there are
                try:
                    with open(local_file_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if len(metadata) > 0:
                        base_name = filename.replace('.json', '')
                        print(f"\n📦 Finding NPZ files for {len(metadata)} samples...")
                        
                        # Get NPZ files from volume
                        # We'll download them one by one based on sample indices
                        npz_files_to_download = []
                        for sample in metadata:
                            npz_filename = sample.get('npz_file')
                            if npz_filename:
                                npz_files_to_download.append(npz_filename)
                        
                        if npz_files_to_download:
                            print(f"   Found {len(npz_files_to_download)} NPZ files to download")
                            print("   Downloading NPZ files (this may take a while)...")
                            
                            downloaded = 0
                            for npz_filename in npz_files_to_download:
                                local_npz = os.path.join(local_draft_path, npz_filename)
                                
                                # Use the full path from listing if available
                                if npz_filename in file_paths:
                                    remote_npz = file_paths[npz_filename]
                                elif working_download_path_base:
                                    # Construct path using working base
                                    if working_download_path_base.endswith('/') or working_download_path_base.endswith('\\'):
                                        remote_npz = f"{working_download_path_base}{npz_filename}"
                                    else:
                                        remote_npz = f"{working_download_path_base}/{npz_filename}"
                                else:
                                    remote_npz = f"{volume_path}/{npz_filename}"
                                
                                # Download using the determined path (function will try variants if needed)
                                success, _ = download_file_from_volume_with_paths(volume_name, remote_npz, local_npz)
                                
                                if success:
                                    downloaded += 1
                                    if downloaded % 100 == 0:
                                        print(f"   Progress: {downloaded}/{len(npz_files_to_download)} NPZ files...")
                                else:
                                    print(f"   ⚠️  Failed to download: {npz_filename}")
                            
                            print(f"✅ Downloaded {downloaded}/{len(npz_files_to_download)} NPZ files")
                        else:
                            print("⚠️  No NPZ files found in metadata")
                except Exception as e:
                    print(f"⚠️  Error reading JSON or downloading NPZ files: {e}")
                    print("   JSON file downloaded, but NPZ files may need to be downloaded separately")
            
            print(f"\n✅ Successfully downloaded to {local_draft_path}")
            return True
        else:
            print(f"❌ Failed to download {filename}")
            return False
    else:
        # Download everything - use a different approach
        print("📥 Downloading all files from draft_data directory...")
        print("   (This will download both JSON and NPZ files)")
        print()
        
        # Download entire directory using recursive approach
        # Modal CLI doesn't support recursive downloads, so we need to list and download individually
        print("   Listing files in volume...")
        files, working_list_path_all, file_paths_all = list_volume_files(volume_name, volume_path)
        
        # Update working path base if we got it from listing
        if working_list_path_all:
            working_download_path_base = working_list_path_all
            if '.' in os.path.basename(working_list_path_all):
                working_download_path_base = os.path.dirname(working_list_path_all)
        
        if not files:
            # Fallback: try downloading the directory itself with different path formats
            print("   Attempting to download directory...")
            path_variants = [
                volume_path,
                volume_path.lstrip('/'),
                volume_path.replace('/checkpoints/', ''),
                volume_path.replace('/checkpoints/', 'checkpoints/'),
            ]
            
            for path_var in path_variants:
                if not path_var or path_var == '/':
                    continue
                # Normalize paths - use forward slashes
                path_var = path_var.replace('\\', '/')
                local_draft_path_normalized = local_draft_path.replace('\\', '/')
                command = f'modal volume get --force {volume_name} "{path_var}" "{local_draft_path_normalized}"'
                success, output = run_modal_cli(command)
                
                if success:
                    print(f"✅ Downloaded directory contents to {local_draft_path}")
                    return True
            
            print(f"❌ Failed to download directory with all path formats")
            print("\n   Try downloading specific files instead:")
            print(f"   python download_draft_data.py --filename 'draft_training_data.json'")
            return False
        else:
            print(f"   Found {len(files)} files")
            print("   Downloading files...")
            
            downloaded = 0
            for f in files:
                local_file = os.path.join(local_draft_path, f)
                
                # Use the full path from listing if available
                if f in file_paths_all:
                    remote_file = file_paths_all[f]
                elif working_download_path_base:
                    working_file_path = f"{working_download_path_base}/{f}" if not working_download_path_base.endswith('/') else f"{working_download_path_base}{f}"
                    remote_file = working_file_path
                else:
                    remote_file = f"{volume_path}/{f}"
                
                # Download using the determined path (function will try variants if needed)
                success, _ = download_file_from_volume_with_paths(volume_name, remote_file, local_file)
                
                if success:
                    downloaded += 1
                    if downloaded % 50 == 0:
                        print(f"   Progress: {downloaded}/{len(files)} files...")
            
            print(f"\n✅ Downloaded {downloaded}/{len(files)} files to {local_draft_path}")
            return True


def main():
    parser = argparse.ArgumentParser(
        description="Download draft training data from Modal volume using CLI"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Specific JSON file to download (default: download all)"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="./downloaded_checkpoints",
        help="Local directory to download to (default: ./downloaded_checkpoints)"
    )
    parser.add_argument(
        "--include-npz",
        action="store_true",
        default=True,
        help="When downloading a JSON file, also download associated NPZ files (default: True)"
    )
    parser.add_argument(
        "--no-npz",
        action="store_false",
        dest="include_npz",
        help="Don't download NPZ files, only JSON"
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default="coconut-checkpoints",
        help="Name of the Modal volume (default: coconut-checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Check if modal CLI is available
    result = subprocess.run(
        "modal --version",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Modal CLI not found. Please install it:")
        print("   pip install modal")
        print("   Then authenticate: modal token new")
        return 1
    
    success = download_draft_training_data(
        filename=args.filename,
        local_path=args.local_path,
        include_npz=args.include_npz,
        volume_name=args.volume_name
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

