"""
Download speculative decoding results from Modal volume.
Downloads:
- /checkpoints/speculative_decoding_results.json
- /checkpoints/output_verification/output_comparison.json
- /checkpoints/output_verification/analysis_summary.json (if exists)
"""

import subprocess
import os
import argparse


def run_modal_cli(command):
    """Run Modal CLI command and return (success, output)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        return result.returncode == 0, result.stdout
    except Exception as e:
        return False, str(e)


def download_result_file(volume_name, remote_path, local_path, verbose=True):
    """Download a single file from Modal volume, trying multiple path formats."""
    path_variants = [
        remote_path,  # Full path
        remote_path.lstrip('/'),  # Without leading slash
        remote_path.replace('/checkpoints/', 'checkpoints/'),  # Relative with checkpoints
        remote_path.replace('/checkpoints/', ''),  # Without checkpoints prefix
    ]
    
    # Normalize local path
    local_path = local_path.replace('\\', '/')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    for path_variant in path_variants:
        if not path_variant or path_variant == '/':
            continue
        
        path_variant = path_variant.replace('\\', '/')
        command = f'modal volume get --force {volume_name} "{path_variant}" "{local_path}"'
        
        if verbose:
            print(f"   Trying: {path_variant}...")
        
        success, output = run_modal_cli(command)
        
        if success:
            # Verify file was downloaded
            if os.path.exists(local_path):
                if verbose:
                    print(f"   ✅ Downloaded: {os.path.basename(local_path)}")
                return True
    
    if verbose:
        print(f"   ❌ Failed to download: {remote_path}")
    return False


def download_all_results(local_path="./downloaded_checkpoints", volume_name="coconut-checkpoints"):
    """
    Download all speculative decoding result files.
    
    Args:
        local_path: Local directory to download to (default: ./downloaded_checkpoints)
        volume_name: Name of Modal volume (default: coconut-checkpoints)
    """
    print("=" * 60)
    print("DOWNLOADING SPECULATIVE DECODING RESULTS")
    print("=" * 60)
    print(f"\nVolume: {volume_name}")
    print(f"Local path: {local_path}\n")
    
    os.makedirs(local_path, exist_ok=True)
    
    # Files to download
    files_to_download = [
        {
            "remote": "/checkpoints/speculative_decoding_results.json",
            "local": os.path.join(local_path, "speculative_decoding_results.json"),
            "name": "Speculative Decoding Results"
        },
        {
            "remote": "/checkpoints/output_verification/output_comparison.json",
            "local": os.path.join(local_path, "output_verification", "output_comparison.json"),
            "name": "Output Comparison"
        },
        {
            "remote": "/checkpoints/output_verification/analysis_summary.json",
            "local": os.path.join(local_path, "output_verification", "analysis_summary.json"),
            "name": "Analysis Summary"
        },
    ]
    
    downloaded = []
    failed = []
    
    for file_info in files_to_download:
        print(f"\n📥 Downloading {file_info['name']}...")
        print(f"   Remote: {file_info['remote']}")
        print(f"   Local:  {file_info['local']}")
        
        success = download_result_file(
            volume_name,
            file_info['remote'],
            file_info['local'],
            verbose=True
        )
        
        if success:
            downloaded.append(file_info['name'])
        else:
            # Check if file might not exist (analysis_summary is optional)
            if 'analysis_summary' in file_info['remote']:
                print(f"   ⚠️  File may not exist (run analysis first)")
            else:
                failed.append(file_info['name'])
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully downloaded: {len(downloaded)}")
    for name in downloaded:
        print(f"   - {name}")
    
    if failed:
        print(f"\n❌ Failed to download: {len(failed)}")
        for name in failed:
            print(f"   - {name}")
        return False
    
    print(f"\n📁 Files saved to: {local_path}")
    print("\n💡 You can now inspect the JSON files:")
    for file_info in files_to_download:
        if file_info['name'] in downloaded:
            print(f"   - {file_info['local']}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download speculative decoding results from Modal volume"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="./downloaded_checkpoints",
        help="Local directory to download to (default: ./downloaded_checkpoints)"
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default="coconut-checkpoints",
        help="Name of Modal volume (default: coconut-checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Check if Modal CLI is available
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
    
    success = download_all_results(
        local_path=args.local_path,
        volume_name=args.volume_name
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

