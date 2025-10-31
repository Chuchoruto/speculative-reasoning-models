"""
Simple script to validate downloaded draft training data.
Checks that JSON metadata and NPZ files are consistent and readable.
"""

import json
import numpy as np
import os
import argparse
from pathlib import Path


def validate_draft_data(json_path: str, verbose: bool = False):
    """
    Validate draft training data by checking:
    1. JSON file is valid and readable
    2. NPZ files referenced in JSON exist
    3. NPZ files contain expected arrays with correct shapes
    4. Metadata matches vector data
    
    Args:
        json_path: Path to the JSON metadata file
        verbose: If True, print detailed information about each sample
    """
    print(f"Validating draft training data from: {json_path}")
    print("=" * 60)
    
    # Check JSON file exists
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False
    
    # Load JSON metadata
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"✅ JSON file loaded successfully: {len(data)} samples")
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return False
    
    if len(data) == 0:
        print("❌ JSON file is empty")
        return False
    
    # Get directory containing the JSON file
    json_dir = os.path.dirname(json_path)
    if json_dir == "":
        json_dir = "."
    
    # Validate each sample
    missing_npz = []
    invalid_npz = []
    shape_mismatches = []
    valid_samples = 0
    
    for i, sample in enumerate(data):
        if verbose and i < 5:
            print(f"\nSample {i}:")
            print(f"  sample_idx: {sample.get('sample_idx', 'missing')}")
            print(f"  npz_file: {sample.get('npz_file', 'missing')}")
            print(f"  num_latent_tokens: {sample.get('num_latent_tokens', 'missing')}")
            print(f"  num_target_tokens: {sample.get('num_target_tokens', 'missing')}")
        
        # Check required fields
        required_fields = ['npz_file', 'num_latent_tokens', 'num_target_tokens', 'latent_positions', 'target_tokens']
        missing_fields = [f for f in required_fields if f not in sample]
        if missing_fields:
            print(f"❌ Sample {i}: Missing required fields: {missing_fields}")
            continue
        
        # Check NPZ file exists
        npz_path = os.path.join(json_dir, sample['npz_file'])
        if not os.path.exists(npz_path):
            missing_npz.append((i, sample['npz_file']))
            continue
        
        # Load and validate NPZ file
        try:
            npz_data = np.load(npz_path)
            
            # Check required arrays exist
            if 'latent_thoughts' not in npz_data:
                invalid_npz.append((i, sample['npz_file'], "Missing 'latent_thoughts' array"))
                continue
            
            if 'token_logits' not in npz_data:
                invalid_npz.append((i, sample['npz_file'], "Missing 'token_logits' array"))
                continue
            
            # Validate shapes
            latent_thoughts = npz_data['latent_thoughts']
            token_logits = npz_data['token_logits']
            
            expected_latent_shape = (sample['num_latent_tokens'],)
            expected_token_shape = (sample['num_target_tokens'],)
            
            if len(latent_thoughts.shape) != 2:
                shape_mismatches.append((i, sample['npz_file'], 
                    f"latent_thoughts: expected 2D, got {len(latent_thoughts.shape)}D"))
                continue
            
            if len(token_logits.shape) != 2:
                shape_mismatches.append((i, sample['npz_file'],
                    f"token_logits: expected 2D, got {len(token_logits.shape)}D"))
                continue
            
            if latent_thoughts.shape[0] != sample['num_latent_tokens']:
                shape_mismatches.append((i, sample['npz_file'],
                    f"latent_thoughts: expected {sample['num_latent_tokens']} tokens, got {latent_thoughts.shape[0]}"))
                continue
            
            if token_logits.shape[0] != sample['num_target_tokens']:
                shape_mismatches.append((i, sample['npz_file'],
                    f"token_logits: expected {sample['num_target_tokens']} tokens, got {token_logits.shape[0]}"))
                continue
            
            valid_samples += 1
            
            if verbose and i < 5:
                print(f"  ✅ NPZ file valid")
                print(f"     latent_thoughts shape: {latent_thoughts.shape}")
                print(f"     token_logits shape: {token_logits.shape}")
                
        except Exception as e:
            invalid_npz.append((i, sample['npz_file'], f"Error loading: {e}"))
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary:")
    print(f"  Total samples: {len(data)}")
    print(f"  Valid samples: {valid_samples}")
    print(f"  Missing NPZ files: {len(missing_npz)}")
    print(f"  Invalid NPZ files: {len(invalid_npz)}")
    print(f"  Shape mismatches: {len(shape_mismatches)}")
    
    if missing_npz:
        print(f"\n⚠️  Missing NPZ files (first 10):")
        for i, npz_file in missing_npz[:10]:
            print(f"   Sample {i}: {npz_file}")
    
    if invalid_npz:
        print(f"\n⚠️  Invalid NPZ files (first 10):")
        for i, npz_file, reason in invalid_npz[:10]:
            print(f"   Sample {i} ({npz_file}): {reason}")
    
    if shape_mismatches:
        print(f"\n⚠️  Shape mismatches (first 10):")
        for i, npz_file, reason in shape_mismatches[:10]:
            print(f"   Sample {i} ({npz_file}): {reason}")
    
    if valid_samples == len(data):
        print("\n✅ All samples are valid!")
        return True
    else:
        print(f"\n⚠️  {len(data) - valid_samples} samples have issues")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate downloaded draft training data")
    parser.add_argument("json_path", help="Path to the JSON metadata file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Print detailed information about samples")
    
    args = parser.parse_args()
    
    validate_draft_data(args.json_path, verbose=args.verbose)


if __name__ == "__main__":
    main()

