"""
Modal script to analyze output verification data.
Compares speculative decoding outputs with baseline autoregressive outputs.
"""

import modal
import json
import os

app = modal.App("analyze-output-verification")

# Create image with dependencies
image = (
    modal.Image
    .debian_slim()
    .pip_install([
        "tqdm==4.67.0",
    ])
)

# Use the same persistent volume
checkpoint_volume = modal.Volume.from_name("coconut-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=60 * 60 * 1,  # 1 hour
)
def analyze_output_verification(
    output_file: str = "/checkpoints/output_verification/output_comparison.json",
    show_examples: str = "True",
    max_examples: int = 10,
):
    """
    Analyze output verification data to compare speculative vs baseline decoding.
    
    Args:
        output_file: Path to output comparison JSON file
        show_examples: Whether to show example matches/mismatches. Accepts "True", "true", "1", etc. (default "True")
        max_examples: Maximum number of examples to show (default 10)
    """
    # Parse show_examples string to boolean
    show_examples_bool = show_examples.lower() in ("true", "1", "yes", "on")
    
    print("=" * 60)
    print("OUTPUT VERIFICATION ANALYSIS")
    print("=" * 60)
    print(f"\nLoading output comparison from: {output_file}")
    
    if not os.path.exists(output_file):
        print(f"❌ Error: File not found: {output_file}")
        print("\n💡 Make sure you've run the evaluation with --record-output True")
        return
    
    with open(output_file, "r") as f:
        output_records = json.load(f)
    
    num_samples = len(output_records)
    print(f"✅ Loaded {num_samples} output records\n")
    
    # Analyze exact matches
    exact_matches = [r for r in output_records if r.get("exact_match", False)]
    mismatches = [r for r in output_records if not r.get("exact_match", False)]
    
    num_exact_matches = len(exact_matches)
    num_mismatches = len(mismatches)
    match_rate = num_exact_matches / num_samples if num_samples > 0 else 0.0
    
    print("=" * 60)
    print("EXACT TOKEN MATCH STATISTICS")
    print("=" * 60)
    print(f"Total samples: {num_samples}")
    print(f"Exact matches: {num_exact_matches} ({match_rate:.1%})")
    print(f"Mismatches: {num_mismatches} ({1-match_rate:.1%})")
    
    # Analyze token-level differences
    if mismatches:
        print("\n" + "=" * 60)
        print("TOKEN-LEVEL DIFFERENCE ANALYSIS")
        print("=" * 60)
        
        total_diff_lengths = []
        common_prefix_lengths = []
        
        for record in mismatches:
            spec_tokens = record.get("tokens_spec_decode", [])
            baseline_tokens = record.get("tokens_standard_decode", [])
            
            # Find common prefix
            common_prefix = 0
            min_len = min(len(spec_tokens), len(baseline_tokens))
            for i in range(min_len):
                if spec_tokens[i] == baseline_tokens[i]:
                    common_prefix += 1
                else:
                    break
            
            common_prefix_lengths.append(common_prefix)
            
            # Calculate length difference
            length_diff = abs(len(spec_tokens) - len(baseline_tokens))
            total_diff_lengths.append(length_diff)
        
        if common_prefix_lengths:
            avg_prefix = sum(common_prefix_lengths) / len(common_prefix_lengths)
            max_prefix = max(common_prefix_lengths)
            min_prefix = min(common_prefix_lengths)
            
            print(f"\nCommon Prefix (where tokens start to differ):")
            print(f"  Average: {avg_prefix:.1f} tokens")
            print(f"  Minimum: {min_prefix} tokens")
            print(f"  Maximum: {max_prefix} tokens")
        
        if total_diff_lengths:
            avg_diff = sum(total_diff_lengths) / len(total_diff_lengths)
            print(f"\nLength Differences:")
            print(f"  Average: {avg_diff:.1f} tokens")
            print(f"  Maximum: {max(total_diff_lengths)} tokens")
    
    # Show examples
    if show_examples_bool:
        print("\n" + "=" * 60)
        print("EXAMPLE OUTPUTS")
        print("=" * 60)
        
        # Show some exact matches
        if exact_matches:
            print(f"\n✅ EXACT MATCHES (showing {min(max_examples, len(exact_matches))} examples):")
            print("-" * 60)
            for i, record in enumerate(exact_matches[:max_examples]):
                print(f"\nExample {i+1}:")
                print(f"  Question: {record.get('question', 'N/A')[:100]}...")
                print(f"  Output: {record.get('output_spec_decode', 'N/A')[:150]}...")
                print(f"  ✓ Both methods produced identical tokens")
        
        # Show some mismatches
        if mismatches:
            print(f"\n❌ MISMATCHES (showing {min(max_examples, len(mismatches))} examples):")
            print("-" * 60)
            for i, record in enumerate(mismatches[:max_examples]):
                print(f"\nExample {i+1}:")
                print(f"  Question: {record.get('question', 'N/A')[:100]}...")
                print(f"  Speculative: {record.get('output_spec_decode', 'N/A')[:150]}...")
                print(f"  Baseline:    {record.get('output_standard_decode', 'N/A')[:150]}...")
                
                # Show token-level info
                spec_tokens = record.get("tokens_spec_decode", [])
                baseline_tokens = record.get("tokens_standard_decode", [])
                common_prefix = 0
                min_len = min(len(spec_tokens), len(baseline_tokens))
                for j in range(min_len):
                    if spec_tokens[j] == baseline_tokens[j]:
                        common_prefix += 1
                    else:
                        break
                
                print(f"  Token info: {len(spec_tokens)} vs {len(baseline_tokens)} tokens, "
                      f"match up to position {common_prefix}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exact token match rate: {match_rate:.1%}")
    
    if match_rate == 1.0:
        print("🎉 Perfect! All outputs match exactly.")
    elif match_rate >= 0.95:
        print("✅ Excellent! Nearly all outputs match.")
    elif match_rate >= 0.80:
        print("⚠️  Good, but some differences exist.")
    else:
        print("❌ Significant differences detected. Review mismatches.")
    
    # Save analysis summary
    summary = {
        "total_samples": num_samples,
        "exact_matches": num_exact_matches,
        "mismatches": num_mismatches,
        "match_rate": match_rate,
    }
    
    if mismatches:
        spec_tokens = [r.get("tokens_spec_decode", []) for r in mismatches]
        baseline_tokens = [r.get("tokens_standard_decode", []) for r in mismatches]
        
        common_prefixes = []
        for spec, base in zip(spec_tokens, baseline_tokens):
            prefix = 0
            min_len = min(len(spec), len(base))
            for i in range(min_len):
                if spec[i] == base[i]:
                    prefix += 1
                else:
                    break
            common_prefixes.append(prefix)
        
        summary.update({
            "avg_common_prefix": sum(common_prefixes) / len(common_prefixes) if common_prefixes else 0,
            "avg_length_diff": sum(abs(len(s) - len(b)) for s, b in zip(spec_tokens, baseline_tokens)) / len(mismatches),
        })
    
    summary_path = "/checkpoints/output_verification/analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis summary saved to: {summary_path}")
    checkpoint_volume.commit()


@app.local_entrypoint()
def main():
    print("Output Verification Analysis")
    print("=" * 60)
    print()
    print("Usage:")
    print("  modal run modal_analyze_outputs.py::analyze_output_verification")
    print()
    print("Optional parameters:")
    print("  --output-file: Path to output comparison JSON (default: /checkpoints/output_verification/output_comparison.json)")
    print("  --show-examples: Show example matches/mismatches (default: True)")
    print("  --max-examples: Maximum examples to show (default: 10)")
    print()
    print("The script will analyze the output comparison data and report:")
    print("  - Exact token match statistics")
    print("  - Token-level difference analysis")
    print("  - Example outputs (matches and mismatches)")
    print("  - Summary with recommendations")

