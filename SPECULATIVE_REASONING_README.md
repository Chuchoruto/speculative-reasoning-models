# Speculative Decoding with Coconut: Complete Workflow Guide

This guide documents the complete workflow for training Coconut models, collecting draft training data, training draft models, and running speculative decoding evaluation on the **ProntoQA dataset**. The workflow is documented for both **standard GPT2** (124M parameters) and **GPT2-medium** (355M parameters) models.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Standard GPT2 Workflow](#standard-gpt2-workflow)
   - [Step 1: Train Coconut Model](#step-1-train-coconut-model-standard-gpt2)
   - [Step 2: Collect Draft Training Data](#step-2-collect-draft-training-data-standard-gpt2)
   - [Step 3: Train Draft Model](#step-3-train-draft-model-standard-gpt2)
   - [Step 4: Run Speculative Decoding](#step-4-run-speculative-decoding-standard-gpt2)
3. [GPT2-Medium Workflow](#gpt2-medium-workflow)
   - [Step 1: Train Coconut Model](#step-1-train-coconut-model-gpt2-medium)
   - [Step 2: Collect Draft Training Data](#step-2-collect-draft-training-data-gpt2-medium)
   - [Step 3: Train Draft Model](#step-3-train-draft-model-gpt2-medium)
   - [Step 4: Run Speculative Decoding](#step-4-run-speculative-decoding-gpt2-medium)
4. [Downloading Results](#downloading-results)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Setup

1. **Install Modal CLI and authenticate:**
   ```bash
   pip install modal
   modal token new
   ```

2. **Create WandB account and attach secret to Modal:**
   - Create a WandB account at https://wandb.ai
   - Create a Modal secret with your WandB API key:
     ```bash
     modal secret create wandb WANDB_API_KEY=your_api_key_here
     ```

3. **Ensure data files are present:**
   - `data/prontoqa_train.json`
   - `data/prontoqa_valid.json`
   - `data/prontoqa_test.json`

---

## Standard GPT2 Workflow

### Step 1: Train Coconut Model (Standard GPT2)

Train the Coconut model on the ProntoQA dataset using the standard GPT2 model (124M parameters).

#### Command:

```bash
modal run modal_coconut_training.py::train_prontoqa_coconut
```

#### Optional: Train from CoT checkpoint:

```bash
modal run modal_coconut_training.py::train_prontoqa_coconut \
  --cot-checkpoint-path '/checkpoints/prontoqa-cot/checkpoint_X'
```

#### What This Does:
- Trains Coconut model on ProntoQA dataset
- Uses config file: `args/prontoqa_coconut-final.yaml`
- Uses 4x A100 GPUs for distributed training
- Saves checkpoints to `/checkpoints/prontoqa-coconut-final/` in Modal volume
- Model ID: `openai-community/gpt2` (124M parameters, 768 hidden dim)
- Training time: ~4-8 hours (depending on epochs)

#### Check Available Checkpoints:

```bash
modal run modal_download.py::list_available_checkpoints
```

Look for checkpoints in `/checkpoints/prontoqa-coconut-final/` (e.g., `checkpoint_50`).

---

### Step 2: Collect Draft Training Data (Standard GPT2)

Collect latent thought vectors and logits from the trained Coconut model. This data will be used to train the smaller draft model.

#### Command for Training Data:

```bash
modal run modal_data_collection.py::collect_draft_training_data \
  --checkpoint-path '/checkpoints/prontoqa-coconut-final/checkpoint_50' \
  --output-filename 'prontoqa_train_draft_training_data_standard.json' \
  --max-samples 500 \
  --data-path 'data/prontoqa_train.json'
```

#### Command for Validation Data:

```bash
modal run modal_data_collection.py::collect_draft_training_data \
  --checkpoint-path '/checkpoints/prontoqa-coconut-final/checkpoint_50' \
  --output-filename 'prontoqa_valid_draft_training_data_standard.json' \
  --max-samples 500 \
  --data-path 'data/prontoqa_valid.json'
```

#### Parameters:
- `--checkpoint-path`: Path to your Coconut checkpoint (required)
- `--output-filename`: Name of output JSON metadata file (required)
- `--max-samples`: Maximum samples to collect (default: `None`, collects all)
- `--data-path`: Path to dataset JSON file (required)

#### What This Does:
- Runs Coconut model inference on the dataset
- Collects latent thought vectors (768-dim) and token logits
- Stores large vectors in compressed NPZ files (one per sample)
- Stores metadata (token IDs, positions, etc.) in JSON
- Saves to `/checkpoints/draft_data_pqa/` in Modal volume
- Auto-detects model configuration from checkpoint path

#### Verify Data Collection:

```bash
# List collected data files
modal run modal_download.py::list_draft_training_data
```

#### (Optional) Download Data Locally:

```bash
# Download JSON and all NPZ files
python download_draft_data.py \
  --filename "prontoqa_train_draft_training_data.json" \
  --local-path "./downloaded_checkpoints" \
  --volume-path "draft_data_pqa"
```

---

### Step 3: Train Draft Model (Standard GPT2)

Train a smaller draft model (GPT-2 Mini, 512 hidden dim) to mimic the Coconut model's behavior. The draft model learns to predict both latent thoughts (using cosine similarity loss) and token logits (using KL divergence loss).

#### Command:

```bash
modal run modal_draft_training.py::train_draft_model \
  --data-json-filename 'prontoqa_train_draft_training_data_standard.json' \
  --val-json-filename 'prontoqa_valid_draft_training_data_standard.json' \
  --batch-size 64 \
  --num-epochs 20 \
  --lr 0.0005 \
  --weight-decay 0.01 \
  --kl-weight 1.0 \
  --cosine-weight 1.0 \
  --temperature 4.0 \
  --gradient-accumulation-steps 1 \
  --warmup-steps 20 \
  --wandb-project 'draft-model-training-prontoqa-standard'
```

#### Parameters:
- `--data-json-filename`: JSON metadata file name in `/checkpoints/draft_data_pqa/` (required)
- `--val-json-filename`: Validation JSON metadata file name (optional, recommended)
- `--batch-size`: Batch size per GPU (default: `32`, effective batch = batch_size × 4 GPUs)
- `--num-epochs`: Number of training epochs (default: `10`)
- `--lr`: Learning rate (default: `5e-4`)
- `--weight-decay`: Weight decay for regularization (default: `0.01`)
- `--kl-weight`: Weight for KL divergence loss (logits) (default: `1.0`)
- `--cosine-weight`: Weight for cosine similarity loss (latent thoughts) (default: `1.0`)
- `--temperature`: Temperature scaling for KL divergence (default: `4.0`)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: `1`)
- `--warmup-steps`: Warmup steps for learning rate (default: `20`)
- `--wandb-project`: WandB project name (default: `draft-model-training`)

#### What This Does:
- Loads training data from `/checkpoints/draft_data_pqa/` in Modal volume
- Trains draft model with mixed loss:
  - **KL divergence loss** for logit distributions
  - **Cosine similarity loss** for latent thought vectors
- Uses 4x A100 GPUs for distributed training
- Saves checkpoints to `/checkpoints/draft_model_final/` in Modal volume
- Runs validation after each epoch (if `--val-json-filename` provided)
- Logs metrics to WandB
- Training time: ~2-4 hours for 20 epochs

#### Check Available Checkpoints:

```bash
# List all draft model checkpoints
python list_draft_checkpoints.py
```

Look for checkpoints like `draft_model_epoch_20.pt` (where 20 is the epoch number).

#### Expected Loss Values:
After training, you should see:
- **Cosine Loss**: ~0.005 (latent thought vector matching)
- **KL Loss**: ~0.2 (logit distribution matching)

Lower is better. If losses are much higher, consider:
- Adjusting learning rate (`--lr 0.0005`)
- Increasing temperature (`--temperature 4.0`)
- Training for more epochs (`--num-epochs 20-30`)

---

### Step 4: Run Speculative Decoding (Standard GPT2)

Run speculative decoding evaluation to compare the draft model + Coconut model against baseline autoregressive decoding.

#### Command:

```bash
modal run modal_speculative_decode.py::evaluate_speculative_decoding \
  --draft-checkpoint-path '/checkpoints/draft_model_final/draft_model_epoch_20.pt' \
  --target-checkpoint-path '/checkpoints/prontoqa-coconut-final/checkpoint_50' \
  --data-path 'data/prontoqa_test.json' \
  --num-latent-thoughts 6 \
  --gamma 6 \
  --max-new-tokens 50 \
  --similarity-threshold 0.9 \
  --max-samples 100 \
  --clock-run True \
  --tokens-speculative False
```

#### Parameters:
- `--draft-checkpoint-path`: Path to draft model checkpoint (required)
- `--target-checkpoint-path`: Path to Coconut (target) model checkpoint (required)
- `--data-path`: Path to test dataset (default: `data/prontoqa_test.json`)
- `--num-latent-thoughts`: Number of latent thoughts to generate (default: `6`)
- `--gamma`: Number of draft tokens per round (for tokens). Note: latent thoughts are processed all at once (default: `6`)
- `--max-new-tokens`: Maximum tokens to generate per sample (default: `50`)
- `--similarity-threshold`: Cosine similarity threshold for latent thought acceptance (default: `0.9`)
- `--max-samples`: Maximum samples to evaluate (default: `100`)
- `--clock-run`: Enable wallclock timing comparison (default: `"False"`)
  - When `True`, also runs baseline autoregressive decoding for comparison
- `--tokens-speculative`: If `False`, uses speculative decoding for latent thoughts only, then autoregressive token generation with target model (default: `"False"`)
- `--record-output`: Record generated outputs for comparison (default: `"False"`)
  - Requires `clock-run=True`
  - Saves outputs to `/checkpoints/output_verification/output_comparison.json`

#### What This Does:

**Speculative Decoding Process:**
1. Generates and verifies latent thoughts (6 latent tokens)
   - Draft model generates latent thoughts sequentially
   - Target model verifies all latent thoughts in parallel (single forward pass)
   - Compare via cosine similarity (threshold: 0.9)
2. Generates tokens autoregressively using target model only
   - Uses normal autoregressive generation (not speculative)
   - Counts target model calls for accurate comparison

**Metrics Reported:**
- Latent thought acceptance rate
- Model call counts (draft vs target)
- Baseline model call counts (for comparison)
- Estimated speedup (based on model calls)
- Actual wallclock speedup (if `--clock-run True`)
- Generation statistics
- Output comparison (if `--record-output True`)

**Results Saved:**
- `/checkpoints/speculative_decoding_results.json` - Main results with all metrics
- `/checkpoints/output_verification/output_comparison.json` - Sample outputs (if `--record-output True`)

---

## GPT2-Medium Workflow

### Step 1: Train Coconut Model (GPT2-Medium)

Train the Coconut model on the ProntoQA dataset using the GPT2-medium model (355M parameters).

#### Command:

```bash
modal run modal_coconut_training.py::train_prontoqa_coconut_medium
```

#### Optional: Train from CoT checkpoint:

```bash
modal run modal_coconut_training.py::train_prontoqa_coconut_medium \
  --cot-checkpoint-path '/checkpoints/path/to/cot/checkpoint'
```

#### Optional: Resume from existing checkpoint:

```bash
modal run modal_coconut_training.py::train_prontoqa_coconut_medium \
  --resume-checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_25'
```

#### What This Does:
- Trains Coconut model on ProntoQA dataset
- Uses config file: `args/prontoqa_coconut.yaml` with overrides
- Uses 4x A100-80GB GPUs for distributed training
- Saves checkpoints to `/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/` in Modal volume
- Model ID: `openai-community/gpt2-medium` (355M parameters, 1024 hidden dim)
- Training time: ~4-8 hours (depending on epochs)

#### Check Available Checkpoints:

```bash
modal run modal_download.py::list_checkpoints_in_path_medium \
  --checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium'
```

Look for checkpoints like `checkpoint_25`, `checkpoint_50`, etc.

---

### Step 2: Collect Draft Training Data (GPT2-Medium)

Collect latent thought vectors and logits from the trained GPT2-medium Coconut model.

#### Command for Training Data:

```bash
modal run modal_data_collection.py::collect_draft_training_data_medium \
  --checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_50' \
  --output-filename 'prontoqa_train_draft_training_data_medium.json' \
  --max-samples 500 \
  --data-path 'data/prontoqa_train.json'
```

#### Command for Validation Data:

```bash
modal run modal_data_collection.py::collect_draft_training_data_medium \
  --checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_50' \
  --output-filename 'prontoqa_valid_draft_training_data_medium.json' \
  --max-samples 500 \
  --data-path 'data/prontoqa_valid.json'
```

#### Parameters:
- `--checkpoint-path`: Path to your GPT2-medium Coconut checkpoint (required)
- `--output-filename`: Name of output JSON metadata file (required)
- `--max-samples`: Maximum samples to collect (default: `None`, collects all)
- `--data-path`: Path to dataset JSON file (required)

#### What This Does:
- Runs GPT2-medium Coconut model inference on the dataset
- Collects latent thought vectors (1024-dim) and token logits
- Stores large vectors in compressed NPZ files (one per sample)
- Stores metadata (token IDs, positions, etc.) in JSON
- Saves to `/checkpoints/gpt2medium-prontoqa-checkpoints/draft_data_pqa/` in Modal volume
- Auto-detects model configuration (gpt2-medium, 1024 hidden dim)

#### Verify Data Collection:

The data will be saved in the GPT2-medium volume. You can verify by checking the volume contents.

---

### Step 3: Train Draft Model (GPT2-Medium)

Train a smaller draft model (GPT-2 Mini, 512 hidden dim) to mimic the GPT2-medium Coconut model's behavior. The draft model includes a linear projection layer to map from 512-dim to 1024-dim for latent thought comparison.

#### Command:

```bash
modal run modal_draft_training.py::train_draft_model_medium \
  --data-json-filename 'prontoqa_train_draft_training_data_medium.json' \
  --val-json-filename 'prontoqa_valid_draft_training_data_medium.json' \
  --batch-size 64 \
  --num-epochs 20 \
  --lr 0.0005 \
  --weight-decay 0.01 \
  --kl-weight 1.0 \
  --cosine-weight 1.0 \
  --temperature 4.0 \
  --gradient-accumulation-steps 1 \
  --warmup-steps 20 \
  --wandb-project 'draft-model-training-prontoqa-medium'
```

#### Parameters:
- Same as standard GPT2 training (see above)
- The model automatically uses GPT2-medium teacher model (1024 hidden dim)
- Draft model projection: 512 → 1024

#### What This Does:
- Loads training data from `/checkpoints/gpt2medium-prontoqa-checkpoints/draft_data_pqa/` in Modal volume
- Trains draft model with mixed loss:
  - **KL divergence loss** for logit distributions
  - **Cosine similarity loss** for latent thought vectors (projected to 1024-dim)
- Uses 4x A100 GPUs for distributed training
- Saves checkpoints to `/checkpoints/gpt2medium-prontoqa-checkpoints/draft_model_final/` in Modal volume
- Runs validation after each epoch (if `--val-json-filename` provided)
- Logs metrics to WandB
- Training time: ~2-4 hours for 20 epochs

#### Check Available Checkpoints:

Checkpoints are saved in the GPT2-medium volume at:
`/checkpoints/gpt2medium-prontoqa-checkpoints/draft_model_final/draft_model_epoch_N.pt`

---

### Step 4: Run Speculative Decoding (GPT2-Medium)

Run speculative decoding evaluation using GPT2-medium models.

#### Command:

```bash
modal run modal_speculative_decode.py::evaluate_speculative_decoding_medium \
  --draft-checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/draft_model_final/draft_model_epoch_20.pt' \
  --target-checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_50' \
  --data-path 'data/prontoqa_test.json' \
  --num-latent-thoughts 6 \
  --gamma 6 \
  --max-new-tokens 50 \
  --similarity-threshold 0.9 \
  --max-samples 100 \
  --clock-run True \
  --tokens-speculative False
```

#### Parameters:
- Same as standard GPT2 evaluation (see above)
- Uses GPT2-medium volume (`coconut-checkpoints-gpt2-medium`)
- Automatically detects model configuration (1024 hidden dim)

#### What This Does:
- Same speculative decoding process as standard GPT2
- Uses GPT2-medium models (larger, more accurate)
- Results saved to GPT2-medium Modal volume

---

## Downloading Results

### Download Speculative Decoding Results

Download the evaluation results and output comparisons from Modal volumes.

#### Standard GPT2 Results:

```bash
python download_results.py \
  --local-path "./downloaded_checkpoints" \
  --volume-name "coconut-checkpoints"
```

#### GPT2-Medium Results:

```bash
python download_results.py \
  --local-path "./downloaded_checkpoints_medium" \
  --volume-name "coconut-checkpoints-gpt2-medium"
```

#### What Gets Downloaded:
- `speculative_decoding_results.json` - Main results with all metrics
- `output_verification/output_comparison.json` - Sample outputs comparison (if recorded)
- `output_verification/analysis_summary.json` - Analysis summary (if analyzed)

---

## Troubleshooting

### Issue: Cannot find checkpoints

**Solution:**
```bash
# List all available checkpoints (standard GPT2)
modal run modal_download.py::list_available_checkpoints

# List GPT2-medium checkpoints
modal run modal_download.py::list_checkpoints_in_path_medium \
  --checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium'
```

### Issue: Draft model training losses are too high

**Solutions:**
- Adjust learning rate: `--lr 0.0005`
- Increase temperature: `--temperature 4.0`
- Train for more epochs: `--num-epochs 20-30`
- Adjust loss weights: `--kl-weight 1.0 --cosine-weight 1.0`
- Check that expected losses are KL ~ 0.2 and Cosine ~ 0.005

### Issue: Speculative decoding shows low acceptance rates

**Solutions:**
- Lower similarity threshold: `--similarity-threshold 0.85`
- Train draft model longer or with better hyperparameters
- Check that draft model losses are reasonable (Cosine ~ 0.005, KL ~ 0.2)

### Issue: Modal volume issues

**Solution:**
```bash
# Check volume contents (standard GPT2)
modal volume ls coconut-checkpoints

# Check volume contents (GPT2-medium)
modal volume ls coconut-checkpoints-gpt2-medium

# Download files directly
modal volume get coconut-checkpoints /checkpoints/path/to/file ./local/path
```

### Issue: WandB not logging

**Solutions:**
- Verify WandB secret is attached: `modal secret list`
- Check WandB project name matches your account
- Ensure `wandb_project` parameter is correct

---

## Quick Reference: All Commands

### Standard GPT2 (ProntoQA) - Complete Workflow

```bash
# 1. Train Coconut model
modal run modal_coconut_training.py::train_prontoqa_coconut

# 2. Collect draft training data
modal run modal_data_collection.py::collect_draft_training_data \
  --checkpoint-path '/checkpoints/prontoqa-coconut-final/checkpoint_50' \
  --output-filename 'prontoqa_train_draft_training_data_standard.json' \
  --data-path 'data/prontoqa_train.json'

modal run modal_data_collection.py::collect_draft_training_data \
  --checkpoint-path '/checkpoints/prontoqa-coconut-final/checkpoint_50' \
  --output-filename 'prontoqa_valid_draft_training_data_standard.json' \
  --data-path 'data/prontoqa_valid.json'

# 3. Train draft model
modal run modal_draft_training.py::train_draft_model \
  --data-json-filename 'prontoqa_train_draft_training_data_standard.json' \
  --val-json-filename 'prontoqa_valid_draft_training_data_standard.json' \
  --batch-size 64 \
  --num-epochs 20 \
  --lr 0.0005 \
  --kl-weight 1.0 \
  --cosine-weight 1.0 \
  --temperature 4.0 \
  --gradient-accumulation-steps 1

# 4. Run speculative decoding
modal run modal_speculative_decode.py::evaluate_speculative_decoding \
  --draft-checkpoint-path '/checkpoints/draft_model_final/draft_model_epoch_20.pt' \
  --target-checkpoint-path '/checkpoints/prontoqa-coconut-final/checkpoint_50' \
  --data-path 'data/prontoqa_test.json' \
  --gamma 6 \
  --max-samples 100 \
  --clock-run True \
  --tokens-speculative False
```

### GPT2-Medium (ProntoQA) - Complete Workflow

```bash
# 1. Train Coconut model
modal run modal_coconut_training.py::train_prontoqa_coconut_medium

# 2. Collect draft training data
modal run modal_data_collection.py::collect_draft_training_data_medium \
  --checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_50' \
  --output-filename 'prontoqa_train_draft_training_data_medium.json' \
  --data-path 'data/prontoqa_train.json'

modal run modal_data_collection.py::collect_draft_training_data_medium \
  --checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_50' \
  --output-filename 'prontoqa_valid_draft_training_data_medium.json' \
  --data-path 'data/prontoqa_valid.json'

# 3. Train draft model
modal run modal_draft_training.py::train_draft_model_medium \
  --data-json-filename 'prontoqa_train_draft_training_data_medium.json' \
  --val-json-filename 'prontoqa_valid_draft_training_data_medium.json' \
  --batch-size 64 \
  --num-epochs 20 \
  --lr 0.0005 \
  --kl-weight 1.0 \
  --cosine-weight 1.0 \
  --temperature 4.0 \
  --gradient-accumulation-steps 1

# 4. Run speculative decoding
modal run modal_speculative_decode.py::evaluate_speculative_decoding_medium \
  --draft-checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/draft_model_final/draft_model_epoch_20.pt' \
  --target-checkpoint-path '/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_50' \
  --data-path 'data/prontoqa_test.json' \
  --gamma 6 \
  --max-samples 100 \
  --clock-run True \
  --tokens-speculative False
```

### Utility Commands

```bash
# List checkpoints (standard GPT2)
modal run modal_download.py::list_available_checkpoints

# List checkpoints (GPT2-medium)
modal run modal_download.py::list_checkpoints_in_path_medium

# Download results (standard GPT2)
python download_results.py --volume-name "coconut-checkpoints"

# Download results (GPT2-medium)
python download_results.py --volume-name "coconut-checkpoints-gpt2-medium"
```

---

## File Structure

```
.
├── data/
│   ├── prontoqa_train.json      # Training dataset
│   ├── prontoqa_valid.json      # Validation dataset
│   └── prontoqa_test.json       # Test dataset
├── args/
│   ├── prontoqa_coconut.yaml           # GPT2-medium Coconut config
│   └── prontoqa_coconut-final.yaml     # Standard GPT2 Coconut config
├── modal_coconut_training.py          # Coconut training script
├── modal_data_collection.py           # Draft data collection script
├── modal_draft_training.py            # Draft model training script
├── modal_speculative_decode.py        # Speculative decoding evaluation
├── download_results.py                # Download evaluation results
├── download_draft_data.py              # Download draft training data
├── list_draft_checkpoints.py          # List draft checkpoints
└── SPECULATIVE_REASONING_README.md     # This file
```

---

## Notes

1. **Volume Storage**: 
   - Standard GPT2: All checkpoints and data stored in Modal volume `coconut-checkpoints`
   - GPT2-Medium: All checkpoints and data stored in Modal volume `coconut-checkpoints-gpt2-medium`
   - Volumes persist across runs

2. **GPU Requirements**:
   - Coconut training: 4x A100 GPUs (standard) or 4x A100-80GB GPUs (medium)
   - Data collection: 1x A100 GPU
   - Draft training: 4x A100 GPUs
   - Speculative decoding: 1x A100 GPU

3. **Training Time Estimates**:
   - Coconut training: 4-8 hours
   - Data collection (500 samples): 15-30 minutes
   - Draft training (20 epochs): 2-4 hours
   - Speculative decoding (100 samples): 10-30 minutes

4. **Checkpoint Paths**:
   - **Standard GPT2 Coconut**: `/checkpoints/prontoqa-coconut-final/checkpoint_N`
   - **Standard GPT2 Draft**: `/checkpoints/draft_model_final/draft_model_epoch_N.pt`
   - **GPT2-Medium Coconut**: `/checkpoints/gpt2medium-prontoqa-checkpoints/prontoqa-coconut-gpt2-medium/checkpoint_N`
   - **GPT2-Medium Draft**: `/checkpoints/gpt2medium-prontoqa-checkpoints/draft_model_final/draft_model_epoch_N.pt`

5. **Loss Functions**:
   - **KL Divergence Loss**: Measures difference between draft and target logit distributions
   - **Cosine Similarity Loss**: Measures difference between draft and target latent thought vectors (after projection)
   - **No Cross-Entropy Loss**: Draft model only learns to mimic Coconut's representations, not next-token prediction

6. **Best Practices**:
   - Always verify checkpoints exist before using them
   - Monitor WandB for training progress
   - Start with small `--max-samples` for testing
   - Use `--clock-run True` only when you need actual speedup measurements
   - Use `--tokens-speculative False` to focus on latent thought speedup (recommended)

---

## Questions or Issues?

- Check the [Troubleshooting](#troubleshooting) section
- Review Modal logs: `modal app logs <app-name>`
- Check WandB dashboard for training metrics
- Verify data paths and checkpoint paths are correct
