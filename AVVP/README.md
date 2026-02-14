# AVVP: Audio-Visual Event Localization

A simplified project for Audio-Visual Event (AVE) localization using LanguageBind/CLIP models.

## Structure
- `main.py`: Main entry point for training/inference.
- `models.py`: Model definitions (LanguageBind, CLIP_CLAP) with caching.
- `dataset.py`: Video dataset loader with 10s normalization.
- `transforms.py`: Data transformations (includes Resize for video).
- `embedding_cache.py`: Feature caching utility.
- `utils.py`: Helper functions.
- `eval_metrics.py`: Evaluation scripts (ported from original).

## Usage

### 1. Environment
Ensure you have the necessary dependencies installed (e.g., `torch`, `transformers`, `decord`, `librosa`, `tqdm`).
If you have an existing environment (e.g., `ava`), activate it:
```bash
conda activate ava
```

### 2. Run Inference
You can run the script directly using `python` or use the provided `run.sh` script.

**Using Python:**
```bash
python main.py \
  --video_dir /path/to/videos \
  --audio_dir /path/to/audio \
  --backbone language_bind \
  --dataset LLP \
  --threshold 0.5 \
  --gpu_id 0  # Specify which GPU to use (default: 0)
```

**Optons:**
- `--gpu_id`: Specify the GPU index to use (e.g., `0`, `1`). Default is `0`.
- `--backbone`: Choose between `language_bind` or `clip_clap`.
- `--threshold`: Similarity threshold for event detection (default: `0.5`).

### 3. Output
- `results/candidates_{backbone}.json`: Raw detection results.
- `segment_analysis/segment_details_{backbone}.txt`: Human-readable segment details.
