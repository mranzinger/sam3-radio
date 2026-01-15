# SA-CO/Gold Evaluation Script

This script (`eval_saco_gold.py`) evaluates the SA-CO/Gold benchmark using either the SAM3 vision encoder or the RADIO adaptor. It generates predictions for all 7 SA-CO/Gold subsets and evaluates them using the cgF1 metric.

## Features

- **Flexible encoder selection**: Use either original SAM3 vision encoder or RADIO adaptor
- **Automatic prediction generation**: Processes images and generates COCO-format predictions
- **Comprehensive evaluation**: Evaluates all 7 subsets with cgF1, IL_MCC, and PM_F1 metrics
- **Single subset evaluation**: Option to evaluate just one subset
- **Resume capability**: Can skip inference and use existing predictions

## Dataset Structure

The script expects the SA-CO/Gold data to be organized as follows:

```
/path/to/saco/gold/
├── gold_metaclip_merged_a_release_test.json
├── gold_metaclip_merged_b_release_test.json
├── gold_metaclip_merged_c_release_test.json
├── gold_sa1b_merged_a_release_test.json
├── gold_sa1b_merged_b_release_test.json
├── gold_sa1b_merged_c_release_test.json
├── gold_crowded_merged_a_release_test.json
├── gold_crowded_merged_b_release_test.json
├── gold_crowded_merged_c_release_test.json
├── ... (other subset annotations)
├── metaclip_images/
│   └── (MetaCLIP images)
└── sa1b_images/
    └── (SA-1B images)
```

## Usage Examples

### 1. Evaluate with RADIO encoder (single GPU)

```bash
python eval_saco_gold.py \
    --radio-model-version radio_v2.5-h \
    --data-root /lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/datasets/segmentation/sa-co/gold \
    --output-dir ./saco_gold_radio_predictions \
    --confidence-threshold 0.3
```

### 1b. Evaluate with RADIO encoder (multi-GPU)

```bash
python eval_saco_gold.py \
    --radio-model-version radio_v2.5-h \
    --data-root /lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/datasets/segmentation/sa-co/gold \
    --output-dir ./saco_gold_radio_predictions \
    --confidence-threshold 0.3 \
    --multi-gpu \
    --batch-size 16
```

### 2. Evaluate with original SAM3 encoder

```bash
python eval_saco_gold.py \
    --skip-radio \
    --data-root /lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/datasets/segmentation/sa-co/gold \
    --output-dir ./saco_gold_sam3_predictions \
    --confidence-threshold 0.3
```

### 3. Evaluate single subset only

```bash
python eval_saco_gold.py \
    --radio-model-version radio_v2.5-h \
    --data-root /lustre/fsw/portfolios/llmservice/projects/llmservice_fm_vision/datasets/segmentation/sa-co/gold \
    --subset metaclip_nps
```

### 4. Use custom paths for images and annotations

```bash
python eval_saco_gold.py \
    --radio-model-version radio_v2.5-h \
    --gt-dir /path/to/annotations \
    --metaclip-img-dir /path/to/metaclip/images \
    --sa1b-img-dir /path/to/sa1b/images \
    --output-dir ./saco_gold_predictions
```

### 5. Skip inference and only evaluate existing predictions

```bash
python eval_saco_gold.py \
    --data-root /path/to/saco/gold \
    --output-dir ./saco_gold_predictions \
    --skip-inference
```

## Command-Line Arguments

### Required Arguments

- `--data-root`: Root directory containing SA-CO/Gold data (annotations and images)

### Encoder Selection (choose one)

- `--radio-model-version`: RADIO model version (e.g., 'radio_v2.5-h', 'radio_v2.1')
- `--skip-radio`: Use original SAM3 encoder instead of RADIO

### Optional Arguments

- `--gt-dir`: Directory containing ground truth annotations (default: data-root/gt-annotations)
- `--metaclip-img-dir`: Directory containing MetaCLIP images (default: data-root/metaclip_images)
- `--sa1b-img-dir`: Directory containing SA-1B images (default: data-root/sa1b_images)
- `--output-dir`: Directory to save predictions and results (default: ./saco_gold_predictions)
- `--sam3-checkpoint`: Path to SAM3 checkpoint (default: downloads from HuggingFace)
- `--confidence-threshold`: Confidence threshold for predictions (default: 0.3)
- `--device`: Device to use - "cuda" or "cpu" (default: cuda)
- `--vitdet`: Run RADIO in ViTDet mode with specified window size
- `--subset`: Evaluate only a specific subset (choices: metaclip_nps, sa1b_nps, crowded, fg_food, fg_sports_equipment, attributes, wiki_common)
- `--skip-inference`: Skip inference and only run evaluation (assumes predictions already exist)
- `--batch-size`: Batch size for inference (default: 1). Use larger values (e.g., 8, 16, 32) for faster processing on each GPU
- `--multi-gpu`: Distribute subsets across all available GPUs for parallel processing (automatically detects GPU count)

## SA-CO/Gold Subsets

The benchmark consists of 7 subsets:

1. **metaclip_nps**: MetaCLIP Captioner noun phrases
2. **sa1b_nps**: SA-1B Captioner noun phrases (uses SA-1B images)
3. **crowded**: Crowded scenes
4. **fg_food**: Wiki-Food/Drink fine-grained categories
5. **fg_sports_equipment**: Wiki-Sports Equipment fine-grained categories
6. **attributes**: Attribute descriptions
7. **wiki_common**: Wiki-Common1K categories

All subsets except `sa1b_nps` use MetaCLIP images.

## Output Structure

The script creates the following output structure:

```
output_dir/
├── gold_metaclip_nps/
│   └── coco_predictions_segm.json
├── gold_sa1b_nps/
│   └── coco_predictions_segm.json
├── gold_crowded/
│   └── coco_predictions_segm.json
├── ... (other subsets)
└── results_summary.json
```

The `results_summary.json` file contains:
- Per-subset metrics (cgF1, IL_MCC, PM_F1)
- Average metrics across all subsets

## Metrics

The script reports three key metrics:

- **cgF1**: Concept-Group F1 score (main metric for SA-CO/Gold)
- **IL_MCC**: Image-Level Matthews Correlation Coefficient
- **PM_F1**: Positive Micro F1 score

Higher values are better for all metrics.

## Performance Tips

1. **Use multi-GPU for faster inference**: The `--multi-gpu` flag distributes subsets across all available GPUs, processing them in parallel. Each GPU runs a separate process with its own model instance.
2. **Batch processing**: Use `--batch-size 4` or higher to process multiple images in parallel on each GPU
3. Use `--confidence-threshold 0.3` for best results (lower threshold captures more instances)
4. For faster evaluation during development, test on a single subset first using `--subset`
5. The script automatically uses TF32 and bfloat16 on compatible GPUs for faster inference
6. If you have limited RAM, predictions are saved per-subset so you can process them one at a time

### Multi-GPU Processing

The `--multi-gpu` flag distributes the 7 subsets across available GPUs in round-robin fashion:

```bash
# Example with 4 GPUs:
# GPU 0: metaclip_nps, fg_food
# GPU 1: sa1b_nps, fg_sports_equipment
# GPU 2: crowded, attributes
# GPU 3: wiki_common
```

Each GPU processes its assigned subsets independently with its own model instance. With 4 GPUs, you can expect roughly **3-4x speedup** compared to single GPU inference. The actual speedup depends on:
- GPU model and memory
- Batch size per GPU
- Subset sizes (some subsets have more images than others)
- Batch size (larger is better, but limited by GPU memory)
- Image size and model complexity
- Number of GPUs

**Recommended configurations:**
- **Single GPU**: `--batch-size 4` (or higher if memory allows)
- **Multi-GPU (auto-detect)**: `--multi-gpu --batch-size 16` (uses all available GPUs)
- **Adjust batch size based on memory**: Start with 16, increase to 32 if you have headroom

## Comparison with demo_sam3_radio.py

This script follows similar argument conventions to `demo_sam3_radio.py`:

- Same `--radio-model-version` and `--skip-radio` flags
- Same `--sam3-checkpoint`, `--device`, and `--vitdet` options
- Similar confidence threshold handling

This makes it easy to switch between demo and evaluation workflows.

## Downloading SA-CO/Gold Data

The GT annotations can be downloaded from:
- [Hugging Face](https://huggingface.co/datasets/facebook/SACo-Gold)
- [Roboflow](https://universe.roboflow.com/sa-co-gold)

MetaCLIP images (6 out of 7 subsets):
- [Roboflow](https://universe.roboflow.com/sa-co-gold/gold-metaclip-merged-a-release-test/)

SA-1B images (sa1b_nps subset only):
- [Roboflow](https://universe.roboflow.com/sa-co-gold/gold-sa-1b-merged-a-release-test/)
- Or download from [SA-1B dataset](https://ai.meta.com/datasets/segment-anything-downloads/) (access link for `sa_co_gold.tar`)

## References

For more information about the SA-CO/Gold benchmark and evaluation protocol, see:
- [SA-CO/Gold README](scripts/eval/gold/README.md)
- [SA-CO/Gold evaluation notebook](examples/saco_gold_silver_eval_example.ipynb)
- SAM3 paper (for cgF1 metric details)
