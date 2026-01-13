# SAM3 with RADIO Vision Encoder Demo

This demo replaces SAM3's default vision encoder with the RADIO model, demonstrating image segmentation with text and visual prompts.

## Overview

The demo script (`demo_sam3_radio.py`) performs the following:
1. Loads a RADIO model from torch.hub with the SAM3 adaptor
2. Replaces SAM3's vision encoder trunk with the RADIO model using `RADIO_Adaptor`
3. Runs segmentation demos with text and box prompts
4. Saves all results to an output folder

## Requirements

```bash
# Install SAM3
pip install git+https://github.com/facebookresearch/sam3.git

# Install additional dependencies
pip install torch torchvision matplotlib pillow einops timm
```

## Usage

### Basic Usage

```bash
python demo_sam3_radio.py \
    --radio-model-version /path/to/radio/checkpoint.pth \
    --output-dir ./output_sam3_radio
```

### Using Custom Image

```bash
python demo_sam3_radio.py \
    --radio-model-version /path/to/radio/checkpoint.pth \
    --image /path/to/your/image.jpg \
    --output-dir ./my_results
```

### All Options

```bash
python demo_sam3_radio.py \
    --radio-model-version /path/to/radio/checkpoint.pth \
    --image /path/to/image.jpg \
    --output-dir ./output \
    --sam3-checkpoint /path/to/sam3/checkpoint.pth \  # Optional: use local SAM3 checkpoint
    --confidence-threshold 0.5 \
    --text-prompt "shoe" \  # Optional: customize text prompt
    --vitdet 16 \  # Optional: run RADIO in ViTDet mode with window size
    --device cuda \
    --skip-radio  # Optional: test with original SAM3 encoder
```

### Command Line Arguments

- `--radio-model-version`: **Required**. Path to RADIO model checkpoint
- `--image`: Input image path (default: uses SAM3's test image)
- `--output-dir`: Output directory for results (default: `./output_sam3_radio`)
- `--sam3-checkpoint`: Local SAM3 checkpoint path (default: downloads from HuggingFace)
- `--confidence-threshold`: Prediction confidence threshold (default: 0.5)
- `--text-prompt`: Text prompt for segmentation (default: "shoe")
- `--vitdet`: Run RADIO in ViTDet mode with specified window size (optional)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--skip-radio`: Skip RADIO replacement to test original SAM3 encoder

## How It Works

### 1. RADIO Model Loading

The RADIO model is loaded using torch.hub with the SAM3 adaptor:

```python
model = torch.hub.load('NVlabs/RADIO', 'radio_model', checkpoint_path, adaptor_names='sam3')
```

This ensures the RADIO model outputs features compatible with the `RADIO_Adaptor`.

### 2. Vision Encoder Replacement

The `replace_sam3_encoder()` function:
1. Loads RADIO model using torch.hub
2. Creates a projection layer (if needed) to match dimensions
3. Creates a `RADIO_Adaptor` that mimics SAM3's ViT trunk output
4. Replaces `sam3_model.backbone.vision_backbone.trunk` with the adaptor

The `RADIO_Adaptor` (from `sam3/sam3_radio_utils.py`):
- Takes student (RADIO) features and projects them to SAM3's expected dimensions
- Reshapes features from `[B, N, C]` to `[B, C, H, W]` format
- Returns features in a list format matching SAM3's ViT trunk output

### 3. Segmentation Demos

The script runs three segmentation demos:

1. **Text Prompt**: Segments objects based on text description ("shoe")
2. **Single Box Prompt**: Uses one bounding box as a visual prompt
3. **Multi-Box Prompt**: Uses multiple boxes with positive/negative labels

## Output Files

The script saves the following images to the output directory:

- `text_prompt_shoe.png`: Text-based segmentation results
- `single_box_input.png`: Input image with box visualization
- `single_box_result.png`: Single box segmentation results
- `multi_box_input.png`: Input image with multiple boxes (green=positive, red=negative)
- `multi_box_result.png`: Multi-box segmentation results

## Key Files

- **`demo_sam3_radio.py`**: Main demo script
- **`sam3/sam3_radio_utils.py`**: Utility functions for RADIO integration (RADIO_Adaptor, model loading, processor creation)

### SAM3 Vision Encoder Structure

```
SAM3Image
├── backbone (SAM3VLBackbone)
│   ├── vision_backbone (Sam3DualViTDetNeck)
│   │   ├── trunk (ViT)  ← This is what we replace with RADIO
│   │   └── neck (FPN layers)
│   └── text_encoder
└── grounding_head
```

### RADIO Integration

```
RADIO_Adaptor
├── student (RADIO wrapped as Student)
├── projection (Linear layer for dimension matching)
└── forward() → [B, C, H, W] features (mimics ViT trunk)
```

The adaptor ensures RADIO's output format matches what SAM3's neck expects.

## Troubleshooting

### CUDA Out of Memory

Try reducing image size or using CPU:
```bash
python demo_sam3_radio.py --radio-model-version path/to/ckpt --device cpu
```

### Missing Dependencies

Ensure all dependencies from `example_sam3_replace.py` are available:
- `einops`
- `timm`
- `student` and `teacher` modules (if separate packages)

### Dimension Mismatch

The script automatically creates a projection layer if RADIO's output dimension doesn't match SAM3's expected dimension. Check the console output for confirmation.

## Example Output

```
Loading RADIO model from /path/to/checkpoint.pth...
RADIO model loaded successfully!
Building SAM3 model...
SAM3 model built successfully!
Replacing SAM3 vision encoder with RADIO...
Vision encoder replaced successfully!
Initializing processor...
Image size: 1500x1125

Running segmentation demos...

=== Text Prompt Demo ===
Saved: output_sam3_radio/text_prompt_shoe.png

=== Single Box Prompt Demo ===
Normalized box input: [0.36333, 0.42222, 0.07333, 0.32]
Saved: output_sam3_radio/single_box_input.png
Saved: output_sam3_radio/single_box_result.png

=== Multi-Box Prompt Demo ===
Saved: output_sam3_radio/multi_box_input.png
Saved: output_sam3_radio/multi_box_result.png

✓ Demo complete! Results saved to: output_sam3_radio
```

## References

- [SAM3 Repository](https://github.com/facebookresearch/sam3)
- [RADIO Repository](https://github.com/NVlabs/RADIO)
