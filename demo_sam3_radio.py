#!/usr/bin/env python3
"""
Demo script that replaces SAM3's vision encoder with RADIO.
This script performs image segmentation with text and box prompts,
saving the results to an output folder.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from sam3.sam3_radio_utils import (
    load_radio_model,
    replace_sam3_encoder,
    create_sam3_radio_processor
)


def save_figure(fig, output_path: str):
    """Save matplotlib figure to file."""
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def demo_text_prompt(processor, inference_state, image_path: str, output_dir: Path, text_prompt: str = 'shoe'):
    """
    Demo text prompt segmentation.

    Args:
        processor: SAM3 processor
        inference_state: Current inference state
        image_path: Path to input image
        output_dir: Directory to save outputs
    """
    print("\n=== Text Prompt Demo ===")

    # Reset prompts and set text prompt
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

    # Load image and generate results
    img = Image.open(image_path)

    # plot_results creates its own figure
    plot_results(img, inference_state)

    # Get the current figure and save it
    fig = plt.gcf()
    output_path = output_dir / "text_prompt_shoe.png"
    save_figure(fig, str(output_path))

    return inference_state


def demo_single_box_prompt(processor, inference_state, image_path: str,
                           width: int, height: int, output_dir: Path):
    """
    Demo single bounding box prompt segmentation.

    Args:
        processor: SAM3 processor
        inference_state: Current inference state
        image_path: Path to input image
        width: Image width
        height: Image height
        output_dir: Directory to save outputs
    """
    print("\n=== Single Box Prompt Demo ===")

    # Define box in (x, y, w, h) format
    box_input_xywh = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

    # Normalize box coordinates
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
    print(f"Normalized box input: {norm_box_cxcywh}")

    # Reset prompts and add box prompt
    processor.reset_all_prompts(inference_state)
    inference_state = processor.add_geometric_prompt(
        state=inference_state, box=norm_box_cxcywh, label=True
    )

    # Load image and draw box
    img = Image.open(image_path)
    image_with_box = draw_box_on_image(img, box_input_xywh.flatten().tolist())

    # Save image with box
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_with_box)
    ax.axis("off")
    output_path = output_dir / "single_box_input.png"
    save_figure(fig, str(output_path))

    # Generate and save segmentation results
    plot_results(img, inference_state)
    fig = plt.gcf()
    output_path = output_dir / "single_box_result.png"
    save_figure(fig, str(output_path))

    return inference_state


def demo_multi_box_prompt(processor, inference_state, image_path: str,
                          width: int, height: int, output_dir: Path):
    """
    Demo multi-box prompt segmentation with positive and negative boxes.

    Args:
        processor: SAM3 processor
        inference_state: Current inference state
        image_path: Path to input image
        width: Image width
        height: Image height
        output_dir: Directory to save outputs
    """
    print("\n=== Multi-Box Prompt Demo ===")

    # Define boxes in (x, y, w, h) format
    box_input_xywh = [[480.0, 290.0, 110.0, 360.0], [370.0, 280.0, 115.0, 375.0]]
    box_input_cxcywh = box_xywh_to_cxcywh(torch.tensor(box_input_xywh).view(-1, 4))
    norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, width, height).tolist()

    # Labels: True for positive, False for negative
    box_labels = [True, False]

    # Reset prompts and add box prompts
    processor.reset_all_prompts(inference_state)
    for box, label in zip(norm_boxes_cxcywh, box_labels):
        inference_state = processor.add_geometric_prompt(
            state=inference_state, box=box, label=label
        )

    # Load image and draw boxes (green for positive, red for negative)
    img = Image.open(image_path)
    image_with_box = img
    for i in range(len(box_input_xywh)):
        color = (0, 255, 0) if box_labels[i] else (255, 0, 0)
        image_with_box = draw_box_on_image(image_with_box, box_input_xywh[i], color)

    # Save image with boxes
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_with_box)
    ax.axis("off")
    output_path = output_dir / "multi_box_input.png"
    save_figure(fig, str(output_path))

    # Generate and save segmentation results
    plot_results(img, inference_state)
    fig = plt.gcf()
    output_path = output_dir / "multi_box_result.png"
    save_figure(fig, str(output_path))

    return inference_state


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 with RADIO encoder demo - Image segmentation with text and box prompts"
    )
    parser.add_argument(
        "--radio-model-version",
        type=str,
        required=True,
        help="RADIO model version. Either a known name, or path to checkpoint file"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (default: uses SAM3 test image)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_sam3_radio",
        help="Directory to save output images"
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint (default: downloads from HuggingFace)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--skip-radio",
        action="store_true",
        help="Skip RADIO replacement and use original SAM3 encoder (for testing)"
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="shoe",
        help="Text prompt for segmentation (default: 'shoe')"
    )
    parser.add_argument(
        "--vitdet",
        type=int,
        default=None,
        help='Run RADIO in ViTDet mode with specified window size',
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Enable optimizations for Ampere GPUs
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Get paths
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    image_path = args.image if args.image else f"{sam3_root}/assets/images/test_image.jpg"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Using image: {image_path}")

    # Build SAM3 model
    print("\nBuilding SAM3 model...")
    sam3_model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=(args.sam3_checkpoint is None),
        eval_mode=True,
        device=args.device
    )
    print("SAM3 model built successfully!")

    # Replace encoder with RADIO if requested
    if not args.skip_radio:
        radio_model = load_radio_model(args.radio_model_version, device=args.device, vitdet=args.vitdet)
        sam3_model = replace_sam3_encoder(sam3_model, radio_model, device=args.device)
    else:
        print("\nSkipping RADIO replacement - using original SAM3 encoder")

    # Initialize processor and set image
    print("\nInitializing processor...")
    if args.skip_radio:
        # Use standard SAM3 processor with default resolution
        processor = Sam3Processor(
            sam3_model,
            resolution=1008,
            confidence_threshold=args.confidence_threshold
        )
    else:
        # Use RADIO-specific processor with adjusted resolution for 16x16 patches
        processor = create_sam3_radio_processor(
            sam3_model,
            confidence_threshold=args.confidence_threshold,
            resolution=None  # Uses default RADIO resolution
        )

    image = Image.open(image_path)
    width, height = image.size
    print(f"Image size: {width}x{height}")

    inference_state = processor.set_image(image)

    # Run demos
    print("\nRunning segmentation demos...")

    # Text prompt demo
    inference_state = demo_text_prompt(processor, inference_state, image_path, output_dir, text_prompt=args.text_prompt)

    # Single box prompt demo
    inference_state = demo_single_box_prompt(
        processor, inference_state, image_path, width, height, output_dir
    )

    # Multi-box prompt demo
    inference_state = demo_multi_box_prompt(
        processor, inference_state, image_path, width, height, output_dir
    )

    print(f"\n✓ Demo complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
