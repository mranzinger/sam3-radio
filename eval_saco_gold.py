#!/usr/bin/env python3
"""
Evaluate SA-CO/Gold benchmark with SAM3 or RADIO encoder.

This script generates predictions for all 7 SA-CO/Gold subsets and evaluates them
using the cgF1 metric. It can use either the original SAM3 vision encoder or
the RADIO adaptor.

Usage examples:
    # With RADIO encoder:
    python eval_saco_gold.py --radio-model-version radio_v2.5-h --data-root /path/to/saco/gold

    # With original SAM3 encoder:
    python eval_saco_gold.py --skip-radio --data-root /path/to/saco/gold

    # Single subset only:
    python eval_saco_gold.py --radio-model-version radio_v2.5-h --data-root /path/to/saco/gold --subset metaclip_nps
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.eval.cgf1_eval import CGF1Evaluator
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.sam3_radio_utils import (
    create_sam3_radio_processor,
    load_radio_model,
    replace_sam3_encoder,
)
from sam3.train.data.coco_json_loaders import SAM3_EVAL_API_FROM_JSON_NP


# SA-CO/Gold subset definitions
SACO_GOLD_SUBSETS = {
    "metaclip_nps": {
        "gt_files": [
            "gold_metaclip_merged_a_release_test.json",
            "gold_metaclip_merged_b_release_test.json",
            "gold_metaclip_merged_c_release_test.json",
        ],
        "image_source": "metaclip",
    },
    "sa1b_nps": {
        "gt_files": [
            "gold_sa1b_merged_a_release_test.json",
            "gold_sa1b_merged_b_release_test.json",
            "gold_sa1b_merged_c_release_test.json",
        ],
        "image_source": "sa1b",
    },
    "crowded": {
        "gt_files": [
            "gold_crowded_merged_a_release_test.json",
            "gold_crowded_merged_b_release_test.json",
            "gold_crowded_merged_c_release_test.json",
        ],
        "image_source": "metaclip",
    },
    "fg_food": {
        "gt_files": [
            "gold_fg_food_merged_a_release_test.json",
            "gold_fg_food_merged_b_release_test.json",
            "gold_fg_food_merged_c_release_test.json",
        ],
        "image_source": "metaclip",
    },
    "fg_sports_equipment": {
        "gt_files": [
            "gold_fg_sports_equipment_merged_a_release_test.json",
            "gold_fg_sports_equipment_merged_b_release_test.json",
            "gold_fg_sports_equipment_merged_c_release_test.json",
        ],
        "image_source": "metaclip",
    },
    "attributes": {
        "gt_files": [
            "gold_attributes_merged_a_release_test.json",
            "gold_attributes_merged_b_release_test.json",
            "gold_attributes_merged_c_release_test.json",
        ],
        "image_source": "metaclip",
    },
    "wiki_common": {
        "gt_files": [
            "gold_wiki_common_merged_a_release_test.json",
            "gold_wiki_common_merged_b_release_test.json",
            "gold_wiki_common_merged_c_release_test.json",
        ],
        "image_source": "metaclip",
    },
}


def convert_rle_to_coco_format(rle_dict: Dict) -> Dict:
    """Convert RLE mask to COCO format."""
    return {
        "size": rle_dict["size"],
        "counts": rle_dict["counts"].decode("utf-8") if isinstance(rle_dict["counts"], bytes) else rle_dict["counts"]
    }


def generate_predictions_for_subset(
    processor: Sam3Processor,
    data_loader: SAM3_EVAL_API_FROM_JSON_NP,
    image_root: str,
    device: str = "cuda",
) -> List[Dict]:
    """
    Generate predictions for a single subset.

    Args:
        processor: SAM3 processor instance
        data_loader: Data loader for the subset
        image_root: Root directory containing images
        device: Device to use for inference

    Returns:
        List of predictions in COCO format
    """
    predictions = []
    datapoint_ids = data_loader.getDatapointIds()

    print(f"Processing {len(datapoint_ids)} datapoints...")

    for idx in tqdm(datapoint_ids):
        # Load image
        img_info = data_loader.loadImagesFromDatapoint(idx)[0]
        img_path = os.path.join(image_root, img_info["file_name"])

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        # Load query information
        queries, annotations = data_loader.loadQueriesAndAnnotationsFromDatapoint(idx)

        # Process each query (typically one per datapoint in SA-CO/Gold)
        for query in queries:
            text_prompt = query["query_text"]
            original_cat_id = query["original_cat_id"]
            coco_img_id = img_info["coco_img_id"]

            # Run inference
            inference_state = processor.set_image(image)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

            # Extract predictions
            masks = inference_state.get("masks")
            boxes = inference_state.get("boxes")
            scores = inference_state.get("scores")

            if masks is None or len(masks) == 0:
                continue

            # Convert to COCO format
            for i in range(len(scores)):
                # Convert mask to RLE
                mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                from pycocotools import mask as mask_utils
                rle = mask_utils.encode(np.asfortranarray(mask))

                # Get bbox from mask
                bbox = mask_utils.toBbox(rle).tolist()

                prediction = {
                    "image_id": coco_img_id,
                    "category_id": original_cat_id,
                    "segmentation": convert_rle_to_coco_format(rle),
                    "bbox": bbox,
                    "score": float(scores[i]),
                }
                predictions.append(prediction)

    return predictions


def evaluate_subset(
    subset_name: str,
    gt_paths: List[str],
    pred_path: str,
    iou_type: str = "segm"
) -> Dict:
    """
    Evaluate predictions for a single subset.

    Args:
        subset_name: Name of the subset
        gt_paths: Paths to ground truth annotation files
        pred_path: Path to prediction file
        iou_type: IoU type ("segm" or "bbox")

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {subset_name}...")
    evaluator = CGF1Evaluator(gt_path=gt_paths, verbose=True, iou_type=iou_type)
    summary = evaluator.evaluate(pred_path)

    metrics = {
        "cgf1": summary[f"cgF1_eval_{iou_type}_cgF1"] * 100,
        "il_mcc": summary[f"cgF1_eval_{iou_type}_IL_MCC"],
        "pmf1": summary[f"cgF1_eval_{iou_type}_positive_micro_F1"] * 100,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SA-CO/Gold benchmark with SAM3 or RADIO encoder"
    )
    parser.add_argument(
        "--radio-model-version",
        type=str,
        default=None,
        help="RADIO model version (e.g., 'radio_v2.5-h'). If not provided with --skip-radio, uses SAM3 encoder"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing SA-CO/Gold data (annotations and images)"
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Directory containing ground truth annotations (default: data-root)"
    )
    parser.add_argument(
        "--metaclip-img-dir",
        type=str,
        default=None,
        help="Directory containing MetaCLIP images (default: data-root/metaclip_images)"
    )
    parser.add_argument(
        "--sa1b-img-dir",
        type=str,
        default=None,
        help="Directory containing SA-1B images (default: data-root/sa1b_images)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./saco_gold_predictions",
        help="Directory to save predictions and results"
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
        default=0.3,
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
        help="Skip RADIO replacement and use original SAM3 encoder"
    )
    parser.add_argument(
        "--vitdet",
        type=int,
        default=None,
        help="Run RADIO in ViTDet mode with specified window size"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=list(SACO_GOLD_SUBSETS.keys()),
        help="Evaluate only a specific subset (default: all subsets)"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference and only run evaluation (assumes predictions already exist)"
    )

    args = parser.parse_args()

    # Setup paths
    data_root = Path(args.data_root)
    gt_dir = Path(args.gt_dir) if args.gt_dir else data_root / "gt-annotations"
    metaclip_img_dir = Path(args.metaclip_img_dir) if args.metaclip_img_dir else data_root / "metaclip_images"
    sa1b_img_dir = Path(args.sa1b_img_dir) if args.sa1b_img_dir else data_root / "sa1b_images"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate RADIO arguments
    if not args.skip_radio and args.radio_model_version is None:
        raise ValueError("Must provide --radio-model-version or use --skip-radio")

    # Determine which subsets to process
    subsets_to_process = {args.subset: SACO_GOLD_SUBSETS[args.subset]} if args.subset else SACO_GOLD_SUBSETS

    # Enable optimizations for Ampere GPUs
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Build model and processor (only if not skipping inference)
    processor = None
    if not args.skip_inference:
        print("\n" + "="*70)
        print("BUILDING SAM3 MODEL")
        print("="*70)

        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

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
            print("\nReplacing SAM3 encoder with RADIO...")
            radio_model = load_radio_model(args.radio_model_version, device=args.device, vitdet=args.vitdet)
            sam3_model = replace_sam3_encoder(sam3_model, radio_model, device=args.device)
            print("RADIO encoder loaded successfully!")

            # Use RADIO-specific processor
            processor = create_sam3_radio_processor(
                sam3_model,
                confidence_threshold=args.confidence_threshold,
                resolution=None
            )
            encoder_name = f"RADIO_{args.radio_model_version}"
        else:
            print("\nUsing original SAM3 encoder")
            processor = Sam3Processor(
                sam3_model,
                resolution=1008,
                confidence_threshold=args.confidence_threshold
            )
            encoder_name = "SAM3"

    # Process each subset
    all_results = {}

    for subset_name, subset_info in subsets_to_process.items():
        print("\n" + "="*70)
        print(f"PROCESSING SUBSET: {subset_name}")
        print("="*70)

        # Determine image directory
        image_root = metaclip_img_dir if subset_info["image_source"] == "metaclip" else sa1b_img_dir

        # Setup output paths
        subset_output_dir = output_dir / f"gold_{subset_name}"
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        pred_file = subset_output_dir / "coco_predictions_segm.json"

        # Generate predictions if not skipping
        if not args.skip_inference:
            # Use first GT file to load data structure
            gt_file = gt_dir / subset_info["gt_files"][0]

            if not gt_file.exists():
                print(f"Warning: GT file not found: {gt_file}")
                print(f"Skipping subset {subset_name}")
                continue

            print(f"Loading data from: {gt_file}")
            data_loader = SAM3_EVAL_API_FROM_JSON_NP(annotation_file=str(gt_file))

            # Generate predictions
            predictions = generate_predictions_for_subset(
                processor=processor,
                data_loader=data_loader,
                image_root=str(image_root),
                device=args.device,
            )

            print(f"Generated {len(predictions)} predictions")

            # Save predictions
            print(f"Saving predictions to: {pred_file}")
            with open(pred_file, "w") as f:
                json.dump(predictions, f)
        else:
            if not pred_file.exists():
                print(f"Warning: Prediction file not found: {pred_file}")
                print(f"Skipping subset {subset_name}")
                continue
            print(f"Using existing predictions: {pred_file}")

        # Evaluate
        gt_paths = [str(gt_dir / gt_file) for gt_file in subset_info["gt_files"]]

        # Check if GT files exist
        missing_gt_files = [p for p in gt_paths if not os.path.exists(p)]
        if missing_gt_files:
            print(f"Warning: Missing GT files: {missing_gt_files}")
            print(f"Skipping evaluation for subset {subset_name}")
            continue

        try:
            metrics = evaluate_subset(
                subset_name=subset_name,
                gt_paths=gt_paths,
                pred_path=str(pred_file),
                iou_type="segm"
            )
            all_results[subset_name] = metrics

            print(f"\nResults for {subset_name}:")
            print(f"  cgF1:   {metrics['cgf1']:.2f}")
            print(f"  IL_MCC: {metrics['il_mcc']:.2f}")
            print(f"  PM_F1:  {metrics['pmf1']:.2f}")
        except Exception as e:
            print(f"Error evaluating subset {subset_name}: {e}")
            continue

    # Compute and display averages
    if all_results:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)

        avg_metrics = {
            "cgf1": np.mean([r["cgf1"] for r in all_results.values()]),
            "il_mcc": np.mean([r["il_mcc"] for r in all_results.values()]),
            "pmf1": np.mean([r["pmf1"] for r in all_results.values()]),
        }

        # Print table
        print(f"\n{'Subset':<25} {'cgF1':>8} {'IL_MCC':>8} {'PM_F1':>8}")
        print("-" * 55)
        for subset_name, metrics in all_results.items():
            print(f"{subset_name:<25} {metrics['cgf1']:>8.2f} {metrics['il_mcc']:>8.2f} {metrics['pmf1']:>8.2f}")
        print("-" * 55)
        print(f"{'Average':<25} {avg_metrics['cgf1']:>8.2f} {avg_metrics['il_mcc']:>8.2f} {avg_metrics['pmf1']:>8.2f}")

        # Save results
        results_file = output_dir / "results_summary.json"
        with open(results_file, "w") as f:
            json.dump({"subsets": all_results, "average": avg_metrics}, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()
