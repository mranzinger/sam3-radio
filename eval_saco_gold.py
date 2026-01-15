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
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
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


def build_model_and_processor(
    sam3_checkpoint: Optional[str],
    skip_radio: bool,
    radio_model_version: Optional[str],
    vitdet: Optional[int],
    confidence_threshold: float,
    device: str
):
    """
    Build SAM3 model and processor with optional RADIO encoder.

    Args:
        sam3_checkpoint: Path to SAM3 checkpoint (or None to load from HF)
        skip_radio: Whether to skip RADIO and use original SAM3 encoder
        radio_model_version: RADIO model version to load
        vitdet: ViTDet window size (or None)
        confidence_threshold: Confidence threshold for predictions
        device: Device to use

    Returns:
        Tuple of (processor, encoder_name)
    """
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    sam3_model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=sam3_checkpoint,
        load_from_HF=(sam3_checkpoint is None),
        eval_mode=True,
        device=device
    )

    # Ensure model is on the correct device
    sam3_model = sam3_model.to(device)

    if not skip_radio:
        radio_model = load_radio_model(
            radio_model_version,
            device=device,
            vitdet=vitdet
        )
        sam3_model = replace_sam3_encoder(sam3_model, radio_model, device=device)
        # Ensure model is still on correct device after encoder replacement
        sam3_model = sam3_model.to(device)
        processor = create_sam3_radio_processor(
            sam3_model,
            confidence_threshold=confidence_threshold,
            resolution=None
        )
        encoder_name = f"RADIO_{radio_model_version}"
    else:
        processor = Sam3Processor(
            sam3_model,
            resolution=1008,
            confidence_threshold=confidence_threshold
        )
        encoder_name = "SAM3"

    return processor, encoder_name


def gpu_worker(
    gpu_id: int,
    work_queue,
    result_queue,
    args_dict: Dict,
    device_str: str
):
    """
    GPU worker that processes images from a shared queue.

    Args:
        gpu_id: GPU device ID to use
        work_queue: Shared queue containing (subset_name, image_path, query, img_info) tuples or None sentinel
        result_queue: Shared queue for pushing (subset_name, prediction) tuples
        args_dict: Dictionary of command-line arguments
        device_str: Device string (e.g., "cuda:0")
    """
    import torch
    import os
    from PIL import Image

    # Set CUDA device for this process
    torch.cuda.set_device(gpu_id)
    device = device_str

    print(f"[GPU {gpu_id}] Worker started", flush=True)

    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Build model and processor once
    processor, _ = build_model_and_processor(
        sam3_checkpoint=args_dict["sam3_checkpoint"],
        skip_radio=args_dict["skip_radio"],
        radio_model_version=args_dict["radio_model_version"],
        vitdet=args_dict["vitdet"],
        confidence_threshold=args_dict["confidence_threshold"],
        device=device
    )

    print(f"[GPU {gpu_id}] Model loaded, ready to process", flush=True)

    # Process images from queue until sentinel
    processed_count = 0
    while True:
        work_item = work_queue.get()  # Blocking get

        if work_item is None:  # Sentinel value to shut down
            print(f"[GPU {gpu_id}] Received shutdown signal", flush=True)
            break

        subset_name, img_path, query, img_info = work_item
        processed_count += 1

        try:
            # Load and process image
            if not os.path.exists(img_path):
                result_queue.put((subset_name, {"error": f"Image not found: {img_path}"}))
                continue

            image = Image.open(img_path).convert("RGB")

            # Run inference
            inference_state = processor.set_image(image)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=query["query_text"])

            # Extract predictions
            masks = inference_state.get("masks")
            scores = inference_state.get("scores")

            # Collect all predictions for this work item
            predictions_for_item = []
            if masks is not None and len(masks) > 0:
                # Convert to COCO format
                from pycocotools import mask as mask_utils
                for i in range(len(scores)):
                    mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    bbox = mask_utils.toBbox(rle).tolist()

                    prediction = {
                        "image_id": img_info["coco_img_id"],
                        "category_id": 1,  # SA-CO/Gold uses category_id=1 for all annotations
                        "segmentation": convert_rle_to_coco_format(rle),
                        "bbox": bbox,
                        "score": float(scores[i]),
                    }
                    predictions_for_item.append(prediction)

            # Emit all predictions for this work item at once (could be empty list)
            result_queue.put((subset_name, predictions_for_item))
        except Exception as e:
            result_queue.put((subset_name, {"error": f"Error processing {img_path}: {str(e)}"}))
        finally:
            work_queue.task_done()

    print(f"[GPU {gpu_id}] Worker shutting down. Processed {processed_count} images total.", flush=True)


def process_with_multi_gpu(
    args,
    subsets_to_process: Dict,
    data_root: Path,
    gt_dir: Path,
    metaclip_img_dir: Path,
    sa1b_img_dir: Path,
    output_dir: Path,
    num_gpus: int
):
    """
    Orchestrate multi-GPU processing using persistent GPU workers with image-level work queue.

    Args:
        args: Command-line arguments
        subsets_to_process: Dictionary of subsets to process
        data_root: Root data directory
        gt_dir: Ground truth annotations directory
        metaclip_img_dir: Metaclip images directory
        sa1b_img_dir: SA-1B images directory
        output_dir: Output directory
        num_gpus: Number of GPUs to use
    """
    # Convert args to dict for pickling
    args_dict = {
        "sam3_checkpoint": args.sam3_checkpoint,
        "radio_model_version": args.radio_model_version,
        "skip_radio": args.skip_radio,
        "vitdet": args.vitdet,
        "confidence_threshold": args.confidence_threshold,
    }

    # Determine encoder name for output files
    if not args.skip_radio:
        encoder_name = f"RADIO_{args.radio_model_version}"
    else:
        encoder_name = "SAM3"

    print(f"\nStarting persistent GPU swarm with {num_gpus} GPUs")

    # Create shared queues
    ctx = mp.get_context('spawn')
    work_queue = ctx.Manager().Queue()
    result_queue = ctx.Manager().Queue()

    # Spawn GPU workers (they will persist across all subsets)
    print(f"Spawning {num_gpus} GPU workers...")
    processes = []
    for gpu_id in range(num_gpus):
        device_str = f"cuda:{gpu_id}"
        p = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, work_queue, result_queue, args_dict, device_str)
        )
        p.start()
        processes.append(p)

    # Process each subset
    for subset_name, subset_info in subsets_to_process.items():
        print("\n" + "="*70)
        print(f"PROCESSING SUBSET: {subset_name}")
        print("="*70)

        # Setup output paths
        subset_output_dir = output_dir / f"gold_{subset_name}"
        subset_output_dir.mkdir(parents=True, exist_ok=True)
        pred_file = subset_output_dir / "coco_predictions_segm.json"

        # Load subset data
        gt_files = subset_info["gt_files"]
        img_dir = metaclip_img_dir if subset_info["image_source"] == "metaclip" else sa1b_img_dir

        # Load datapoints from the FIRST GT file only (merged_a)
        # The three GT files contain the same images with annotations from 3 different annotators
        # We run inference once per image, then evaluate against all 3 annotation sets
        gt_file = gt_dir / gt_files[0]
        if not gt_file.exists():
            print(f"Warning: GT file not found: {gt_file}")
            continue

        data_loader = SAM3_EVAL_API_FROM_JSON_NP(annotation_file=str(gt_file))
        datapoint_ids = data_loader.getDatapointIds()

        print(f"Loading {len(datapoint_ids)} images into work queue...")

        # Populate work queue with all images for this subset
        work_items_count = 0
        for idx in datapoint_ids:
            img_info = data_loader.loadImagesFromDatapoint(idx)[0]
            img_path = os.path.join(str(img_dir), img_info["file_name"])
            queries, _ = data_loader.loadQueriesAndAnnotationsFromDatapoint(idx)

            for query in queries:
                work_queue.put((subset_name, img_path, query, img_info))
                work_items_count += 1

        print(f"Processing {work_items_count} work items with {num_gpus} GPUs...")

        # Collect results with progress bar
        predictions = []
        errors = []
        for _ in tqdm(range(work_items_count), desc=f"Processing {subset_name}", position=0, leave=True):
            result_subset_name, result_data = result_queue.get()
            if result_subset_name == subset_name:
                if isinstance(result_data, dict) and "error" in result_data:
                    errors.append(result_data)
                elif isinstance(result_data, list):
                    # result_data is a list of predictions from one work item
                    predictions.extend(result_data)
                else:
                    # Unexpected format, treat as error
                    errors.append({"error": f"Unexpected result format: {type(result_data)}"})

        if errors:
            print(f"\nEncountered {len(errors)} errors during processing")
        print(f"Collected {len(predictions)} predictions for {subset_name}")

        # Save predictions
        with open(pred_file, "w") as f:
            json.dump(predictions, f)

        print(f"Saved predictions to {pred_file}")

    # Send shutdown signals to all workers
    print("\nShutting down GPU workers...")
    for _ in range(num_gpus):
        work_queue.put(None)

    # Wait for all workers to complete
    for p in processes:
        p.join()

    print("\nAll GPU workers shut down successfully!")


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

    # Run inference with GPU swarm if not skipping
    if not args.skip_inference:
        num_gpus = torch.cuda.device_count()
        print(f"\nDetected {num_gpus} GPU(s)")

        if num_gpus == 0:
            raise RuntimeError("No GPUs available for inference")

        # Use GPU swarm for inference
        process_with_multi_gpu(
            args=args,
            subsets_to_process=subsets_to_process,
            data_root=data_root,
            gt_dir=gt_dir,
            metaclip_img_dir=metaclip_img_dir,
            sa1b_img_dir=sa1b_img_dir,
            output_dir=output_dir,
            num_gpus=num_gpus
        )

    # Evaluate results
    all_results = {}

    for subset_name, subset_info in subsets_to_process.items():
        print("\n" + "="*70)
        print(f"EVALUATING SUBSET: {subset_name}")
        print("="*70)

        # Setup output paths
        subset_output_dir = output_dir / f"gold_{subset_name}"
        pred_file = subset_output_dir / "coco_predictions_segm.json"

        # Check if predictions exist
        if not pred_file.exists():
            print(f"Warning: Prediction file not found: {pred_file}")
            print(f"Skipping subset {subset_name}")
            continue

        print(f"Using predictions: {pred_file}")

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
