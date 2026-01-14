#!/usr/bin/env python3
"""
Benchmark script comparing SAM3's vision encoder with RADIO encoder.
Tests inference speed across different configurations including ViTDet window sizes.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sam3
from sam3 import build_sam3_image_model
from sam3.sam3_radio_utils import load_radio_model


class BenchmarkTimer:
    """Context manager for timing code execution."""

    def __init__(self, name: str, warmup: bool = False):
        self.name = name
        self.warmup = warmup
        self.elapsed = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
        if not self.warmup:
            print(f"  {self.name}: {self.elapsed*1000:.2f}ms")


def benchmark_encoder(
    encoder: nn.Module,
    images: torch.Tensor,
    name: str,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark an encoder's inference speed.

    Args:
        encoder: The vision encoder to benchmark
        images: Input images tensor
        name: Name for this benchmark
        num_iterations: Number of iterations for timing
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    encoder.eval()

    # Warmup
    print(f"Warmup ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = encoder(images)

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []

    for i in range(num_iterations):
        with BenchmarkTimer(f"Iteration {i+1}", warmup=True) as timer:
            with torch.no_grad():
                _ = encoder(images)
        times.append(timer.elapsed)

    times = np.array(times) * 1000  # Convert to milliseconds

    results = {
        'name': name,
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
    }

    # Print summary
    print(f"\nResults for {name}:")
    print(f"  Mean:   {results['mean_ms']:.2f} ms ± {results['std_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  Min:    {results['min_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")
    print(f"  P95:    {results['p95_ms']:.2f} ms")
    print(f"  P99:    {results['p99_ms']:.2f} ms")

    return results


def prepare_image(image_path: str, resolution: int, device: str) -> torch.Tensor:
    """Load and prepare image for benchmarking."""
    img = Image.open(image_path).convert('RGB')

    # Convert to tensor and normalize to [-1, 1] (SAM3 convention)
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor * 2.0 - 1.0

    # Resize to target resolution
    img_tensor = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0),
        size=(resolution, resolution),
        mode='bilinear',
        align_corners=False
    )

    return img_tensor.to(device)


def create_synthetic_image(resolution: int, batch_size: int, device: str) -> torch.Tensor:
    """Create a synthetic image tensor for benchmarking.

    Args:
        resolution: Image resolution (height and width)
        batch_size: Batch size
        device: Device to create tensor on

    Returns:
        Tensor of shape [batch_size, 3, resolution, resolution] normalized to [-1, 1]
    """
    # Create random image normalized to [-1, 1] (SAM3 convention)
    images = torch.randn(batch_size, 3, resolution, resolution, device=device)
    return images


def plot_benchmark_results(results: List[Dict], output_path: str):
    """Plot benchmark results using seaborn.

    Args:
        results: List of benchmark result dictionaries
        output_path: Path to save the plot
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)

    # Organize data by model and window size
    # Get unique models in order
    models_seen = []
    for result in results:
        model_name = result.get('model_name', 'Unknown')
        if model_name not in models_seen:
            models_seen.append(model_name)

    # Create color palette with one color per model
    colors = sns.color_palette("husl", n_colors=len(models_seen))
    model_colors = {model: colors[i] for i, model in enumerate(models_seen)}

    # Organize data by model
    model_data = {model: {'window_sizes': [], 'means': [], 'stds': []} for model in models_seen}

    for result in results:
        model_name = result.get('model_name', 'Unknown')
        window_size = result.get('window_size', 72)

        model_data[model_name]['window_sizes'].append(window_size)
        model_data[model_name]['means'].append(result['mean_ms'])
        model_data[model_name]['stds'].append(result['std_ms'])

    # Sort each model's data by window size
    for model in models_seen:
        data = model_data[model]
        sorted_indices = np.argsort(data['window_sizes'])
        data['window_sizes'] = np.array(data['window_sizes'])[sorted_indices].tolist()
        data['means'] = np.array(data['means'])[sorted_indices]
        data['stds'] = np.array(data['stds'])[sorted_indices]

    # Create line plot
    fig, ax = plt.subplots(figsize=(14, 8))

    for model in models_seen:
        data = model_data[model]
        window_sizes = np.array(data['window_sizes'])
        means = data['means']
        stds = data['stds']
        color = model_colors[model]

        # Plot line with markers
        ax.plot(window_sizes, means, marker='o', linewidth=2, markersize=8,
                label=model, color=color, alpha=0.8)

        # Add std deviation band
        ax.fill_between(window_sizes, means - stds, means + stds,
                       color=color, alpha=0.2)

        # Add text labels at each data point
        for ws, mean in zip(window_sizes, means):
            ax.text(ws, mean, f'{mean:.1f}', fontsize=9, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

    # Customize plot
    ax.set_xlabel('VitDet Window Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Vision Encoder Benchmark: SAM3 vs RADIO', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set x-axis to show specific window size values
    all_window_sizes = sorted(set([ws for model in models_seen
                                   for ws in model_data[model]['window_sizes']]))
    ax.set_xticks(all_window_sizes)
    ax.set_xticklabels([str(ws) for ws in all_window_sizes])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()


def plot_from_json(json_path: str, output_path: Optional[str] = None):
    """Load results from JSON and create plot.

    Args:
        json_path: Path to JSON file with benchmark results
        output_path: Path to save plot (defaults to same directory as JSON)
    """
    print(f"Loading results from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data['results']

    if output_path is None:
        json_path_obj = Path(json_path)
        output_path = str(json_path_obj.parent / f"{json_path_obj.stem}_plot.pdf")

    plot_benchmark_results(results, output_path)
    print(f"✓ Plot created from existing results")


def benchmark_vitdet_windows(
    radio_model_version: str,
    radio_model_name: str,
    images: torch.Tensor,
    window_sizes: List[int],
    device: str,
    num_iterations: int,
    warmup_iterations: int
) -> List[Dict[str, float]]:
    """Benchmark RADIO with different ViTDet window sizes."""
    results = []

    for window_size in window_sizes:
        print(f"\n{'='*60}")
        print(f"Loading RADIO model ({radio_model_name}) with vitdet_window_size={window_size}")
        print(f"{'='*60}")

        # Load RADIO with specific vitdet window size
        radio_model: torch.nn.Module = torch.hub.load(
            'NVlabs/RADIO',
            'radio_model',
            radio_model_version,
            adaptor_names='sam3',
            vitdet_window_size=window_size
        )
        radio_model = radio_model.to(device)
        radio_model.eval()
        print("RADIO model loaded successfully!")

        # Benchmark
        result = benchmark_encoder(
            radio_model,
            images,
            f"{radio_model_name} (ViTDet w={window_size})",
            num_iterations,
            warmup_iterations
        )
        result['window_size'] = window_size
        result['model_name'] = radio_model_name
        results.append(result)

        # Clean up
        del radio_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SAM3 vs RADIO vision encoder performance"
    )
    parser.add_argument(
        "--radio-model-versions",
        type=str,
        nargs='+',
        help="RADIO model versions to benchmark. Either known names or paths to checkpoint files"
    )
    parser.add_argument(
        "--radio-model-names",
        type=str,
        nargs='+',
        help="Display names for RADIO models (must match length of --radio-model-versions)"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plot from existing results without running benchmarks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./benchmark_results.json",
        help="Input JSON file for plot-only mode, or output file for benchmark results"
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Output file for plot (default: same directory as JSON with _plot.png suffix)"
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=str,
        default=None,
        help="Path to SAM3 checkpoint (default: downloads from HuggingFace)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking (default: 1)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmarking (default: 100)"
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--vitdet-window-sizes",
        type=int,
        nargs='+',
        default=[8, 12, 18, 24],
        help="ViTDet window sizes to benchmark (default: 8 12 18 24)"
    )
    parser.add_argument(
        "--skip-sam3",
        action="store_true",
        help="Skip benchmarking original SAM3 encoder"
    )
    parser.add_argument(
        "--skip-radio-standard",
        action="store_true",
        help="Skip benchmarking RADIO in standard mode"
    )
    parser.add_argument(
        "--skip-vitdet",
        action="store_true",
        help="Skip benchmarking ViTDet configurations"
    )

    args = parser.parse_args()

    # Plot-only mode
    if args.plot_only:
        plot_from_json(args.output, args.plot_output)
        return

    # Validate RADIO model arguments
    if not args.radio_model_versions:
        parser.error("--radio-model-versions is required when not using --plot-only")

    if args.radio_model_names:
        if len(args.radio_model_names) != len(args.radio_model_versions):
            parser.error("--radio-model-names must have the same length as --radio-model-versions")
    else:
        # Generate default names
        args.radio_model_names = [f"RADIO-{i+1}" for i in range(len(args.radio_model_versions))]

    # Enable optimizations for Ampere GPUs
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Get paths
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    print(f"Batch size: {args.batch_size}")

    all_results = []

    # Create synthetic images for SAM3 (1008x1008)
    sam3_images = create_synthetic_image(1008, args.batch_size, args.device)
    print(f"SAM3 input tensor shape: {sam3_images.shape}")

    # Create synthetic images for RADIO (1152x1152)
    radio_images = create_synthetic_image(1152, args.batch_size, args.device)
    print(f"RADIO input tensor shape: {radio_images.shape}")

    # Benchmark original SAM3 encoder
    if not args.skip_sam3:
        print("\n" + "="*70)
        print("BUILDING SAM3 MODEL (ORIGINAL ENCODER)")
        print("="*70)

        sam3_model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=args.sam3_checkpoint,
            load_from_HF=(args.sam3_checkpoint is None),
            eval_mode=True,
            device=args.device
        )

        encoder = sam3_model.backbone.vision_backbone

        result = benchmark_encoder(
            encoder,
            sam3_images,
            "SAM3 (Original)",
            args.num_iterations,
            args.warmup_iterations
        )
        result['window_size'] = 24  # SAM3 uses window size 24
        result['model_name'] = 'SAM3'
        all_results.append(result)

        del sam3_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Benchmark RADIO models
    for radio_version, radio_name in zip(args.radio_model_versions, args.radio_model_names):
        # Benchmark RADIO in standard mode
        if not args.skip_radio_standard:
            print("\n" + "="*70)
            print(f"LOADING RADIO MODEL: {radio_name} (STANDARD MODE)")
            print("="*70)

            radio_model = load_radio_model(radio_version, device=args.device)

            result = benchmark_encoder(
                radio_model,
                radio_images,
                f"{radio_name} (Standard)",
                args.num_iterations,
                args.warmup_iterations
            )
            result['window_size'] = 72  # Non-ViTDet treated as window size 72
            result['model_name'] = radio_name
            all_results.append(result)

            del radio_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Benchmark RADIO with different ViTDet window sizes
        if not args.skip_vitdet and args.vitdet_window_sizes:
            vitdet_results = benchmark_vitdet_windows(
                radio_version,
                radio_name,
                radio_images,
                args.vitdet_window_sizes,
                args.device,
                args.num_iterations,
                args.warmup_iterations
            )
            all_results.extend(vitdet_results)

    # Print summary comparison
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<30} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12}")
    print("-" * 70)
    for result in all_results:
        name = result['name']
        print(f"{name:<30} {result['mean_ms']:<12.2f} {result['median_ms']:<12.2f} {result['p95_ms']:<12.2f}")

    # Calculate speedups relative to SAM3 original
    sam3_baseline = None
    for result in all_results:
        if result.get('encoder_type') == 'sam3_original':
            sam3_baseline = result['mean_ms']
            break

    if sam3_baseline:
        print(f"\n{'Configuration':<30} {'Speedup vs SAM3':<20}")
        print("-" * 70)
        for result in all_results:
            speedup = sam3_baseline / result['mean_ms']
            status = "✓ FASTER" if speedup > 1.0 else "✗ SLOWER"
            print(f"{result['name']:<30} {speedup:.3f}x {status}")

    # Save results to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark_data = {
        'config': {
            'batch_size': args.batch_size,
            'num_iterations': args.num_iterations,
            'warmup_iterations': args.warmup_iterations,
            'device': args.device,
            'sam3_resolution': 1008,
            'radio_resolution': 1152,
            'radio_models': list(zip(args.radio_model_versions, args.radio_model_names)),
        },
        'results': all_results
    }

    with open(output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Generate plot
    plot_output = args.plot_output
    if plot_output is None:
        output_path_obj = Path(args.output)
        plot_output = str(output_path_obj.parent / f"{output_path_obj.stem}_plot.png")

    plot_benchmark_results(all_results, plot_output)


if __name__ == "__main__":
    main()
