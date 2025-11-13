#!/usr/bin/env python3
"""
Portable inference script using pathlib and configurable inputs.

Example:
    python planktivore-inference.py \
        --model-dir /Volumes/patrick_ssd/Synchro_April_2025/mbari-ptvr-vits-b8-20250826 \
        --images-root /Volumes/patrick_ssd/Synchro_April_2025/high_mag_cam/20250414T224000\
        --pattern "*.png" \
        --batch-size 32 \
        --outdir ./inference_Synchro_April_2025
"""

from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
import argparse
import time
import torch
import pandas as pd
from typing import List


def pick_device(prefer: str | None = None) -> str:
    """
    Pick the best available device.
    Order: explicit 'cuda'/'mps'/'cpu' if valid -> CUDA (if available) -> MPS -> CPU.
    """
    if prefer:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        if prefer == "mps" and torch.backends.mps.is_available():
            return "mps"
        if prefer == "cpu":
            return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def discover_images(images_root: Path, pattern: str) -> List[Path]:
    """
    Recursively find images under images_root that match a glob pattern (e.g., '*.png').
    """
    return sorted(images_root.rglob(pattern))


def load_images(paths: List[Path]) -> List[Image.Image]:
    """
    Load images as RGB PIL images. Skips files that fail to open.
    """
    imgs = []
    for p in paths:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
        except Exception:
            # Skip unreadable images but continue processing others
            continue
    return imgs


def run_inference(
    model_dir: Path,
    image_paths: List[Path],
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """
    Run batched inference over image_paths and return a DataFrame of top-3 predictions.
    """
    print(f"Using device: {device}")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
    model.eval()

    id2label = model.config.id2label
    if isinstance(id2label, dict):
        # keys may be strings in some configs
        id2label = {int(k): v for k, v in id2label.items()}

    all_results = []
    start_time = time.time()

    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inference"):
        batch_paths = image_paths[i : i + batch_size]

        # Load PIL images (keep order aligned with paths)
        pil_imgs = []
        kept_paths = []
        for p in batch_paths:
            try:
                with Image.open(p) as im:
                    pil_imgs.append(im.convert("RGB"))
                    kept_paths.append(p)
            except Exception:
                # skip unreadable files in this batch
                continue

        if not pil_imgs:
            continue

        inputs = processor(images=pil_imgs, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Collect top-3 predictions per image
        for path, prob_vec in zip(kept_paths, probs):
            top_probs, top_ids = prob_vec.topk(min(3, prob_vec.shape[-1]))
            result = {
                "file": str(path),
                "label1": id2label[top_ids[0].item()],
                "score1": float(top_probs[0].item()),
            }
            if len(top_ids) > 1:
                result.update(
                    {
                        "label2": id2label[top_ids[1].item()],
                        "score2": float(top_probs[1].item()),
                    }
                )
            if len(top_ids) > 2:
                result.update(
                    {
                        "label3": id2label[top_ids[2].item()],
                        "score3": float(top_probs[2].item()),
                    }
                )
            all_results.append(result)

    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.2f} seconds. Batch size: {batch_size}")

    return pd.DataFrame(all_results)


def make_default_outnames(model_dir: Path, outdir: Path) -> tuple[Path, Path]:
    """
    Parquet output paths based on model directory name.
    """
    tag = model_dir.name
    parquet_path = outdir / f"inference_results_{tag}.parquet"
    return parquet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image classification inference.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to directory containing the model and preprocessor (e.g., a local HF folder).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Root directory containing images to process (searched recursively).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern (relative) to match images within images-root (default: *.png).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Force a specific device (cuda/mps/cpu). Defaults to best available.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./"),
        help="Directory to write outputs (CSV and Parquet). Default: current directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = pick_device(args.device)

    # Validate paths
    model_dir = args.model_dir.expanduser().resolve()
    images_root = args.images_root.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    # Discover images
    image_paths = discover_images(images_root, args.pattern)
    print(f"Number of images to process: {len(image_paths)}")

    if not image_paths:
        print("No images found. Exiting.")
        return

    # Run inference
    df_results = run_inference(
        model_dir=model_dir,
        image_paths=image_paths,
        batch_size=args.batch_size,
        device=device,
    )

    # Save outputs
    parquet_path = make_default_outnames(model_dir, outdir)

    try:
        df_results.to_parquet(parquet_path, index=False)
    
    except Exception as e:
        print(f"Warning: failed to write Parquet ({parquet_path}): {e}")

    if parquet_path.exists():
        print(f"Wrote: {parquet_path}")


if __name__ == "__main__":
    main()
