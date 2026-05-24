import argparse
import json
import os
from pathlib import Path

import imageio
import lpips
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_video_rgb_tensor(video_path: Path) -> torch.Tensor:
    """Load an mp4 as float tensor in [0, 1] with shape [T, C, H, W]."""
    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        # frame: [H, W, C], uint8
        t = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        frames.append(t)
    reader.close()

    if not frames:
        raise ValueError(f"No frames found in {video_path}")

    return torch.stack(frames, dim=0)


def calculate_ssim(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    """Compute SSIM for a single frame pair, inputs in [0, 1], shape [1, C, H, W]."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = F.avg_pool2d(frame_a, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(frame_b, kernel_size=11, stride=1, padding=5)

    sigma1_sq = F.avg_pool2d(frame_a * frame_a, kernel_size=11, stride=1, padding=5) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(frame_b * frame_b, kernel_size=11, stride=1, padding=5) - mu2 ** 2
    sigma12 = F.avg_pool2d(frame_a * frame_b, kernel_size=11, stride=1, padding=5) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def compute_video_metrics(pred: torch.Tensor, ref: torch.Tensor, lpips_model, device: torch.device, skip_frames: int = 0) -> dict:
    """Compute average PSNR/SSIM/LPIPS over aligned frames."""
    if pred.ndim != 4 or ref.ndim != 4:
        raise ValueError(f"Expected 4D tensors [T, C, H, W], got {pred.shape} and {ref.shape}")

    total_frames = min(pred.shape[0], ref.shape[0])
    if total_frames <= skip_frames:
        raise ValueError(f"Not enough frames after skip_frames={skip_frames}, total={total_frames}")

    pred = pred[:total_frames]
    ref = ref[:total_frames]
    pred = pred[skip_frames:]
    ref = ref[skip_frames:]

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for i in range(pred.shape[0]):
        frame_pred = pred[i].unsqueeze(0).to(device)
        frame_ref = ref[i].unsqueeze(0).to(device)

        mse = F.mse_loss(frame_pred, frame_ref, reduction="mean")
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
        ssim = calculate_ssim(frame_pred, frame_ref)

        # LPIPS expects normalized inputs in [-1, 1].
        lpips_pred = frame_pred * 2.0 - 1.0
        lpips_ref = frame_ref * 2.0 - 1.0
        lpips_val = lpips_model(lpips_pred, lpips_ref)

        psnr_values.append(psnr.item())
        ssim_values.append(ssim.item())
        lpips_values.append(lpips_val.item())

    return {
        "num_eval_frames": len(psnr_values),
        "PSNR": float(sum(psnr_values) / len(psnr_values)),
        "SSIM": float(sum(ssim_values) / len(ssim_values)),
        "LPIPS": float(sum(lpips_values) / len(lpips_values)),
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM/LPIPS between two video folders.")
    parser.add_argument("--pred_folder", type=str, required=True, help="Folder containing predicted mp4 files")
    parser.add_argument("--ref_folder", type=str, required=True, help="Folder containing reference mp4 files")
    parser.add_argument("--output_json", type=str, required=True, help="Output summary JSON path")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Optional per-video JSONL path")
    parser.add_argument("--skip_frames", type=int, default=0, help="Skip first N frames during metric computation")
    parser.add_argument("--device", type=str, default="cuda", help="Metric compute device: cuda or cpu")
    args = parser.parse_args()

    pred_dir = Path(args.pred_folder)
    ref_dir = Path(args.ref_folder)
    out_json = Path(args.output_json)
    out_jsonl = Path(args.output_jsonl) if args.output_jsonl else out_json.with_suffix(".jsonl")

    if not pred_dir.exists() or not pred_dir.is_dir():
        raise FileNotFoundError(f"pred_folder not found: {pred_dir}")
    if not ref_dir.exists() or not ref_dir.is_dir():
        raise FileNotFoundError(f"ref_folder not found: {ref_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    pred_files = sorted(pred_dir.glob("*.mp4"))
    if not pred_files:
        raise RuntimeError(f"No mp4 files found in {pred_dir}")

    rows = []
    for pred_path in tqdm(pred_files, desc="Evaluating videos"):
        ref_path = ref_dir / pred_path.name
        if not ref_path.exists():
            # Skip unmatched files but keep traceability.
            rows.append({
                "file": pred_path.name,
                "status": "missing_reference",
            })
            continue

        pred_video = load_video_rgb_tensor(pred_path)
        ref_video = load_video_rgb_tensor(ref_path)

        metrics = compute_video_metrics(
            pred=pred_video,
            ref=ref_video,
            lpips_model=lpips_model,
            device=device,
            skip_frames=args.skip_frames,
        )
        row = {
            "file": pred_path.name,
            "status": "ok",
            **metrics,
        }
        rows.append(row)

    valid_rows = [r for r in rows if r.get("status") == "ok"]
    if not valid_rows:
        raise RuntimeError("No matched videos were evaluated. Check pred/ref filenames.")

    summary = {
        "pred_folder": str(pred_dir),
        "ref_folder": str(ref_dir),
        "num_pred_files": len(pred_files),
        "num_evaluated_files": len(valid_rows),
        "num_missing_reference": len([r for r in rows if r.get("status") == "missing_reference"]),
        "avg_PSNR": float(sum(r["PSNR"] for r in valid_rows) / len(valid_rows)),
        "avg_SSIM": float(sum(r["SSIM"] for r in valid_rows) / len(valid_rows)),
        "avg_LPIPS": float(sum(r["LPIPS"] for r in valid_rows) / len(valid_rows)),
        "skip_frames": args.skip_frames,
    }

    write_json(out_json, summary)
    write_jsonl(out_jsonl, rows)

    print("Saved summary:", out_json)
    print("Saved per-video metrics:", out_jsonl)
    print(
        f"Averages -> PSNR: {summary['avg_PSNR']:.4f}, "
        f"SSIM: {summary['avg_SSIM']:.4f}, LPIPS: {summary['avg_LPIPS']:.4f}"
    )


if __name__ == "__main__":
    main()
