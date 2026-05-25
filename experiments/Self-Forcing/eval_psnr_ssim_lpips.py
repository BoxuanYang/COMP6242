import argparse
import importlib.util
import json
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_longcat_metric_impl():
    """Load the original LongCat metric implementation used by the upstream repo."""
    metric_path = Path(__file__).resolve().parents[1] / "LongCat" / "longcat_video" / "utils" / "metric.py"
    if not metric_path.exists():
        raise FileNotFoundError(f"LongCat metric implementation not found: {metric_path}")

    spec = importlib.util.spec_from_file_location("longcat_metric_impl", str(metric_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from: {metric_path}")
    longcat_metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(longcat_metric)

    return longcat_metric


def resolve_device(device_arg: str) -> torch.device:
    """Resolve requested device and fall back safely when CUDA is unavailable."""
    device_arg = device_arg.lower().strip()
    if device_arg not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported --device value: {device_arg}. Use 'cpu' or 'cuda'.")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device=cuda requested but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def compute_psnr_ssim_lpips(video1_tensor, video2_tensor, longcat_metric, video_label: str):
    """Compute only PSNR/SSIM/LPIPS with LongCat-compatible frame semantics."""
    if video1_tensor.shape != video2_tensor.shape:
        raise ValueError(f"Videos must have the same shape. {video1_tensor.shape} != {video2_tensor.shape}")

    num_frames = int(video1_tensor.shape[0])
    if num_frames <= 1:
        raise ValueError(f"Not enough frames for metric computation: {num_frames}")

    psnr_values = []
    ssim_values = []
    lpips_values = []

    with torch.no_grad():
        # Keep the original LongCat convention: start from frame index 1.
        for i in tqdm(range(1, num_frames), desc=f"frames[{video_label}]", leave=False):
            frame1 = video1_tensor[i].unsqueeze(0)
            frame2 = video2_tensor[i].unsqueeze(0)

            mse = F.mse_loss(frame1, frame2, reduction="mean")
            psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
            ssim = longcat_metric.calculate_ssim(frame1, frame2)
            lpips_value = longcat_metric.lpips_model(frame1, frame2)

            psnr_values.append(psnr.item())
            ssim_values.append(ssim.item())
            lpips_values.append(lpips_value.item())

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

    device = resolve_device(args.device)

    longcat_metric = load_longcat_metric_impl()
    longcat_metric.lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

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

        try:
            pred_video = longcat_metric.load_video(str(pred_path)).to(device)
            ref_video = longcat_metric.load_video(str(ref_path)).to(device)

            if args.skip_frames > 0:
                pred_video = pred_video[args.skip_frames:]
                ref_video = ref_video[args.skip_frames:]

            metrics = compute_psnr_ssim_lpips(
                video1_tensor=pred_video,
                video2_tensor=ref_video,
                longcat_metric=longcat_metric,
                video_label=pred_path.name,
            )
        except Exception as exc:
            rows.append(
                {
                    "file": pred_path.name,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        row = {
            "file": pred_path.name,
            "status": "ok",
            "num_eval_frames": int(metrics["num_eval_frames"]),
            "PSNR": float(metrics["PSNR"]),
            "SSIM": float(metrics["SSIM"]),
            "LPIPS": float(metrics["LPIPS"]),
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
