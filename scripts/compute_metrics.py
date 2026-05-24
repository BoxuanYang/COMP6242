"""
Compute PSNR / SSIM / LPIPS between a reference video (BF16) and a predicted video.

Usage:
    python scripts/compute_metrics.py --ref outputs/BF16/0-0_ema.mp4 --pred outputs/QVG_2bit/0-0_ema.mp4

Outputs per-frame mean for PSNR, SSIM, LPIPS, plus number of frames compared.
LPIPS uses AlexNet backbone (lightweight, CPU-friendly).
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import lpips


def read_video_frames(path):
    """Read all frames from an mp4 as a list of uint8 RGB numpy arrays (H, W, 3)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2 returns BGR; convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def to_lpips_tensor(frame_uint8):
    """Convert (H, W, 3) uint8 RGB -> (1, 3, H, W) float in [-1, 1]."""
    t = torch.from_numpy(frame_uint8).float() / 255.0  # [0, 1]
    t = t.permute(2, 0, 1).unsqueeze(0)                # (1, 3, H, W)
    t = t * 2.0 - 1.0                                  # [-1, 1]
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference video (BF16)")
    parser.add_argument("--pred", required=True, help="Predicted video (compressed method)")
    args = parser.parse_args()

    print(f"Reading reference: {args.ref}")
    ref_frames = read_video_frames(args.ref)
    print(f"Reading predicted: {args.pred}")
    pred_frames = read_video_frames(args.pred)

    n_ref, n_pred = len(ref_frames), len(pred_frames)
    n = min(n_ref, n_pred)
    if n_ref != n_pred:
        print(f"WARNING: frame count mismatch (ref={n_ref}, pred={n_pred}); comparing first {n}")
    else:
        print(f"Frame count: {n} (matched)")

    if ref_frames[0].shape != pred_frames[0].shape:
        raise RuntimeError(
            f"Resolution mismatch: ref={ref_frames[0].shape} vs pred={pred_frames[0].shape}"
        )
    print(f"Resolution: {ref_frames[0].shape}")

    print("Loading LPIPS (AlexNet)...")
    loss_fn = lpips.LPIPS(net="alex", verbose=False)
    loss_fn.eval()

    psnrs, ssims, lpipses = [], [], []
    with torch.no_grad():
        for i in range(n):
            ref = ref_frames[i]
            pred = pred_frames[i]
            # PSNR & SSIM on uint8 RGB
            psnrs.append(psnr_fn(ref, pred, data_range=255))
            ssims.append(ssim_fn(ref, pred, channel_axis=2, data_range=255))
            # LPIPS on [-1, 1] float tensor
            r_t = to_lpips_tensor(ref)
            p_t = to_lpips_tensor(pred)
            lp = loss_fn(r_t, p_t).item()
            lpipses.append(lp)
            if (i + 1) % 30 == 0 or i == n - 1:
                print(f"  frame {i+1}/{n}  PSNR={psnrs[-1]:.2f}  SSIM={ssims[-1]:.4f}  LPIPS={lpipses[-1]:.4f}")

    print("\n========== Results ==========")
    print(f"Reference : {args.ref}")
    print(f"Predicted : {args.pred}")
    print(f"Frames    : {n}")
    print(f"PSNR  (dB, higher better) : {np.mean(psnrs):.4f}  ± {np.std(psnrs):.4f}")
    print(f"SSIM  (0-1, higher better): {np.mean(ssims):.4f}  ± {np.std(ssims):.4f}")
    print(f"LPIPS (lower better)      : {np.mean(lpipses):.4f}  ± {np.std(lpipses):.4f}")
    print("=============================")


if __name__ == "__main__":
    main()