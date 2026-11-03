"""CPU-only codec smoke/overfit probe on three existing RGB snapshot pairs.

Run with python tools/smoke_dual_view_codec.py --source-dir /path/to/paired_rgb.
Artifacts go into a new outputs/codec_smoke_* directory. No datasets,
pretrained weights, VLA, simulator, GPU, or formal training run are required.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codec-smoke-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from compress.dual_view_codec import DualViewCodec


def save_tensor(tensor, path):
    array = tensor.detach().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(path)


def to_tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()[None] / 255


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory with *_agentview_image.png and matching *_robot0_eye_in_hand_image.png")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")
    sources = sorted(args.source_dir.glob("*_agentview_image.png"))[:3]
    if len(sources) != 3:
        parser.error("Need three existing main/wrist RGB snapshot pairs")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or REPO / "outputs" / f"codec_smoke_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs/native", "inputs/smoke", "inputs/overfit", "before", "after", "comparisons"):
        (output / name).mkdir(parents=True)
    log_file = (output / "run.log").open("w", buffering=1)

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")

    log(f"Output: {output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261103)
    sizes = [((128, 128), (112, 96)), ((95, 113), (83, 97)), ((128, 96), (65, 81))]
    smoke, overfit, provenance = [], [], []
    for i, source in enumerate(sources):
        smoke_pair, train_pair = [], []
        for view, source_path, hw in zip(("main", "wrist"),
                (source, source.with_name(source.name.replace("agentview", "robot0_eye_in_hand"))), sizes[i]):
            image = Image.open(source_path).convert("RGB")
            image.save(output / "inputs/native" / f"f{i}_{view}.png")
            scaled = image.resize((hw[1], hw[0]), Image.Resampling.LANCZOS)
            scaled.save(output / "inputs/smoke" / f"f{i}_{view}.png")
            small = image.resize((96, 96), Image.Resampling.LANCZOS)
            small.save(output / "inputs/overfit" / f"f{i}_{view}.png")
            smoke_pair.append(to_tensor(scaled))
            train_pair.append(to_tensor(small))
            provenance.append({"frame": i, "view": view, "source": str(source_path),
                               "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                               "original_wh": image.size, "smoke_hw": hw})
        smoke.append(smoke_pair)
        overfit.append(train_pair)
    model = DualViewCodec()
    config = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "seed": 20261103, "device": "cpu", "threads": args.threads,
              "steps": args.steps, "pairs": 3, "overfit_hw": [96, 96],
              "loss": "estimated_bpp + 20*(MSE_main+MSE_wrist); separate aux loss",
              "main_lr": 0.001, "aux_lr": 0.001, "torch": torch.__version__,
              "note": "Sparse ordered snapshots; only a functional overfit probe, not RD evaluation.",
              "inputs": provenance}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    log("CPU only; 3 sparse RGB pairs, no external downloads or pretrained weights")
    rates = [(0, 0.), (3, .5), (7, 0.)]
    metric_rows = []
    checks = {"exact_forward_decode": True, "exact_cache": True, "episode_reset": True,
              "missing_history_rejected": True, "different_sizes": True}

    @torch.no_grad()
    def evaluate(stage):
        model.eval()
        model.update(force=True)  # Rebuild learned CDFs before actual coding.
        receiver = DualViewCodec().eval()
        receiver.load_state_dict(model.state_dict())
        del receiver.analysis.g_a
        calls = {"spatial": 0, "temporal": 0}
        timing = {}

        def count(name):
            def hook(*unused):
                calls[name] += 1
            return hook

        def start(*unused):
            timing["start"] = perf_counter()

        def stop(*unused):
            timing["reconstruction_ms"] = (perf_counter() - timing["start"]) * 1000

        handles = [receiver.decoder.spatial.register_forward_hook(count("spatial")),
                   receiver.decoder.temporal.register_forward_hook(count("temporal")),
                   receiver.decoder.register_forward_pre_hook(start),
                   receiver.decoder.register_forward_hook(stop)]
        state = None
        for rate_id, rate in enumerate(rates):
            directory = output / stage / f"gain_{rate[0]}_a{rate[1]}"
            directory.mkdir()
            expected_state = None
            for i, (main_rgb, wrist_rgb) in enumerate(smoke):
                episode = rate_id + 1  # Reuse old state intentionally at new frame 0.
                expected = model(main_rgb, wrist_rgb, expected_state, episode, i,
                                 main_rate=rate, wrist_rate=rate)
                begin = perf_counter()
                packet = model.compress(main_rgb, wrist_rgb, episode, i,
                                        main_rate=rate, wrist_rate=rate)
                encode_ms = (perf_counter() - begin) * 1000
                begin = perf_counter()
                actual = receiver.decompress(packet, state)
                decode_ms = (perf_counter() - begin) * 1000
                reconstruction_ms = timing["reconstruction_ms"]
                if i == 0:
                    clean = receiver.decompress(packet)
                    torch.testing.assert_close(clean["main_hat"], actual["main_hat"], atol=0, rtol=0)
                    assert actual["history_used"] == 0
                else:
                    try:
                        receiver.decompress(packet)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Missing history was accepted")
                assert actual["history_used"] == min(i, 2)
                for cached, reference in zip(actual["state"].history, expected["state"].history):
                    torch.testing.assert_close(cached.latent, reference.latent, atol=0, rtol=0)
                    torch.testing.assert_close(cached.rgb, reference.rgb, atol=0, rtol=0)
                psnrs = {}
                for view, original in (("main", main_rgb), ("wrist", wrist_rgb)):
                    reconstructed = actual[f"{view}_hat"]
                    torch.testing.assert_close(reconstructed, expected[f"{view}_hat"], atol=0, rtol=0)
                    assert reconstructed.shape == original.shape
                    mse = F.mse_loss(reconstructed, original).item()
                    psnrs[view] = -10 * math.log10(max(mse, 1e-12))
                    save_tensor(reconstructed, directory / f"f{i}_{view}_reconstructed.png")
                (directory / f"f{i}.bin").write_bytes(packet)
                pixels = sum(t.shape[-2] * t.shape[-1] for t in (main_rgb, wrist_rgb))
                assert actual["actual_bpp"] == 8 * len(packet) / pixels
                row = {"stage": stage, "level": rate[0], "alpha": rate[1], "frame": i,
                       "main_hw": f"{main_rgb.shape[-2]}x{main_rgb.shape[-1]}",
                       "wrist_hw": f"{wrist_rgb.shape[-2]}x{wrist_rgb.shape[-1]}",
                       "history_used": actual["history_used"], "bytes": len(packet), "pixels": pixels,
                       "actual_bpp": actual["actual_bpp"], "estimated_bpp": expected["estimated_bpp"].item(),
                       "psnr_main_db": psnrs["main"], "psnr_wrist_db": psnrs["wrist"],
                       "encode_ms": encode_ms, "decode_ms": decode_ms,
                       "reconstruction_ms": reconstruction_ms}
                metric_rows.append(row)
                expected_state, state = expected["state"], actual["state"]
            log(f"{stage} rate={rate}: mean PSNR main/wrist="
                f"{np.mean([r['psnr_main_db'] for r in metric_rows[-3:]]):.2f}/"
                f"{np.mean([r['psnr_wrist_db'] for r in metric_rows[-3:]]):.2f} dB")
        for handle in handles:
            handle.remove()
        assert calls["spatial"] > 0 and calls["temporal"] > 0
        checks[f"{stage}_attention_calls"] = calls

    evaluate("before")
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    quantiles = model.analysis.hyperprior.entropy_bottleneck.quantiles
    optimizer = torch.optim.Adam([p for p in model.parameters() if p is not quantiles], lr=.001)
    aux_optimizer = torch.optim.Adam([quantiles], lr=.001)
    train_rows = []
    seen_grad = {name: False for name in initial}
    rate_schedule = [(s, 0.) for s in range(8)] + [(s, .5) for s in range(7)]
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)
        state = None
        total, rate_total, distortion_total = 0, 0, 0
        for i, (main_rgb, wrist_rgb) in enumerate(overfit):
            rate = rate_schedule[(step + i) % len(rate_schedule)]
            result = model(main_rgb, wrist_rgb, state, episode_id=0, frame_index=i,
                           main_rate=rate, wrist_rate=rate)
            distortion = F.mse_loss(result["main_hat"], main_rgb) + F.mse_loss(result["wrist_hat"], wrist_rgb)
            rate_total = rate_total + result["estimated_bpp"] / 3
            distortion_total = distortion_total + distortion / 3
            total = total + (result["estimated_bpp"] + 20 * distortion) / 3
            state = result["state"]
        assert torch.isfinite(total)
        total.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
                seen_grad[name] |= p.grad.abs().sum().item() > 0
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p is not quantiles], 5.)
        optimizer.step()
        aux_optimizer.zero_grad(set_to_none=True)
        aux = model.aux_loss()
        aux.backward()
        seen_grad["analysis.hyperprior.entropy_bottleneck.quantiles"] |= quantiles.grad.abs().sum().item() > 0
        aux_optimizer.step()
        row = {"step": step + 1, "loss": total.item(), "estimated_bpp": rate_total.item(),
               "distortion_mse_sum": distortion_total.item(), "aux_loss": aux.item()}
        train_rows.append(row)
        if step == 0 or (step + 1) % 10 == 0:
            log(f"overfit {step+1:03d}/{args.steps}: loss={row['loss']:.6f}, "
                f"R={row['estimated_bpp']:.6f}, D={row['distortion_mse_sum']:.6f}")
    write_csv(output / "loss.csv", train_rows)
    update_rows = []
    for name, p in model.named_parameters():
        delta = (p.detach() - initial[name]).abs()
        update_rows.append({"parameter": name, "nonzero_finite_gradient_seen": seen_grad[name],
                            "changed_elements": (delta != 0).sum().item(), "elements": p.numel(),
                            "max_abs_change": delta.max().item()})
    write_csv(output / "parameter_updates.csv", update_rows)
    assert all(r["nonzero_finite_gradient_seen"] and r["changed_elements"] > 0 for r in update_rows), \
        [r["parameter"] for r in update_rows if not r["nonzero_finite_gradient_seen"] or not r["changed_elements"]]
    assert train_rows[-1]["loss"] < train_rows[0]["loss"], "Overfit loss did not decrease"
    evaluate("after")
    torch.save(model.state_dict(), output / "overfit_checkpoint.pt")
    write_csv(output / "metrics.csv", metric_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, key in zip(axes, ("loss", "estimated_bpp", "distortion_mse_sum")):
        ax.plot([r["step"] for r in train_rows], [r[key] for r in train_rows])
        ax.set(xlabel="Functional overfit step", ylabel=key)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=150)
    plt.close(fig)
    # Image panels keep the input/resolution visible, with no image enhancement.
    for rate in rates:
        tag = f"gain_{rate[0]}_a{rate[1]}"
        canvas = Image.new("RGB", (600, 6 * 180 + 30), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        for col, title in enumerate(("Input RGB", "Before overfit", "After overfit")):
            draw.text((col * 200 + 5, 8), title, fill="black")
        for i in range(3):
            for v, view in enumerate(("main", "wrist")):
                row = i * 2 + v
                paths = [output / "inputs/smoke" / f"f{i}_{view}.png"] + [
                    output / stage / tag / f"f{i}_{view}_reconstructed.png" for stage in ("before", "after")]
                for col, path in enumerate(paths):
                    image = Image.open(path)
                    image.thumbnail((190, 150))
                    canvas.paste(image, (col * 200 + 5, 30 + row * 180))
                draw.text((5, 185 + row * 180), f"frame {i} / {view}", fill="black")
        canvas.save(output / "comparisons" / f"{tag}.png")
        if rate == (3, .5):
            canvas.crop((0, 0, 600, 390)).save(output / "comparisons/preview.png")
    summary = {"output": str(output), "checks": checks, "parameter_tensors": len(update_rows),
               "all_parameter_tensors_updated": True,
               "initial_loss": train_rows[0]["loss"], "final_loss": train_rows[-1]["loss"],
               "loss_drop_percent": 100 * (1 - train_rows[-1]["loss"] / train_rows[0]["loss"]),
               "rates": []}
    for stage in ("before", "after"):
        for level, alpha in rates:
            rows = [r for r in metric_rows if r["stage"] == stage and r["level"] == level]
            summary["rates"].append({"stage": stage, "level": level, "alpha": alpha,
                "actual_bpp": 8 * sum(r["bytes"] for r in rows) / sum(r["pixels"] for r in rows),
                "mean_psnr_main_db": float(np.mean([r["psnr_main_db"] for r in rows])),
                "mean_psnr_wrist_db": float(np.mean([r["psnr_wrist_db"] for r in rows])),
                "mean_decode_ms": float(np.mean([r["decode_ms"] for r in rows])),
                "mean_reconstruction_ms": float(np.mean([r["reconstruction_ms"] for r in rows]))})
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"PASS: all {len(update_rows)} parameter tensors updated; "
        f"loss {summary['initial_loss']:.6f} -> {summary['final_loss']:.6f}")
    log(json.dumps(summary["rates"], indent=2))
    log_file.close()


if __name__ == "__main__":
    main()
