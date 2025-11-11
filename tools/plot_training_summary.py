import os
import re
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


LOSS_NAMES = [
    "loss_disc",
    "loss_gen",
    "loss_fm",
    "loss_mel",
    "loss_kl",
    "loss_subband",
]


def parse_train_log(log_path: str) -> Dict[int, List[List[float]]]:
    """Parse train.log and return a mapping: epoch -> list of [6 losses].

    The logger writes two kinds of lines we care about:
      - 'Train Epoch: {epoch} [...]'
      - '[loss_disc, loss_gen, loss_fm, loss_mel, loss_kl, loss_subband, global_step, lr]'

    We pair the most recent epoch number with subsequent numeric list lines.
    """
    epoch_losses: Dict[int, List[List[float]]] = {}
    cur_epoch: int = None

    # Regex patterns
    re_epoch = re.compile(r"Train Epoch:\s*(\d+)")
    re_list = re.compile(r"\[([^\]]+)\]")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_ep = re_epoch.search(line)
            if m_ep:
                try:
                    cur_epoch = int(m_ep.group(1))
                except Exception:
                    continue
                continue

            # Try to capture the numeric list line
            m_lst = re_list.search(line)
            if m_lst and cur_epoch is not None:
                try:
                    # Split by comma, parse floats
                    parts = [p.strip() for p in m_lst.group(1).split(",")]
                    nums = [float(p) for p in parts]
                    # Expect at least 8 numbers: 6 losses + global_step + lr
                    if len(nums) >= 8:
                        losses6 = nums[:6]
                        epoch_losses.setdefault(cur_epoch, []).append(losses6)
                except Exception:
                    continue

    return epoch_losses


def summarize_epochs(epoch_losses: Dict[int, List[List[float]]]) -> Tuple[int, int, np.ndarray, np.ndarray]:
    if not epoch_losses:
        raise RuntimeError("No epoch loss data parsed from train.log")

    epochs = sorted(epoch_losses.keys())
    first_ep, last_ep = epochs[0], epochs[-1]

    def mean_losses(ep: int) -> np.ndarray:
        arr = np.array(epoch_losses[ep], dtype=np.float64)  # (steps, 6)
        return arr.mean(axis=0)

    first_mean = mean_losses(first_ep)
    last_mean = mean_losses(last_ep)
    return first_ep, last_ep, first_mean, last_mean


def summarize_all_epochs(epoch_losses: Dict[int, List[List[float]]]) -> Tuple[List[int], np.ndarray]:
    """Return (sorted_epochs, means_per_epoch) where means_per_epoch has shape (E, 6)."""
    if not epoch_losses:
        raise RuntimeError("No epoch loss data parsed from train.log")

    epochs = sorted(epoch_losses.keys())
    means = []
    for ep in epochs:
        arr = np.array(epoch_losses[ep], dtype=np.float64)
        means.append(arr.mean(axis=0))
    means_arr = np.stack(means, axis=0)  # (E, 6)
    return epochs, means_arr


def plot_first_last_bar(out_png: str, first_ep: int, last_ep: int, first_mean: np.ndarray, last_mean: np.ndarray):
    x = np.arange(len(LOSS_NAMES))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - w/2, first_mean, width=w, label=f"epoch {first_ep}")
    plt.bar(x + w/2, last_mean, width=w, label=f"epoch {last_ep}")
    plt.xticks(x, LOSS_NAMES, rotation=20)
    plt.ylabel("loss (mean per epoch)")
    plt.title("First vs Last Epoch Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_accuracy_proxy(out_png: str, first_ep: int, last_ep: int, first_mean: np.ndarray, last_mean: np.ndarray):
    # Use mel loss improvement ratio as a proxy for "accuracy"
    mel_idx = LOSS_NAMES.index("loss_mel")
    mel_first = first_mean[mel_idx]
    mel_last = last_mean[mel_idx]
    if mel_first <= 0:
        acc_first = 0.0
        acc_last = 0.0
    else:
        # improvement ratio = (mel_first - mel_current) / mel_first
        acc_first = 0.0
        acc_last = max(0.0, (mel_first - mel_last) / mel_first)

    plt.figure(figsize=(5, 5))
    plt.bar(["epoch1(=0%)", f"epoch{last_ep}"], [acc_first, acc_last], color=["#7777ff", "#55cc55"])
    plt.ylabel("mel loss improvement ratio (proxy accuracy)")
    plt.title("Accuracy (Proxy) from Mel Loss Improvement")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_losses_over_epochs_line(out_png: str, epochs: List[int], means_arr: np.ndarray):
    """Line plot of all 6 losses over epochs.
    means_arr: (E, 6)
    """
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(LOSS_NAMES):
        plt.plot(epochs, means_arr[:, i], label=name)
    plt.xlabel("epoch")
    plt.ylabel("loss (mean per epoch)")
    plt.title("Losses over Epochs (line)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_mel_accuracy_proxy_over_epochs_line(out_png: str, epochs: List[int], means_arr: np.ndarray):
    """Line plot of mel loss improvement ratio over epochs (baseline = first epoch)."""
    mel_idx = LOSS_NAMES.index("loss_mel")
    mel_first = means_arr[0, mel_idx]
    if mel_first <= 0:
        acc = np.zeros(len(epochs), dtype=np.float64)
    else:
        acc = np.clip((mel_first - means_arr[:, mel_idx]) / mel_first, 0.0, 1.0)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, marker="o", color="#55cc55", label="mel improvement ratio")
    plt.xlabel("epoch")
    plt.ylabel("mel loss improvement ratio (proxy accuracy)")
    plt.title("Proxy Accuracy over Epochs (line)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, required=True, help="Path to logs/<model_name> directory (contains train.log)")
    ap.add_argument("--outdir", type=str, default=None, help="Where to save PNGs (default: same as logdir; if --use-gurafu set, logs/<model>/gurafu)")
    ap.add_argument("--use-gurafu", action="store_true", help="If set and --outdir not provided, save under <logdir>/gurafu")
    args = ap.parse_args()

    logdir = args.logdir
    # Resolve output directory preference
    if args.outdir:
        outdir = args.outdir
    elif args.use_gurafu:
        outdir = os.path.join(logdir, "gurafu")
    else:
        outdir = logdir
    os.makedirs(outdir, exist_ok=True)

    log_path = os.path.join(logdir, "train.log")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"train.log not found in {logdir}")

    epoch_losses = parse_train_log(log_path)
    first_ep, last_ep, first_mean, last_mean = summarize_epochs(epoch_losses)
    epochs, means_arr = summarize_all_epochs(epoch_losses)

    out_png1 = os.path.join(outdir, "first_last_epoch_losses.png")
    plot_first_last_bar(out_png1, first_ep, last_ep, first_mean, last_mean)

    out_png2 = os.path.join(outdir, "accuracy_proxy_first_last.png")
    plot_accuracy_proxy(out_png2, first_ep, last_ep, first_mean, last_mean)

    print("Saved:", out_png1)
    print("Saved:", out_png2)

    # New: line plots over all epochs
    out_png3 = os.path.join(outdir, "losses_over_epochs_line.png")
    plot_losses_over_epochs_line(out_png3, epochs, means_arr)

    out_png4 = os.path.join(outdir, "mel_accuracy_proxy_over_epochs_line.png")
    plot_mel_accuracy_proxy_over_epochs_line(out_png4, epochs, means_arr)

    print("Saved:", out_png3)
    print("Saved:", out_png4)


if __name__ == "__main__":
    main()
