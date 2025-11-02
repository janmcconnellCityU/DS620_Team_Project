"""check_features.py is a quick sanity-check tool. It loads a few of the generated .npy spectrogram files, prints their shape,
data type, and value range, and saves a sample image (sample_preview.png) that visualizes one spectrogram.
Basically, it confirms that the feature extraction worked and that the files contain valid numerical data before starting model training."""


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".")
FEAT = ROOT / "data" / "spectrograms"

def describe(arr, name):
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} "
          f"min={float(arr.min()):.2f} max={float(arr.max()):.2f} mean={float(arr.mean()):.2f}")

def main():
    files = sorted(FEAT.rglob("*.npy"))
    if not files:
        print("No .npy files found under data/spectrograms. Did you run extraction?")
        return

    # peek at a few
    for i, f in enumerate(files[:5], 1):
        x = np.load(f)
        describe(x, f.name)

    # save a quick visualization of the first one
    x0 = np.load(files[0])
    plt.figure()
    plt.imshow(x0, origin="lower", aspect="auto")
    plt.title(files[0].name)
    plt.xlabel("time frames"); plt.ylabel("mel bins")
    out = ROOT / "data" / "spectrograms" / "sample_preview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved preview to {out}")

if __name__ == "__main__":
    main()
