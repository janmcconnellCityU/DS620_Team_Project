import subprocess
import pathlib
import sys

# AudioMNIST dataset slug
DATASET = "sripaadsrinivasan/audio-mnist"
OUTDIR = pathlib.Path("data/raw")
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Downloading AudioMNIST dataset from Kaggle...")
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(OUTDIR), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Download complete.")
    except subprocess.CalledProcessError:
        print("⚠️  Download failed. Make sure your Kaggle API credentials are set up.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
