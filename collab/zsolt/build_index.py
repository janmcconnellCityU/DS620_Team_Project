"""This script scans through all generated .npy feature files
 and creates a CSV file that maps each spectrogram file path
 to its corresponding digit label (0–9).
 The resulting "index.csv" will be used later during training
 to load data with labels."""


from pathlib import Path
import csv
import re
import sys

# Define main directories
ROOT = Path(".")
RAW = ROOT / "data" / "raw"             # original audio files (unused here)
FEAT = ROOT / "data" / "spectrograms"   # folder containing processed .npy features
OUT = ROOT / "data" / "processed"       # output folder for the index.csv
OUT.mkdir(parents=True, exist_ok=True)

def infer_digit_from_name(name: str):
    """
    Try to guess the spoken digit (0–9) from a filename pattern.
    Supported examples:
      - '7_speaker_003.wav'  -> 7
      - 'speaker_7_003.wav'  -> 7
      - or parent folder named '7'
    Returns the digit as int, or None if it can’t be found.
    """
    # Case 1: filename starts with a digit (e.g., '3_speaker...')
    m = re.match(r"^([0-9])[_-]", name)
    if m:
        return int(m.group(1))

    # Case 2: single digit surrounded by underscores or hyphens
    m = re.search(r"[_-]([0-9])[_-]", name)
    if m:
        return int(m.group(1))

    # Case 3: fallback handled later (folder name check)
    return None

def main():
    # Find all spectrogram feature files (.npy) recursively
    npys = sorted(FEAT.rglob("*.npy"))
    if not npys:
        print("No .npy files found in data/spectrograms. Run feature extraction first.")
        sys.exit(1)

    rows, bad = [], []  # store successful and failed label detections

    for p in npys:
        # Try to infer the digit label from the filename
        digit = infer_digit_from_name(p.stem)

        # If that fails, check if the parent folder is named with a digit
        if digit is None:
            parent = p.parent.name
            if parent.isdigit() and len(parent) == 1:
                digit = int(parent)

        # If still no label, record it as a problematic file
        if digit is None:
            bad.append(str(p))
            continue

        # Store a row for the CSV
        rows.append({"path": str(p.as_posix()), "label": digit})

    # Warn if some files could not be labeled
    if bad:
        print(f"⚠ Could not infer labels for {len(bad)} files. First few:\n  - " +
              "\n  - ".join(bad[:10]))
        print("Tip: If filenames don't include the digit, put those files "
              "into folders named 0–9 under data/spectrograms/ and rerun.")

    # Write the collected paths and labels to a CSV file
    out_csv = OUT / "index.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} indexed samples to {out_csv}")

# Execute only if run directly (not imported)
if __name__ == "__main__":
    main()