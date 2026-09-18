"""
===============================================================================
🎯 Shoot_Catcher — Kaggle Gunshots Dataset (GSD) Organizer
===============================================================================
Organizes the 716 raw files from 'Gunshots dataset (GSD)' into:
  - External_Datasets/Kaggle_Gunshots_Dataset/Actual_Gunshots/  (Discharges, bursts, shots)
  - External_Datasets/Kaggle_Gunshots_Dataset/Fake_Gunshots/    (Mechanical clicks, dry fire, safety, reloads)
  - External_Datasets/Kaggle_Gunshots_Dataset/Not_Gunshots/     (Ambient / Non-firing elements)
===============================================================================
"""

import os
import sys
import io
import shutil
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

PROJECT_ROOT = Path(r"c:\Users\aadit\Desktop\Shoot_Catcher")
SOURCE_DIR = PROJECT_ROOT / "Gunshots dataset (GSD)"
TARGET_DIR = PROJECT_ROOT / "External_Datasets" / "Kaggle_Gunshots_Dataset"

DIR_ACTUAL = TARGET_DIR / "Actual_Gunshots"
DIR_FAKE = TARGET_DIR / "Fake_Gunshots"
DIR_NOT = TARGET_DIR / "Not_Gunshots"

for d in [DIR_ACTUAL, DIR_FAKE, DIR_NOT]:
    d.mkdir(parents=True, exist_ok=True)

MECHANICAL_KEYWORDS = [
    "dry fire", "safety on", "safety off", "insert round", "full cycle",
    "detach", "bolt back", "bolt forward", "bolt up", "load mag",
    "drop slide", "full rack", "extract shells", "close cylinder",
    "cock hammer", "open cylinder", "empty cylinder", "holster",
    "unholster", "slide release", "magazine release", "cocking",
    "handling", "unload", "eject", "reloading"
]

DISCHARGE_KEYWORDS = [
    "shot", "burst", "fire", "blast", "pop", "crack", "spacious",
    "crisp", "tail", "gunshot", "silencer", "suppressed", "loud"
]

def classify_filename(filename_lower):
    """Categorize file into Actual_Gunshots, Fake_Gunshots (mechanical clicks), or Not_Gunshots."""
    # Check for live firing discharges first (e.g. burst, single shot)
    has_discharge = any(kw in filename_lower for kw in ["burst", "single shot", "gunshot", "blast", "shots", "firing"])
    has_mechanical = any(kw in filename_lower for kw in MECHANICAL_KEYWORDS)

    if has_discharge:
        return "Actual_Gunshots"
    elif has_mechanical:
        return "Fake_Gunshots"
    elif any(kw in filename_lower for kw in DISCHARGE_KEYWORDS):
        return "Actual_Gunshots"
    else:
        # Default mechanical/foley if uncertain
        return "Fake_Gunshots"

def main():
    print("=" * 80)
    print("📂 KAGGLE GUNSHOTS DATASET (GSD) — 3-FOLDER SEPARATION")
    print("=" * 80)
    print(f"Source Folder: {SOURCE_DIR}")
    print(f"Target Base  : {TARGET_DIR}")
    print(f" ├─ [Actual_Gunshots] : {DIR_ACTUAL}")
    print(f" ├─ [Fake_Gunshots]   : {DIR_FAKE}")
    print(f" └─ [Not_Gunshots]    : {DIR_NOT}")
    print("=" * 80)

    if not SOURCE_DIR.exists():
        print(f"❌ Source directory '{SOURCE_DIR}' does not exist!")
        return

    # Find all wav files in source
    source_files = list(SOURCE_DIR.rglob("*.wav"))
    print(f"Found {len(source_files)} total audio files in '{SOURCE_DIR.name}'\n")

    stats = {"Actual_Gunshots": 0, "Fake_Gunshots": 0, "Not_Gunshots": 0}

    for idx, fpath in enumerate(source_files, 1):
        cat = classify_filename(fpath.name.lower())
        stats[cat] += 1
        dest_dir = TARGET_DIR / cat

        # Prefix with parent category (weapon type) to avoid collision and keep clean metadata
        weapon_prefix = fpath.parent.name.replace(" ", "_").replace(",", "")
        dest_name = f"{weapon_prefix}__{fpath.name}"
        dest_file = dest_dir / dest_name

        # Copy if not exists
        if not dest_file.exists():
            # Use extended path for Windows long filenames
            src_long = "\\\\?\\" + str(fpath.resolve())
            dst_long = "\\\\?\\" + str(dest_file.resolve())
            shutil.copy2(src_long, dst_long)

    print("\n" + "=" * 80)
    print("✅ KAGGLE DATASET SEPARATION COMPLETE!")
    print(f"   🔫 Actual Gunshots (Discharges / Bursts) : {stats['Actual_Gunshots']} files")
    print(f"   ⚙️ Fake Gunshots (Mechanical / Handling): {stats['Fake_Gunshots']} files")
    print(f"   🌿 Not Gunshots (Ambient / Other)       : {stats['Not_Gunshots']} files")
    print(f"   📊 Total Organized Files                : {sum(stats.values())} files")
    print(f"   📁 Target Location                      : {TARGET_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
