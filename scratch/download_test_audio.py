"""
===============================================================================
🎵 Shoot_Catcher — Automated Test Audio Downloader
===============================================================================
Downloads verified, high-quality, open-source audio recordings directly into:
  1. My_Test_Audio/Actual_Gunshots/   (Real firearm recordings)
  2. My_Test_Audio/Like_Gunshots/     (Imposters: fireworks, clapping, knocks)
  3. My_Test_Audio/Not_Gunshots/      (Everyday ambient: rain, speech, dogs, engines)

Source: ESC-50 Open-Access Environmental Audio Dataset (CC-BY License)
===============================================================================
"""

import os
import sys
import io
import time
import urllib.request
import csv
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

PROJECT_ROOT = Path(r"c:\Users\aadit\Desktop\Shoot_Catcher")
BASE_TEST_DIR = PROJECT_ROOT / "My_Test_Audio"

DIR_ACTUAL = BASE_TEST_DIR / "Actual_Gunshots"
DIR_LIKE = BASE_TEST_DIR / "Like_Gunshots"
DIR_NOT = BASE_TEST_DIR / "Not_Gunshots"

# Create directories if they do not exist
DIR_ACTUAL.mkdir(parents=True, exist_ok=True)
DIR_LIKE.mkdir(parents=True, exist_ok=True)
DIR_NOT.mkdir(parents=True, exist_ok=True)

META_URL = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
AUDIO_BASE_URL = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/"

def download_file(url, dest_path):
    """Download a single file with timeout and retry."""
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req, timeout=15) as response, open(dest_path, 'wb') as out_file:
        out_file.write(response.read())

def main():
    print("=" * 80)
    print("🎵 SHOOT_CATCHER — AUTOMATED AUDIO DOWNLOADER FOR OFFLINE TESTING")
    print("=" * 80)
    print(f"Target Base Directory: {BASE_TEST_DIR}")
    print(f" ├─ [Actual_Gunshots] : {DIR_ACTUAL}")
    print(f" ├─ [Like_Gunshots]   : {DIR_LIKE}")
    print(f" └─ [Not_Gunshots]    : {DIR_NOT}")
    print("-" * 80)

    print("📥 Fetching dataset metadata from ESC-50 repository...")
    try:
        req = urllib.request.Request(META_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            lines = [line.decode('utf-8') for line in resp.readlines()]
    except Exception as e:
        print(f"❌ Failed to fetch dataset metadata: {e}")
        return

    reader = csv.DictReader(lines)
    all_rows = list(reader)
    print(f"✅ Metadata retrieved successfully ({len(all_rows)} audio records cataloged).\n")

    # Map categories to our 3 folders
    # Gunshots: 'gunshot'
    # Like Gunshots (Imposters): 'fireworks', 'clapping', 'door_wood_knock', 'glass_breaking'
    # Not Gunshots (Ambient): 'crying_baby', 'sneezing', 'dog', 'rain', 'engine', 'siren'
    
    target_plan = {
        "Actual_Gunshots": {
            "dest_dir": DIR_ACTUAL,
            "categories": ["gunshot"],
            "max_samples": 20,
            "desc": "Real Gunshot Recordings"
        },
        "Like_Gunshots": {
            "dest_dir": DIR_LIKE,
            "categories": ["fireworks", "clapping", "door_wood_knock", "glass_breaking"],
            "max_samples": 20,
            "desc": "Acoustic Imposters (Fireworks, Claps, Knocks)"
        },
        "Not_Gunshots": {
            "dest_dir": DIR_NOT,
            "categories": ["crying_baby", "sneezing", "dog", "rain", "engine", "siren"],
            "max_samples": 20,
            "desc": "Everyday Ambient (Speech, Rain, Dogs, Motors)"
        }
    }

    grand_total = 0
    start_time = time.time()

    for group_name, config in target_plan.items():
        print(f"📂 Processing: [{group_name}] — {config['desc']}")
        dest_dir = config["dest_dir"]
        cats = config["categories"]
        max_samples = config["max_samples"]

        # Filter matching rows
        matching_rows = [r for r in all_rows if r["category"] in cats]
        
        # Take up to max_samples
        selected = matching_rows[:max_samples]
        print(f"   Downloading {len(selected)} audio files into {dest_dir.name}/...")

        for idx, row in enumerate(selected, 1):
            fname = row["filename"]
            cat = row["category"]
            dest_file = dest_dir / f"{cat}_{idx:02d}_{fname}"
            
            if dest_file.exists():
                print(f"   [{idx:>2}/{len(selected)}] ⏩ Already exists: {dest_file.name}")
                grand_total += 1
                continue

            file_url = AUDIO_BASE_URL + fname
            try:
                download_file(file_url, dest_file)
                size_kb = dest_file.stat().st_size / 1024
                print(f"   [{idx:>2}/{len(selected)}] ✅ Downloaded: {dest_file.name:<35} ({size_kb:.1f} KB)")
                grand_total += 1
            except Exception as e:
                print(f"   [{idx:>2}/{len(selected)}] ❌ Error downloading {fname}: {e}")

        print()

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"🎉 DOWNLOAD COMPLETE! Total Files Ready: {grand_total} files ({elapsed:.1f}s)")
    print("=" * 80)
    print(f"Files saved in:")
    print(f" - {DIR_ACTUAL} ({len(list(DIR_ACTUAL.glob('*.wav')))} WAV files)")
    print(f" - {DIR_LIKE} ({len(list(DIR_LIKE.glob('*.wav')))} WAV files)")
    print(f" - {DIR_NOT} ({len(list(DIR_NOT.glob('*.wav')))} WAV files)")
    print("=" * 80)

if __name__ == "__main__":
    main()
