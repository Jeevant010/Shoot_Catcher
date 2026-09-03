"""
===============================================================================
🎵 Shoot_Catcher — Automated Audio Downloader for Option A Testing
===============================================================================
Downloads 60 curated, real-world .wav audio files directly into:
  1. My_Test_Audio/Actual_Gunshots/  (20 Real Gunshots: AK-47, Desert Eagle, Magnum, Rifles, Snipers)
  2. My_Test_Audio/Like_Gunshots/    (20 Imposters: Fireworks, Clapping, Door Knocks, Glass Breaking)
  3. My_Test_Audio/Not_Gunshots/     (20 Ambient: Rain, Engines, Dogs, Sirens, Sneezing)

Sources:
  - Audio Event Analysis Open Database (Firearms: AK47, Magnum, Rifles, Snipers)
  - ESC-50 Environmental Audio Benchmark (CC-BY License)
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

DIR_ACTUAL.mkdir(parents=True, exist_ok=True)
DIR_LIKE.mkdir(parents=True, exist_ok=True)
DIR_NOT.mkdir(parents=True, exist_ok=True)

# Base URLs
FIREARMS_BASE_URL = "https://raw.githubusercontent.com/h-sami-ullah/Audio-event-analysis-and-feature-extraction-using-MATLAB/main/data/"
ESC50_META_URL = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO_BASE = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/"

# 20 Curated Real Gunshot Files
GUNSHOT_FILENAMES = [
    "ak47_004.wav", "SMG3.wav", "de_001.wav", "de_003.wav",
    "magnum_001.wav", "magnum_003.wav", "magnum_004.wav", "magnum_005.wav",
    "rifle001.wav", "rifle002.wav", "rifle003.wav", "rifle004.wav", "rifle005.wav", "rifle006.wav",
    "sniper_001.wav", "sniper_002.wav", "sniper_003.wav", "sniper_004.wav", "sniper_005.wav",
    "blast.wav"
]

def download_file(url, dest_path):
    """Download single audio file."""
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req, timeout=20) as response, open(dest_path, 'wb') as out_file:
        out_file.write(response.read())

def main():
    print("=" * 80)
    print("🎯 SHOOT_CATCHER — AUTOMATIC AUDIO TEST DATA DOWNLOADER")
    print("=" * 80)
    print(f"Target Base Folder: {BASE_TEST_DIR}")
    print(f" ├─ [Actual_Gunshots] : {DIR_ACTUAL}")
    print(f" ├─ [Like_Gunshots]   : {DIR_LIKE}")
    print(f" └─ [Not_Gunshots]    : {DIR_NOT}")
    print("=" * 80)

    start_time = time.time()
    total_downloaded = 0

    # -------------------------------------------------------------
    # 1. DOWNLOAD ACTUAL GUNSHOTS (20 files)
    # -------------------------------------------------------------
    print("\n🔫 [1/3] Downloading 20 Real Firearm Recordings into 'Actual_Gunshots'...")
    for idx, fname in enumerate(GUNSHOT_FILENAMES, 1):
        dest_file = DIR_ACTUAL / fname
        if dest_file.exists() and dest_file.stat().st_size > 1000:
            print(f"   [{idx:>2}/20] ⏩ Already exists: {fname}")
            total_downloaded += 1
            continue

        url = FIREARMS_BASE_URL + fname
        try:
            download_file(url, dest_file)
            size_kb = dest_file.stat().st_size / 1024
            print(f"   [{idx:>2}/20] ✅ Downloaded: {fname:<22} ({size_kb:.1f} KB)")
            total_downloaded += 1
        except Exception as e:
            print(f"   [{idx:>2}/20] ❌ Error downloading {fname}: {e}")

    # -------------------------------------------------------------
    # FETCH ESC-50 METADATA FOR IMPOSTERS AND AMBIENT
    # -------------------------------------------------------------
    print("\n📥 Fetching ESC-50 metadata for Imposters and Ambient sounds...")
    try:
        req = urllib.request.Request(ESC50_META_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            lines = [line.decode('utf-8') for line in resp.readlines()]
        reader = csv.DictReader(lines)
        all_esc_rows = list(reader)
        print(f"✅ Metadata retrieved successfully ({len(all_esc_rows)} records cataloged).")
    except Exception as e:
        print(f"❌ Could not fetch ESC-50 metadata: {e}")
        all_esc_rows = []

    # -------------------------------------------------------------
    # 2. DOWNLOAD LIKE_GUNSHOTS (20 files)
    # -------------------------------------------------------------
    print("\n💥 [2/3] Downloading 20 Imposter Sounds (Fireworks, Claps, Knocks) into 'Like_Gunshots'...")
    imposter_categories = ["fireworks", "clapping", "door_wood_knock", "glass_breaking"]
    imposter_rows = [r for r in all_esc_rows if r["category"] in imposter_categories][:20]

    for idx, row in enumerate(imposter_rows, 1):
        fname = row["filename"]
        cat = row["category"]
        dest_file = DIR_LIKE / f"{cat}_{idx:02d}_{fname}"
        if dest_file.exists() and dest_file.stat().st_size > 1000:
            print(f"   [{idx:>2}/20] ⏩ Already exists: {dest_file.name}")
            total_downloaded += 1
            continue

        url = ESC50_AUDIO_BASE + fname
        try:
            download_file(url, dest_file)
            size_kb = dest_file.stat().st_size / 1024
            print(f"   [{idx:>2}/20] ✅ Downloaded: {dest_file.name:<32} ({size_kb:.1f} KB)")
            total_downloaded += 1
        except Exception as e:
            print(f"   [{idx:>2}/20] ❌ Error downloading {fname}: {e}")

    # -------------------------------------------------------------
    # 3. DOWNLOAD NOT_GUNSHOTS (20 files)
    # -------------------------------------------------------------
    print("\n🌿 [3/3] Downloading 20 Ambient Sounds (Rain, Engines, Dogs, Sirens) into 'Not_Gunshots'...")
    ambient_categories = ["rain", "engine", "dog", "siren", "sneezing"]
    ambient_rows = [r for r in all_esc_rows if r["category"] in ambient_categories][:20]

    for idx, row in enumerate(ambient_rows, 1):
        fname = row["filename"]
        cat = row["category"]
        dest_file = DIR_NOT / f"{cat}_{idx:02d}_{fname}"
        if dest_file.exists() and dest_file.stat().st_size > 1000:
            print(f"   [{idx:>2}/20] ⏩ Already exists: {dest_file.name}")
            total_downloaded += 1
            continue

        url = ESC50_AUDIO_BASE + fname
        try:
            download_file(url, dest_file)
            size_kb = dest_file.stat().st_size / 1024
            print(f"   [{idx:>2}/20] ✅ Downloaded: {dest_file.name:<32} ({size_kb:.1f} KB)")
            total_downloaded += 1
        except Exception as e:
            print(f"   [{idx:>2}/20] ❌ Error downloading {fname}: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 ALL DOWNLOADS COMPLETE! Total Files: {total_downloaded} ({elapsed:.1f} seconds)")
    print("=" * 80)
    print(f"Summary of Your Folders:")
    print(f" ├─ [Actual_Gunshots] : {len(list(DIR_ACTUAL.glob('*.wav')))} WAV files")
    print(f" ├─ [Like_Gunshots]   : {len(list(DIR_LIKE.glob('*.wav')))} WAV files")
    print(f" └─ [Not_Gunshots]    : {len(list(DIR_NOT.glob('*.wav')))} WAV files")
    print("=" * 80)

if __name__ == "__main__":
    main()
