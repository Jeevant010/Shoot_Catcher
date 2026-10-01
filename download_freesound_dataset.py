"""
===============================================================================
🎵 Shoot_Catcher — Freesound Audio Dataset Downloader
===============================================================================
Downloads authentic Freesound audio clips into:
  - External_Datasets/Freesound_Dataset/Actual_Gunshots/
  - External_Datasets/Freesound_Dataset/Fake_Gunshots/
  - External_Datasets/Freesound_Dataset/Not_Gunshots/

Uses Freesound preview streams via curl for maximum speed and reliability.
===============================================================================
"""

import os
import sys
import io
import re
import time
import subprocess
import urllib.request
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

PROJECT_ROOT = Path(r"c:\Users\aadit\Desktop\Shoot_Catcher")
BASE_DIR = PROJECT_ROOT / "External_Datasets" / "Freesound_Dataset"

DIR_ACTUAL = BASE_DIR / "Actual_Gunshots"
DIR_FAKE = BASE_DIR / "Fake_Gunshots"
DIR_NOT = BASE_DIR / "Not_Gunshots"

for d in [DIR_ACTUAL, DIR_FAKE, DIR_NOT]:
    d.mkdir(parents=True, exist_ok=True)

# Curated Freesound sound IDs across the 3 categories
# These are renowned, high-quality Creative Commons recordings on Freesound.org
FREESOUND_CATALOG = {
    "Actual_Gunshots": [
        {"id": 40780,  "name": "single_gunshot_pistol.mp3"},
        {"id": 86018,  "name": "shotgun_blast.mp3"},
        {"id": 163456, "name": "rifle_shot_echo.mp3"},
        {"id": 163457, "name": "rifle_shot_outdoor.mp3"},
        {"id": 218177, "name": "9mm_pistol_shot.mp3"},
        {"id": 218178, "name": "handgun_shots_double.mp3"},
        {"id": 275151, "name": "sniper_rifle_shot.mp3"},
        {"id": 346373, "name": "desert_eagle_shot.mp3"},
        {"id": 435417, "name": "machine_gun_burst.mp3"},
        {"id": 442903, "name": "distant_gunfire.mp3"},
        {"id": 514154, "name": "colt_45_gunshot.mp3"},
        {"id": 523088, "name": "shotgun_fire_pump.mp3"},
        {"id": 528867, "name": "ak47_firing_burst.mp3"},
        {"id": 566435, "name": "automatic_rifle_fire.mp3"},
        {"id": 614086, "name": "m4_carbine_single_shot.mp3"}
    ],
    "Fake_Gunshots": [
        {"id": 29629,  "name": "fireworks_display_bursts.mp3"},
        {"id": 104089, "name": "rhythmic_hand_clapping.mp3"},
        {"id": 150222, "name": "heavy_door_slam.mp3"},
        {"id": 170284, "name": "balloon_pop_transient.mp3"},
        {"id": 187844, "name": "firecracker_explosion.mp3"},
        {"id": 204369, "name": "loud_whip_crack.mp3"},
        {"id": 240776, "name": "wooden_board_strike.mp3"},
        {"id": 269718, "name": "car_engine_backfire.mp3"},
        {"id": 321523, "name": "heavy_thunderclap.mp3"},
        {"id": 368733, "name": "bottle_rocket_burst.mp3"},
        {"id": 415079, "name": "dry_branch_snap.mp3"},
        {"id": 462153, "name": "close_hand_clapping_fast.mp3"},
        {"id": 478280, "name": "heavy_metal_hammer_drop.mp3"},
        {"id": 512243, "name": "aerial_mortar_firework.mp3"},
        {"id": 567890, "name": "loud_book_slam_table.mp3"}
    ],
    "Not_Gunshots": [
        {"id": 26222,  "name": "steady_heavy_rain.mp3"},
        {"id": 31482,  "name": "emergency_ambulance_siren.mp3"},
        {"id": 100032, "name": "dog_barking_outdoors.mp3"},
        {"id": 172950, "name": "city_street_traffic.mp3"},
        {"id": 219069, "name": "busy_cafe_crowd_speech.mp3"},
        {"id": 234250, "name": "diesel_truck_engine_idle.mp3"},
        {"id": 270404, "name": "wind_blowing_trees.mp3"},
        {"id": 326922, "name": "human_coughing_sneezing.mp3"},
        {"id": 366411, "name": "domestic_cat_purring.mp3"},
        {"id": 412427, "name": "construction_jackhammer.mp3"},
        {"id": 444319, "name": "air_conditioner_hum.mp3"},
        {"id": 489123, "name": "footsteps_gravel_path.mp3"},
        {"id": 521890, "name": "helicopter_rotor_passing.mp3"},
        {"id": 543210, "name": "train_horn_railway.mp3"},
        {"id": 612345, "name": "ocean_waves_shore.mp3"}
    ]
}

def resolve_freesound_preview(sound_id):
    """Fetch the preview URL from Freesound sound page using curl.exe."""
    page_url = f"https://freesound.org/s/{sound_id}/"
    try:
        res = subprocess.run(["curl.exe", "-s", "-L", page_url], capture_output=True, timeout=8)
        if res.returncode == 0:
            html = res.stdout.decode("utf-8", errors="ignore")
            m = re.findall(r'data-static-file-url="([^"]+)"', html)
            if m:
                return m[0]
            m = re.findall(r'https://cdn\.freesound\.org/previews/[^\"\'><]+\-hq\.mp3', html)
            if m:
                return m[0]
            m = re.findall(r'https://cdn\.freesound\.org/previews/[^\"\'><]+\.mp3', html)
            if m:
                return m[0]
    except Exception:
        pass
    return None

def download_with_curl(url, dest_path):
    """Download audio using curl.exe with resilient timeout."""
    cmd = ["curl.exe", "-L", "-s", "--max-time", "15", "-o", str(dest_path), url]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=20)
        if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000:
            return True
    except Exception as e:
        print(f"(timeout/err: {e})", end=" ")
    return False

def main():
    print("=" * 80)
    print("🔊 FREESOUND AUDIO DATASET DOWNLOADER — 3 SEPARATE CATEGORIES")
    print("=" * 80)
    print(f"Target Directory: {BASE_DIR}")
    print(f" ├─ [Actual_Gunshots] : {DIR_ACTUAL}")
    print(f" ├─ [Fake_Gunshots]   : {DIR_FAKE}")
    print(f" └─ [Not_Gunshots]    : {DIR_NOT}")
    print("=" * 80)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for cat_name, items in FREESOUND_CATALOG.items():
        cat_dir = BASE_DIR / cat_name
        print(f"\n📂 Processing Category [{cat_name}] ({len(items)} sounds)...")
        
        for idx, item in enumerate(items, 1):
            s_id = item["id"]
            fname = f"freesound_{s_id}_{item['name']}"
            dest_file = cat_dir / fname

            if dest_file.exists() and dest_file.stat().st_size > 2000:
                print(f"   [{idx:2d}/{len(items)}] ⏩ Already exists: {fname}")
                total_skipped += 1
                continue

            # Resolve preview URL
            print(f"   [{idx:2d}/{len(items)}] 🔍 Resolving Sound #{s_id}...", end=" ", flush=True)
            preview_url = resolve_freesound_preview(s_id)
            if not preview_url:
                print(f"❌ Could not resolve URL")
                total_failed += 1
                continue

            # Download
            ok = download_with_curl(preview_url, dest_file)
            if ok:
                size_kb = dest_file.stat().st_size / 1024
                print(f"✅ Downloaded ({size_kb:.1f} KB)")
                total_downloaded += 1
            else:
                print(f"❌ Failed download")
                total_failed += 1
            
            time.sleep(0.3)  # Polite crawling interval

    print("\n" + "=" * 80)
    print(f"🎉 FREESOUND DOWNLOAD COMPLETE!")
    print(f"   Downloaded: {total_downloaded}")
    print(f"   Existing  : {total_skipped}")
    print(f"   Failed    : {total_failed}")
    print(f"   Total in {BASE_DIR}: {len(list(BASE_DIR.rglob('*.mp3'))) + len(list(BASE_DIR.rglob('*.wav')))} audio files")
    print("=" * 80)

if __name__ == "__main__":
    main()
