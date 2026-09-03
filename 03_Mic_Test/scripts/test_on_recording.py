"""
===============================================================================
🎵 Shoot_Catcher — 5-Model Audio File Benchmark & Tester
===============================================================================
Slides a window across any .wav audio recording file and evaluates predictions
side-by-side across all 5 models:
  1. Baseline 1D CNN
  2. Baseline 2D CNN (Mel Spectrogram)
  3. Robust CRNN (PCEN)
  4. Enhanced 1D CNN (Dual-Head)
  5. Enhanced 2D CNN (Dual-Head)
===============================================================================
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import live_demo

def main():
    parser = argparse.ArgumentParser(description="Shoot_Catcher — 5-Model Audio File Benchmark")
    parser.add_argument("--input", type=str, default=None, help="Path to .wav file to test")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold (default: 0.50)")
    args = parser.parse_args()

    print("\n" + "=" * 85)
    print("🎵 SHOOT_CATCHER — 5-MODEL AUDIO RECORDING BENCHMARK TESTER")
    print("=" * 85)
    
    live_demo.CONFIDENCE_THRESHOLD = args.threshold

    manager = live_demo.ModelManager()
    manager.print_audit_table()
    
    if not manager.trained_models:
        print("❌ No trained models found. Please train models first.")
        return
        
    wav_path = None
    if args.input:
        wav_path = Path(args.input)
        if not wav_path.exists():
            print(f"❌ Specified file not found: {wav_path}")
            return

    live_demo.run_file_benchmark(manager.trained_models, wav_path=wav_path)

if __name__ == "__main__":
    main()
