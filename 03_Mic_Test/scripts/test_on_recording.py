"""
===============================================================================
🎵 Shoot_Catcher — Multi-Model Audio File Tester & Benchmark
===============================================================================
Slides a window across any .wav audio recording file and evaluates predictions
side-by-side across all active trained models (1D, 2D, Enhanced).
===============================================================================
"""

import sys
import os
from pathlib import Path

# Add current directory to path
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import live_demo

def main():
    print("\n" + "=" * 80)
    print("🎵 SHOOT_CATCHER — AUDIO RECORDING BENCHMARK TESTER")
    print("=" * 80)
    
    manager = live_demo.ModelManager()
    manager.print_audit_table()
    
    if not manager.trained_models:
        print("❌ No trained models found. Please train models in 01_1D_CNN or 02_2D_CNN first.")
        return
        
    live_demo.run_file_benchmark(manager.trained_models)

if __name__ == "__main__":
    main()
