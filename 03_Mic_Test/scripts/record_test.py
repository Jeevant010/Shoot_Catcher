"""
===============================================================================
🎙️ Shoot_Catcher — Microphone Recording & Verification Tool
===============================================================================
Records a test audio snippet from your microphone, performs volume analysis,
saves 'test_recording.wav', and optionally runs all 5 models on the recording.
===============================================================================
"""

import sys
import os
from pathlib import Path
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("❌ sounddevice is required. Run: pip install sounddevice")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("❌ soundfile is required. Run: pip install soundfile")
    sys.exit(1)

# Add current directory to path
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def find_working_mic():
    devices = sd.query_devices()
    valid_mics = []
    seen_names = set()

    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            name = d['name'].strip()
            clean_name = name.split("(")[0].strip()
            if clean_name not in seen_names:
                seen_names.add(clean_name)
                try:
                    native_sr = int(d['default_samplerate'])
                    channels = min(d['max_input_channels'], 2)
                    test = sd.rec(int(0.1 * native_sr), samplerate=native_sr, channels=channels, device=i, dtype='float32')
                    sd.wait()
                    valid_mics.append((i, name, native_sr, channels))
                except Exception:
                    pass
    return valid_mics


def main():
    print("\n" + "=" * 80)
    print("🎙️ SHOOT_CATCHER — MICROPHONE RECORD & TEST TOOL")
    print("=" * 80)

    print("🔍 Scanning for microphones...")
    valid_mics = find_working_mic()

    if not valid_mics:
        print("❌ Could not find a working microphone.")
        sys.exit(1)

    print("\n🎤 Available Microphones:")
    for idx, (dev_id, name, sr, ch) in enumerate(valid_mics, 1):
        print(f" [{idx}] {name} ({sr}Hz, {ch}ch)")

    selected = valid_mics[0]
    if len(valid_mics) > 1:
        choice = input(f"\n👉 Select microphone (1-{len(valid_mics)}, default 1): ").strip()
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(valid_mics):
                selected = valid_mics[idx]

    device_id, device_name, native_sr, channels = selected
    print(f"\n✅ Using Device [{device_id}]: {device_name}")

    duration_str = input("👉 Enter recording duration in seconds (default: 5): ").strip()
    try:
        duration = float(duration_str) if duration_str else 5.0
    except ValueError:
        duration = 5.0

    print(f"\n🎤 RECORDING {duration:.1f} SECONDS... (Play gunshot / audio now!)")

    try:
        recording = sd.rec(int(duration * native_sr), samplerate=native_sr, channels=channels, device=device_id, dtype='float32')
        sd.wait()
        print("✅ Recording finished!")

        # Save to file in parent directory
        filename = SCRIPT_DIR.parent / "test_recording.wav"
        sf.write(str(filename), recording, native_sr)

        # Quick volume check
        mono = recording.mean(axis=1) if recording.ndim > 1 else recording.flatten()
        peak_volume = np.max(np.abs(mono))
        rms_volume = np.sqrt(np.mean(mono ** 2))

        print(f"\n📊 Audio Analysis:")
        print(f"   Peak Amplitude: {peak_volume:.5f}")
        print(f"   RMS Energy    : {rms_volume:.5f}")

        if peak_volume < 0.005:
            print("   ⚠️ WARNING: The recording is extremely quiet or silent.")
            print("      Check your Windows microphone volume and permissions.")
        else:
            print("   🔊 Good audio level detected!")

        print(f"\n💾 Saved to: {filename}")

        # Prompt to run 5-model benchmark
        run_now = input("\n👉 Test all 5 models on this recording now? (Y/n): ").strip().lower()
        if run_now != 'n':
            import live_demo
            manager = live_demo.ModelManager()
            manager.print_audit_table()
            if manager.trained_models:
                live_demo.run_file_benchmark(manager.trained_models, wav_path=filename)

    except Exception as e:
        print(f"❌ Error during recording: {e}")


if __name__ == "__main__":
    main()
