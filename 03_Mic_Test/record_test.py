import sounddevice as sd
import soundfile as sf
import sys
import numpy as np

def find_working_mic():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            try:
                native_sr = int(d['default_samplerate'])
                channels = min(d['max_input_channels'], 2)
                test = sd.rec(int(0.1 * native_sr), samplerate=native_sr, channels=channels, device=i, dtype='float32')
                sd.wait()
                return i, native_sr, channels
            except:
                continue
    return None, None, None

print("🔍 Finding microphone...")
device_id, native_sr, channels = find_working_mic()

if device_id is None:
    print("❌ Could not find a working microphone.")
    sys.exit(1)

device_name = sd.query_devices()[device_id]['name']
print(f"✅ Found Device: [{device_id}] {device_name}")

duration = 5  # seconds
print(f"\n🎤 RECORDING {duration} SECONDS... (Play your gunshot now!)")

try:
    recording = sd.rec(int(duration * native_sr), samplerate=native_sr, channels=channels, device=device_id, dtype='float32')
    sd.wait()
    print("✅ Recording finished!")
    
    # Save to file
    filename = "test_recording.wav"
    sf.write(filename, recording, native_sr)
    
    # Quick volume check
    peak_volume = np.max(np.abs(recording))
    print(f"\n📊 Volume Analysis:")
    if peak_volume < 0.01:
        print(f"   ⚠️ WARNING: The recording is almost completely SILENT (Peak: {peak_volume:.4f}).")
        print("      Your microphone might be muted in Windows, or it's selecting the wrong mic.")
    else:
        print(f"   🔊 Audio detected! (Peak: {peak_volume:.4f})")
    
    print(f"\n💾 Saved to: {filename}")
    print("👉 Please open this folder in Windows File Explorer and play 'test_recording.wav' to hear what the script actually heard.")

except Exception as e:
    print(f"❌ Error during recording: {e}")
