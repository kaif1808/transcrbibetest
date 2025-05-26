import torchaudio
import platform # To help identify OS

print(f"Attempting to load audio with torchaudio.")
print(f"Python version: {platform.python_version()}")
print(f"Torchaudio version: {torchaudio.__version__}")

audio_file = "IsabelleAudio.wav"

try:
    waveform, sample_rate = torchaudio.load(audio_file)
    print(f"\nSuccessfully loaded '{audio_file}' with torchaudio!")
    print(f"Waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate} Hz")
except Exception as e:
    print(f"\nError loading '{audio_file}' with torchaudio: {e}")
    print("\nTroubleshooting suggestions:")
    print("1. Ensure FFmpeg is correctly installed AND its 'bin' directory is in your system's PATH environment variable.")
    print("2. Restart your terminal/IDE after modifying the PATH.")
    print("3. Test FFmpeg by opening a terminal and typing 'ffmpeg -version'. It should display version info.")
    print(f"4. Check if the file '{audio_file}' exists in the same directory as this script or provide the full path.")
    print(f"5. Ensure the file '{audio_file}' is not corrupted and can be played by a media player.")