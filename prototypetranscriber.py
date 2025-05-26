import whisper
import torch
import time
from pathlib import Path
from whisper.utils import get_writer
from tqdm import tqdm
import sys


class WhisperProgress(tqdm):
    """Custom progress bar that integrates with Whisper's internal logging"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Initial value

    def update(self, n=1):
        super().update(n)
        self._current += n
        elapsed = time.time() - self.start_t
        avg_time = elapsed / self._current
        remaining = avg_time * (self.total - self._current)
        self.set_postfix_str(f"ETA: {remaining:.0f}s | Speed: {self.format_dict['rate']:.1f} it/s")


# Monkey-patch Whisper's tqdm implementation
import whisper.transcribe

whisper.transcribe.tqdm = WhisperProgress


def transcribe_audio_file():
    """
    Professional transcription script with enhanced progress tracking
    for M2 Max GPU acceleration[1][3]
    """
    audio_file = "IsabelleAudio.mp4"
    model_size = "large"
    language = "en"
    output_dir = "./output/"

    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: File '{audio_file}' not found")
        return

    print(f"Starting transcription of 1-hour audio file...")
    start_time = time.time()

    # Use verbose=True for built-in progress updates
    result = model.transcribe(
        audio=str(audio_path),
        language=language,
        verbose=True,  # Enables Whisper's native progress bar[2][4]
        fp16=False  # Required for MPS compatibility[1][3]
    )

    print(f"\nTranscription completed in {time.time() - start_time:.2f}s")

    # Save academic-grade outputs
    Path(output_dir).mkdir(exist_ok=True)
    save_transcription_files(result, output_dir)
    print(f"\nOutput saved to: {output_dir}")


def save_transcription_files(result, output_dir):
    """Save results in formats suitable for economic research[1][4]"""
    formats = ['txt', 'srt', 'tsv', 'json']
    options = {
        'max_line_width': 80,
        'max_line_count': 2,
        'highlight_words': False
    }

    for fmt in formats:
        try:
            writer = get_writer(fmt, output_dir)
            writer(result, 'IsabelleAudio_transcription', options)
            print(f"✓ {fmt.upper()} file created")
        except Exception as e:
            print(f"✗ Error creating {fmt}: {str(e)}")


if __name__ == "__main__":
    print("=" * 50)
    print("Mekong Economics Transcription System")
    print("=" * 50)

    if torch.backends.mps.is_available():
        print("✓ Apple Silicon GPU acceleration enabled")
    else:
        print("✗ Using CPU processing")

    transcribe_audio_file()
