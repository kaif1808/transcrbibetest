import whisper
import torch
import time
from pathlib import Path
from whisper.utils import get_writer
from tqdm import tqdm
import sys

# --- pyannote.audio imports ---
from pyannote.audio import Pipeline
import os  # For Hugging Face Token if passed directly

print("--- diarisation.py script execution started (top of file) ---")  # DIAGNOSTIC PRINT


# Helper function to format seconds into HH:MM:SS or MM:SS
def format_time_display(seconds):
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"
    elif minutes > 0:
        return f"{minutes:02d}m:{secs:02d}s"
    else:
        return f"{secs:02d}s"


class WhisperProgress(tqdm):
    """
    Custom progress bar for Whisper transcription.
    Leverages tqdm to show percentage, time taken, ETA, and speed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()  # Record start time for this progress bar

    def update(self, n=1):  # n is the increment for this update step
        super().update(n)  # This updates tqdm's internal state (self.n, self.total, etc.)

        elapsed_seconds = time.time() - self.start_time

        postfix_parts = []
        postfix_parts.append(f"Time: {format_time_display(elapsed_seconds)}")

        if self.total is not None and self.n > 0:
            if 'remaining' in self.format_dict and isinstance(self.format_dict['remaining'], (int, float)):
                postfix_parts.append(f"ETA: {format_time_display(self.format_dict['remaining'])}")

        if 'rate' in self.format_dict and isinstance(self.format_dict['rate'], (int, float)):
            postfix_parts.append(f"Speed: {self.format_dict['rate']:.1f}it/s")

        if postfix_parts:
            self.set_postfix_str(", ".join(postfix_parts), refresh=True)


# Monkey-patch Whisper's tqdm implementation
import whisper.transcribe

whisper.transcribe.tqdm = WhisperProgress


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    """Formats a timestamp in seconds to HH:MM:SS.mmm or MM:SS.mmm string for SRT/VTT."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds_val = milliseconds // 1_000
    milliseconds %= 1_000
    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds_val:02d}{decimal_marker}{milliseconds:03d}"
    else:
        return f"{minutes:02d}:{seconds_val:02d}{decimal_marker}{milliseconds:03d}"


def assign_speakers_to_segments(transcription_segments, speaker_timeline):
    """
    Assigns speaker labels to transcription segments based on a speaker timeline.
    """
    for segment in transcription_segments:
        segment_midpoint = segment['start'] + (segment['end'] - segment['start']) / 2
        assigned_speaker = "Unknown_Speaker"
        for speaker_label, start_time, end_time in speaker_timeline:
            if start_time <= segment_midpoint <= end_time:
                assigned_speaker = speaker_label
                break
        segment['speaker'] = assigned_speaker
    return transcription_segments


def transcribe_audio_file():
    """
    Main function to perform speaker diarization and audio transcription.
    Includes progress readouts and MPS/GPU checks with explicit MPS attempt for pyannote.
    Whisper is forced to CPU for troubleshooting MPS issues.
    """
    print("--- Entered transcribe_audio_file() function ---")  # DIAGNOSTIC PRINT
    # --- Configuration ---
    audio_file = "IsabelleAudio.wav"  # Using the trimmed test file
    model_size = "turbo"  # Using "base" model as per previous step
    language = "en"
    output_dir = "./output/"
    hf_token = None  # Optional: Your Hugging Face User Access Token string

    overall_start_time = time.time()

    # --- Initial System Checks ---
    print("=" * 60)
    print("Transcription & Diarization System Initializing")
    print("=" * 60)

    device_info = "CPU for PyTorch operations."
    mps_available_flag = torch.backends.mps.is_available()
    cuda_available_flag = torch.cuda.is_available()

    if cuda_available_flag:
        device_info = "NVIDIA GPU (CUDA) acceleration available for PyTorch."
    elif mps_available_flag:
        device_info = "Apple Silicon GPU (MPS) acceleration available for PyTorch."
    print(f"✓ System Check: {device_info}")

    # --- Step 1: Speaker Diarization ---
    print("\n--- Step 1 of 2: Speaker Diarization ---")
    diarization_step_start_time = time.time()
    speaker_timeline = []
    diarization_pipeline = None

    try:
        print("Loading diarization pipeline (pyannote.audio)...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token if hf_token else True
        )

        if mps_available_flag:
            print("MPS is available. Attempting to move pyannote pipeline to MPS device...")
            try:
                diarization_pipeline.to(torch.device("mps"))
                print("✓ Pyannote pipeline explicitly set to MPS device.")
            except Exception as e:
                print(f"Warning: Could not explicitly move pyannote pipeline to MPS: {e}")
                print("Proceeding with default device placement for pyannote (may use CPU or some MPS operations).")
        elif cuda_available_flag:
            print("CUDA is available. Attempting to move pyannote pipeline to CUDA device...")
            try:
                diarization_pipeline.to(torch.device("cuda"))
                print("✓ Pyannote pipeline explicitly set to CUDA device.")
            except Exception as e:
                print(f"Warning: Could not explicitly move pyannote pipeline to CUDA: {e}")
                print("Proceeding with default device placement for pyannote (likely CPU).")
        else:
            print("No CUDA or MPS detected by PyTorch for explicit pipeline placement. Pyannote will use CPU.")

        print(f"Running diarization on: {audio_file} (this may take a while)...")
        audio_path_obj_diar = Path(audio_file)  # Use a different variable name
        if not audio_path_obj_diar.exists():
            print(
                f"Critical Error: Audio file '{audio_file}' not found for diarization at path: {audio_path_obj_diar.resolve()}")
            sys.exit(1)

        diarization_result = diarization_pipeline(str(audio_path_obj_diar), num_speakers=None)  # Pass path as string

        if diarization_result and hasattr(diarization_result, 'itertracks'):
            raw_labels = sorted(list(set([
                speaker for _, _, speaker in diarization_result.itertracks(yield_label=True)
            ])))
            speaker_mapping = {label: f"Speaker {i + 1}" for i, label in enumerate(raw_labels)}

            for turn, _, raw_speaker_label in diarization_result.itertracks(yield_label=True):
                mapped_speaker = speaker_mapping.get(raw_speaker_label, raw_speaker_label)
                speaker_timeline.append((mapped_speaker, turn.start, turn.end))

            print(f"Diarization analysis complete. Found {len(speaker_mapping)} distinct speaker(s).")
        else:
            print("Warning: Diarization did not produce speaker tracks or a valid result.")

    except Exception as e:
        print(f"Critical Error during speaker diarization: {e}")
        print("Troubleshooting suggestions:")
        print(" - Ensure 'pyannote.audio' and its dependencies are correctly installed.")
        print(
            " - Ensure FFmpeg is installed and accessible in your system's PATH (if processing non-WAV or problematic WAV files).")
        print(" - Check Hugging Face token and model card agreements for 'pyannote/speaker-diarization-3.1'.")
        print("Exiting script due to diarization failure.")
        sys.exit(1)
    finally:
        diarization_elapsed_time = time.time() - diarization_step_start_time
        print(f"Speaker Diarization (Step 1) processing time: {format_time_display(diarization_elapsed_time)}.")

    if not speaker_timeline:
        print("Critical Error: Speaker timeline is empty after diarization.")
        print("This could be due to silent audio, very short audio, or an issue in the diarization process.")
        print("Exiting script as a speaker-diarized output cannot be effectively produced.")
        sys.exit(1)

    # --- Step 2: Audio Transcription ---
    print("\n--- Step 2 of 2: Audio Transcription ---")
    transcription_step_start_time = time.time()
    whisper_model = None

    try:
        print(f"Loading Whisper model ('{model_size}')...")

        # --- MODIFICATION: Force Whisper to CPU for debugging MPS issue ---
        target_whisper_device = "cpu"
        print(f"INFO: Forcing Whisper model to load on '{target_whisper_device}' device for troubleshooting.")
        # --- END MODIFICATION ---

        whisper_model = whisper.load_model(model_size, device=target_whisper_device)

        if whisper_model:
            try:
                model_device = str(next(whisper_model.parameters()).device)
                print(f"Whisper model loaded on device: {model_device}")
                if "mps" in model_device:
                    print("✓ Whisper is utilizing MPS for acceleration.")
                elif "cuda" in model_device:
                    print("✓ Whisper is utilizing CUDA for acceleration.")
                elif "cpu" in model_device:
                    print(f"✓ Whisper is running on {model_device} as intended for this test.")
            except Exception as e:
                print(f"Could not determine Whisper model device: {e}")
        else:
            print("Critical Error: Whisper model could not be loaded.")
            sys.exit(1)

        audio_path_obj_trans = Path(audio_file)  # Use a different variable name
        if not audio_path_obj_trans.exists():
            print(
                f"Critical Error: Audio file '{audio_file}' not found for transcription at path: {audio_path_obj_trans.resolve()}")
            sys.exit(1)

        print(f"Starting transcription of: {audio_file}")
        result = whisper_model.transcribe(
            str(audio_path_obj_trans),  # Pass path as string
            language=language,
            verbose=False,
            fp16=False
        )

    except Exception as e:
        print(f"Critical Error during Whisper transcription or model loading: {e}")
        print("Exiting script due to transcription failure.")
        sys.exit(1)
    finally:
        transcription_elapsed_time = time.time() - transcription_step_start_time
        print(f"\nTranscription (Step 2) processing time: {format_time_display(transcription_elapsed_time)}.")

    # --- Post-processing and Saving ---
    print("\n--- Post-processing and Saving Results ---")
    # Ensure 'result' is defined; it would be if try block for transcription completed.
    # If an error happened before 'result' was assigned, this part might not be reached due to sys.exit().
    if 'result' not in locals() or result is None:
        print("Error: Transcription result is missing. Cannot save output.")
        # This case should ideally be prevented by sys.exit() in the try-except block,
        # but adding a check here for robustness before accessing result["segments"] or result["text"]
    elif "segments" in result and result["segments"]:
        print("Assigning speaker labels to transcript segments...")
        result["segments"] = assign_speakers_to_segments(result["segments"], speaker_timeline)
    else:
        print("No text segments found in transcription result. Output might be minimal.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if 'result' in locals() and result is not None:  # Only save if result exists
        save_transcription_files(result, Path(audio_file).stem, output_dir)

    overall_elapsed_time = time.time() - overall_start_time
    print(f"\n--- All processing finished in {format_time_display(overall_elapsed_time)} ---")
    print(f"Output files saved to: {output_dir}")


def save_transcription_files(result, base_filename, output_dir):
    """Saves transcription results in various formats."""
    output_path = Path(output_dir)

    txt_file_path = output_path / f"{base_filename}_diarized_transcript.txt"
    try:
        with open(txt_file_path, "w", encoding="utf-8") as f:
            if "segments" in result and result["segments"]:
                for segment in result["segments"]:
                    speaker = segment.get("speaker", "UnknownSpeaker")
                    text = segment.get("text", "").strip()
                    start_time_str = format_timestamp(segment['start'])
                    end_time_str = format_timestamp(segment['end'])
                    f.write(f"[{start_time_str} -> {end_time_str}] {speaker}: {text}\n")
            elif result.get("text"):
                f.write(f"Full Text (no segmentation/diarization):\n{result['text']}\n")
            else:
                f.write("No transcription content available.\n")
        print(f"✓ Custom TXT (diarized) saved: {txt_file_path}")
    except Exception as e:
        print(f"✗ Error saving custom TXT file: {e}")

    plain_txt_path = output_path / f"{base_filename}_plain_transcript.txt"
    try:
        with open(plain_txt_path, "w", encoding="utf-8") as f:
            f.write(result.get("text", "No plain text transcription available.\n"))
        print(f"✓ Plain TXT (Whisper) saved: {plain_txt_path}")
    except Exception as e:
        print(f"✗ Error saving plain TXT file: {e}")

    standard_formats = ['srt', 'tsv', 'json']
    options = {
        'max_line_width': 80,
        'max_line_count': 2,
        'highlight_words': False
    }
    for fmt in standard_formats:
        writer = get_writer(fmt, str(output_path))
        try:
            writer(result, f"{base_filename}_standard_transcript", options)
            print(f"✓ {fmt.upper()} (standard) saved: {output_path / f'{base_filename}_standard_transcript.{fmt}'}")
        except Exception as e:
            print(f"✗ Error saving {fmt.upper()} file: {e}")


if __name__ == "__main__":
    print("--- In __main__ block, preparing to call transcribe_audio_file() ---")  # DIAGNOSTIC PRINT
    transcribe_audio_file()
    print("--- transcribe_audio_file() call completed ---")  # DIAGNOSTIC PRINT
    print("--- diarisation.py script execution finished (end of file) ---")  # DIAGNOSTIC PRINT