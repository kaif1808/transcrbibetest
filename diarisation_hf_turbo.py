import torch
import time
from pathlib import Path
import sys
import math
from pyannote.audio import Pipeline
import os

# --- Hugging Face Transformers imports ---
from transformers import pipeline

print("--- diarisation_hf_turbo.py script execution started (top of file) ---")


# Helper function to format seconds into HH:MM:SS or MM:SS
def format_time_display(seconds):
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
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


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    """Formats a timestamp in seconds to HH:MM:SS.mmm or MM:SS.mmm string for file output."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "00:00:00" + decimal_marker + "000"
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = int(milliseconds // 3_600_000);
    milliseconds %= 3_600_000
    minutes = int(milliseconds // 60_000);
    milliseconds %= 60_000
    seconds_val = int(milliseconds // 1_000);
    milliseconds %= 1_000
    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds_val:02d}{decimal_marker}{milliseconds:03d}"
    else:
        return f"{minutes:02d}:{seconds_val:02d}{decimal_marker}{milliseconds:03d}"


def format_srt_timestamp(seconds: float):
    """Formats a timestamp in seconds to HH:MM:SS,mmm string for SRT."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "00:00:00,000"
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = int(milliseconds // 3_600_000);
    milliseconds %= 3_600_000
    minutes = int(milliseconds // 60_000);
    milliseconds %= 60_000
    secs = int(milliseconds // 1_000);
    milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def assign_speakers_to_segments(transcription_segments, speaker_timeline):
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
    print("--- Entered transcribe_audio_file() function ---")
    # --- Configuration ---
    audio_file_path_str = "IsabelleAudio.wav"  # Your trimmed audio file

    # Using a "turbo" model from Hugging Face Hub
    whisper_model_hf_id = "openai/whisper-large-v3-turbo"  # Target "turbo" model
    # You can also try other "turbo" variants if you find them, or smaller standard models
    # e.g., "openai/whisper-base", "openai/whisper-small" if large-v3-turbo is too slow/heavy

    language_code = "en"
    output_dir = "./output_hf_turbo/"
    hf_token = None

    overall_start_time = time.time()

    # --- Initial System Checks ---
    print("=" * 60)
    print(f"Transcription & Diarization System (HF Whisper: {whisper_model_hf_id}) Initializing")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print(
            "ERROR: Transformers library not found. Please install it: pip install transformers accelerate sentencepiece")
        sys.exit(1)

    pytorch_device_info = "CPU for PyTorch operations."
    mps_available_flag = torch.backends.mps.is_available()
    cuda_available_flag = torch.cuda.is_available()  # Should not be true on Mac
    selected_torch_device = torch.device("cpu")

    if mps_available_flag:
        pytorch_device_info = "Apple Silicon GPU (MPS) acceleration available."
        selected_torch_device = torch.device("mps")
        print("✓ Attempting to use MPS for PyTorch operations.")
    elif cuda_available_flag:
        pytorch_device_info = "NVIDIA GPU (CUDA) acceleration available."
        selected_torch_device = torch.device("cuda")
        print("✓ Attempting to use CUDA for PyTorch operations.")
    else:
        print("✓ Using CPU for PyTorch operations (MPS/CUDA not available).")
    print(f"✓ System Check: {pytorch_device_info}")

    # --- Step 1: Speaker Diarization (using pyannote.audio) ---
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

        print(f"Attempting to move pyannote pipeline to device: {selected_torch_device}...")
        try:
            diarization_pipeline.to(selected_torch_device)
            print(f"✓ Pyannote pipeline explicitly set to {selected_torch_device} device.")
        except Exception as e:
            print(f"Warning: Could not explicitly move pyannote pipeline to {selected_torch_device}: {e}")
            print("Proceeding with default device placement for pyannote (likely CPU if move failed).")

        print(f"Running diarization on: {audio_file_path_str} (this may take a while)...")
        audio_path_for_diar = Path(audio_file_path_str)
        if not audio_path_for_diar.exists():
            print(f"Critical Error: Audio file '{audio_file_path_str}' not found for diarization.")
            sys.exit(1)

        diarization_result = diarization_pipeline(str(audio_path_for_diar), num_speakers=None)

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
        print("Exiting script due to diarization failure.")
        sys.exit(1)
    finally:
        diarization_elapsed_time = time.time() - diarization_step_start_time
        print(f"Speaker Diarization (Step 1) processing time: {format_time_display(diarization_elapsed_time)}.")

    if not speaker_timeline:
        print("Critical Error: Speaker timeline is empty after diarization.")
        print("Exiting script as a speaker-diarized output cannot be effectively produced.")
        sys.exit(1)

    # --- Step 2: Audio Transcription (using Hugging Face Transformers Whisper) ---
    print("\n--- Step 2 of 2: Audio Transcription (using Hugging Face Transformers) ---")
    transcription_step_start_time = time.time()
    transcription_result_segments = []
    result_hf = {}

    try:
        print(f"Loading Whisper model ('{whisper_model_hf_id}') via Hugging Face Transformers...")

        # Determine torch_dtype based on device
        # Using float16 on GPU (MPS/CUDA) can offer speed benefits but might have precision trade-offs for some models.
        # float32 is safer for CPU.
        current_torch_dtype = torch.float32
        if selected_torch_device.type != 'cpu':
            # Check if fp16 is supported on MPS for your PyTorch version
            # For older PyTorch versions, full fp16 on MPS was problematic.
            # Let's default to float32 for MPS for wider compatibility first,
            # can change to torch.float16 if performance is good and stable.
            # For now, to maximize chance of avoiding the sparse tensor error, let's keep float32,
            # or if user wants to try fp16 for speed:
            # current_torch_dtype = torch.float16
            # print(f"Attempting to use torch_dtype: {current_torch_dtype} on {selected_torch_device}")
            pass  # Keep float32 for MPS by default for stability given past issues

        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model_hf_id,
            chunk_length_s=30,
            device=selected_torch_device,
            torch_dtype=current_torch_dtype
        )
        print(f"Whisper ASR pipeline loaded on device: {asr_pipeline.device} with dtype: {current_torch_dtype}")

        audio_path_for_trans = Path(audio_file_path_str)
        if not audio_path_for_trans.exists():
            print(f"Critical Error: Audio file '{audio_file_path_str}' not found for transcription.")
            sys.exit(1)

        print(f"Starting transcription of: {audio_file_path_str} with Transformers Whisper...")
        print("Note: Transcription progress for this step is not granular with the pipeline API.")

        transcription_output = asr_pipeline(
            str(audio_path_for_trans),
            generate_kwargs={"language": language_code},  # Force language
            return_timestamps="chunks"
        )

        full_text_parts = []
        if "chunks" in transcription_output and transcription_output["chunks"] is not None:
            for chunk in transcription_output["chunks"]:
                start_sec, end_sec = chunk["timestamp"]
                text = chunk["text"].strip()

                # Handle None timestamps which can sometimes occur
                current_start_sec = float(start_sec) if start_sec is not None else (
                    transcription_result_segments[-1]['end'] if transcription_result_segments else 0.0)
                current_end_sec = float(end_sec) if end_sec is not None else (
                            current_start_sec + 1.0)  # Arbitrary 1s duration if end is None

                transcription_result_segments.append({
                    "start": current_start_sec,
                    "end": current_end_sec,
                    "text": text
                })
                full_text_parts.append(text)
        else:
            print("Warning: Chunk timestamps not found or empty in Transformers output. Using full text.")
            full_text_parts.append(transcription_output.get("text", ""))

        result_hf = {
            "text": " ".join(full_text_parts).strip(),
            "segments": transcription_result_segments,
            "language": language_code  # Or one detected by the pipeline
        }
        print("Transcription with Transformers Whisper finished.")

    except Exception as e:
        print(f"Critical Error during Transformers Whisper transcription or model loading: {e}")
        if "aten::_sparse_coo_tensor_with_dims_and_tensors" in str(e):
            print(
                "ERROR: The 'aten::_sparse_coo_tensor_with_dims_and_tensors' error re-occurred even with Transformers.")
            print("This suggests a deeper PyTorch MPS backend issue with this model or operations on your system.")
            print("Consider updating PyTorch to the latest nightly or stable version, or using CPU for Whisper.")
        else:
            print("Ensure 'transformers', 'accelerate', 'torch', 'sentencepiece' are installed correctly.")
        print("Exiting script due to transcription failure.")
        sys.exit(1)
    finally:
        transcription_elapsed_time = time.time() - transcription_step_start_time
        print(
            f"\nTranscription (Step 2 with Transformers) processing time: {format_time_display(transcription_elapsed_time)}.")

    # --- Post-processing and Saving ---
    print("\n--- Post-processing and Saving Results ---")
    if not result_hf or ("segments" not in result_hf and "text" not in result_hf) or (
            not result_hf.get("segments") and not result_hf.get("text")):
        print("Critical Error: Transcription result is empty or invalid. Cannot save output.")
        result_hf = {"text": "Transcription failed or produced no text.",
                     "segments": []}  # Ensure result_hf is a dict for save_transcription_files
    elif result_hf.get("segments"):
        print("Assigning speaker labels to transcript segments...")
        result_hf["segments"] = assign_speakers_to_segments(result_hf["segments"], speaker_timeline)
    else:
        print("No text segments found in transcription result, but full text might exist. Output might be minimal.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_transcription_files(result_hf, Path(audio_file_path_str).stem, output_dir)

    overall_elapsed_time = time.time() - overall_start_time
    print(f"\n--- All processing finished in {format_time_display(overall_elapsed_time)} ---")
    print(f"Output files saved to: {output_dir}")


def save_transcription_files(result, base_filename, output_dir):
    """Saves transcription results in various formats."""
    output_path = Path(output_dir)

    # Output format from user's previous request (no timestamps in this TXT line)
    txt_file_path = output_path / f"{base_filename}_diarized_transcript_hf_turbo.txt"
    try:
        with open(txt_file_path, "w", encoding="utf-8") as f:
            if "segments" in result and result["segments"]:
                for segment in result["segments"]:
                    speaker = segment.get("speaker", "UnknownSpeaker")
                    text = segment.get("text", "").strip()
                    f.write(f"{speaker}: {text}\n")
            elif result.get("text"):
                f.write(f"Full Text (no segmentation/diarization applied):\n{result['text']}\n")
            else:
                f.write("No transcription content available.\n")
        print(f"✓ Custom TXT (diarized) saved: {txt_file_path}")
    except Exception as e:
        print(f"✗ Error saving custom TXT file: {e}")

    plain_txt_path = output_path / f"{base_filename}_plain_transcript_hf_turbo.txt"
    try:
        with open(plain_txt_path, "w", encoding="utf-8") as f:
            f.write(result.get("text", "No plain text transcription available.\n"))
        print(f"✓ Plain TXT (Hugging Face) saved: {plain_txt_path}")
    except Exception as e:
        print(f"✗ Error saving plain TXT file: {e}")

    srt_file_path = output_path / f"{base_filename}_diarized_transcript_hf_turbo.srt"
    try:
        with open(srt_file_path, "w", encoding="utf-8") as f:
            if "segments" in result and result["segments"]:
                for i, segment in enumerate(result["segments"]):
                    speaker = segment.get("speaker", "UnknownSpeaker")
                    text = segment.get("text", "").strip()
                    start_time_str = format_srt_timestamp(segment['start'])
                    end_time_str = format_srt_timestamp(segment['end'])
                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    f.write(f"({speaker}) {text}\n\n")
            else:
                f.write("1\n00:00:00,000 --> 00:00:01,000\nNo transcription content available.\n\n")
        print(f"✓ SRT (diarized) saved: {srt_file_path}")
    except Exception as e:
        print(f"✗ Error saving SRT file: {e}")

    print("Note: TSV/JSON output would require custom formatting for Transformers pipeline output.")


if __name__ == "__main__":
    print("--- In __main__ block, preparing to call transcribe_audio_file() ---")
    transcribe_audio_file()
    print("--- transcribe_audio_file() call completed ---")
    print("--- diarisation_hf_turbo.py script execution finished (end of file) ---")