# Advanced Audio Transcription and Diarization Script

This Python script performs speaker diarization followed by audio transcription on an audio file. It is designed to leverage GPU acceleration on Apple Silicon (M1/M2/M3 series Macs) using Metal Performance Shaders (MPS) where possible.

- **Speaker Diarization:** Uses `pyannote.audio` to identify different speakers and their speech segments.
- **Audio Transcription:** Uses Whisper models via the Hugging Face `transformers` library for accurate speech-to-text, with an attempt to use MPS for acceleration.

The script processes an audio file, identifies who spoke when, transcribes the speech for each speaker, and saves the results in various formats including a diarized text file and an SRT file.

## Features

* **Speaker Diarization:** Identifies and timestamps speech segments for different speakers.
* **Accurate Transcription:** Utilizes Whisper models (e.g., "openai/whisper-large-v3-turbo") for high-quality transcription.
* **GPU Acceleration (MPS):**
    * Attempts to use MPS for `pyannote.audio` pipeline.
    * Attempts to use MPS for Whisper model inference via Hugging Face `transformers`.
* **Consistent Speaker Labels:** Outputs speaker labels in a consistent format (Speaker 1, Speaker 2, etc.).
* **Multiple Output Formats:**
    * Diarized transcript text file (`_diarized_transcript_hf_turbo.txt`) with speaker labels.
    * Plain text transcript (`_plain_transcript_hf_turbo.txt`).
    * Diarized SRT subtitle file (`_diarized_transcript_hf_turbo.srt`) with speaker labels.
* **Progress Indicators & Timings:** Provides console output for script progress and timings for major steps.
* **Error Handling:** Includes basic error handling for critical steps.

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Version 3.10 or newer (Python 3.12 was used during development of later versions of this script).
2.  **FFmpeg:** Required by many audio processing libraries for handling various audio formats.
    * On macOS (recommended): `brew install ffmpeg`
3.  **Build Tools (for some dependencies, especially on macOS):**
    * Xcode Command Line Tools: `xcode-select --install`
    * CMake: `brew install cmake` (if not already installed for other packages)
4.  **Hugging Face Account & Token (for `pyannote.audio`):**
    * You need a Hugging Face account.
    * Accept user conditions for `pyannote/speaker-diarization-3.1` on the Hugging Face Hub.
    * Log in via `huggingface-cli login` or set the `hf_token` variable in the script.

## Setup

1.  **Clone the Repository (if applicable) or Download the Script:**
    Ensure you have the latest version of the script (e.g., `diarisation_hf_turbo.py`).

2.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to use a Python virtual environment to manage dependencies for this project.
    ```bash
    # Navigate to your project directory
    cd /path/to/your/project_directory

    # Create a virtual environment using your desired Python version (e.g., python3.12)
    python3.12 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate 
    # On Windows: .venv\Scripts\activate
    ```

3.  **Install Python Packages:**
    While your virtual environment is active, install the required libraries:
    ```bash
    pip install --upgrade pip setuptools wheel
    pip install torch torchvision torchaudio # For PyTorch with MPS/CUDA support
    pip install pyannote.audio
    pip install transformers accelerate sentencepiece # For Hugging Face Whisper
    # tqdm is used internally by pyannote and was used for progress in earlier script versions
    pip install tqdm 
    ```
    *Note on PyTorch installation for Apple Silicon (MPS):* The standard `pip install torch` command for recent PyTorch versions should include MPS support for Apple Silicon Macs. Ensure your PyTorch version is up-to-date.

## Running the Script

1.  **Activate your virtual environment** (if you created one):
    ```bash
    source .venv/bin/activate
    ```
2.  **Navigate to the script's directory.**
3.  **Configure the script:** Open the Python script (e.g., `diarisation_hf_turbo.py`) and modify the configuration section at the top of the `transcribe_audio_file()` function as needed:
    ```python
    # --- Configuration ---
    audio_file_path_str = "IsabelleAudio_trimmed_test.wav" # PATH TO YOUR AUDIO FILE
    whisper_model_hf_id = "openai/whisper-large-v3-turbo"  # Whisper model ID from Hugging Face
    language_code = "en"                                 # Language of the audio
    output_dir = "./output_hf_turbo/"                    # Where to save output files
    hf_token = None # Your Hugging Face token string, or log in via CLI
    ```
    Ensure your target audio file (e.g., `IsabelleAudio_trimmed_test.wav`) is accessible. Using WAV format is generally recommended for fewer compatibility issues.

4.  **Execute the script:**
    ```bash
    python diarisation_hf_turbo.py 
    ```

## Expected Output

* **Console Output:** The script will print progress messages, including:
    * Initialization messages and system checks (PyTorch version, MPS/CUDA availability).
    * Status for "Step 1: Speaker Diarization" (loading pipeline, running diarization, time taken).
    * Status for "Step 2: Audio Transcription" (loading Whisper model, running transcription, time taken).
    * Status of file saving operations.
    * Total processing time.
* **Output Files:** Files will be saved in the directory specified by `output_dir` (e.g., `./output_hf_turbo/`). These include:
    * `[your_audio_basename]_diarized_transcript_hf_turbo.txt`: Text file with speaker labels and transcribed text (format: `Speaker X: Text line`).
    * `[your_audio_basename]_plain_transcript_hf_turbo.txt`: Plain text of the full transcription.
    * `[your_audio_basename]_diarized_transcript_hf_turbo.srt`: SRT subtitle file with speaker labels and timestamps.

## Troubleshooting

* **`ModuleNotFoundError`:** Ensure all packages listed in the "Setup" section are installed within the active Python virtual environment.
* **FFmpeg Errors / `torchaudio` "Format not recognised":** Ensure FFmpeg is installed correctly and accessible in your system's PATH. This is crucial for `torchaudio` (used by `pyannote.audio` and sometimes `transformers`) to load various audio formats. Using 16kHz mono WAV files as input can minimize these issues.
* **`pyannote.audio` Hugging Face Token Issues:** Make sure you have accepted the terms for the `pyannote/speaker-diarization-3.1` model on Hugging Face Hub and are either logged in via `huggingface-cli login` or have provided a valid `hf_token` in the script.
* **PyTorch MPS Errors (e.g., `aten::_sparse_coo_tensor_with_dims_and_tensors` with `openai-whisper`):**
    * The current script uses Hugging Face `transformers` for Whisper, which often has a more stable MPS integration path than direct `openai-whisper`.
    * If you encounter MPS errors with `transformers`, ensure PyTorch, `transformers`, and `accelerate` are fully up-to-date.
    * As a last resort for the transcription step, you can force PyTorch to use the CPU by setting `selected_torch_device = torch.device("cpu")` before loading the `transformers` pipeline (this will be slower).
* **Low GPU Usage on Mac M1/M2/M3 (MPS):**
    * Verify that `torch.backends.mps.is_available()` returns `True`.
    * The script attempts to move `pyannote.audio` and the `transformers` Whisper pipeline to the MPS device.
    * Monitor GPU usage via Activity Monitor (`Window > GPU History`).
    * Not all operations within these complex pipelines may be fully optimized for or offloaded to MPS, so 100% GPU utilization is not always expected. Some parts might remain CPU-bound.
    * Ensure `torch_dtype=torch.float16` is used for the `transformers` pipeline on MPS if you are seeking better performance (the provided script defaults to `torch.float32` for MPS for stability but can be changed).
* **Build Failures for C++ backed libraries (like `whisper-cpp-python` if you were trying that path previously):**
    * Ensure Xcode Command Line Tools and CMake are installed and up-to-date.
    * Check the specific package's GitHub issues page for build instructions or known problems on your OS/Python version.
    * Sometimes, `CMAKE_ARGS` environment variables might be needed during `pip install`.

This README should provide a good overview and guide for using the script.
