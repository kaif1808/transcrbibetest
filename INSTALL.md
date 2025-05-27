# 🛠️ Installation & Setup Guide

## 📋 Prerequisites

- **Python 3.8+** (Python 3.12 recommended)
- **Apple Silicon Mac** (for optimal MLX performance)
- **16GB+ RAM** recommended for large audio files
- **FFMPEG** for audio processing

## 🚀 Quick Installation

### 1. Clone and Setup Virtual Environment

```bash
# Clone the repository
git clone <repository-url>
cd transcrbibetest

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
# Install optimized requirements (recommended)
pip install -r requirements/requirements_optimized.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Alternative requirements based on use case:
# pip install -r requirements/requirements_lightning_mlx.txt  # For Lightning MLX only
# pip install -r requirements/requirements_gpu_optimized.txt  # For GPU optimization
# pip install -r requirements/requirements_correction.txt     # For correction features only
```

### 3. Setup MLX Models

```bash
# The models will be downloaded automatically on first run
# Or manually download to data/models/ directory
mkdir -p data/models
```

## 🎯 Quick Start

### Test the System

```bash
# Run the optimized transcription test
python scripts/run_optimized_test.py

# Run full audio processing test
python scripts/run_full_audio_test.py
```

### Basic Usage

```python
# Import the optimized system
from src.optimizations.optimized_transcription_system import run_optimized_test

# Process audio files
results = await run_optimized_test(
    audio_files=["path/to/audio.wav"],
    document_path="path/to/document.docx"
)
```

## 📁 Project Structure Overview

```
├── src/                    # Core source code
│   ├── core/              # Main transcription modules
│   │   ├── lightning_whisper_mlx_transcriber.py
│   │   ├── transcription_corrector.py
│   │   ├── noun_extraction_system.py
│   │   └── diarisation_hf_turbo.py
│   └── optimizations/     # Performance optimization modules
│       └── optimized_transcription_system.py
├── scripts/               # Execution scripts
│   ├── run_optimized_test.py
│   └── run_full_audio_test.py
├── tests/                 # Test files
├── docs/                  # Documentation
├── requirements/          # Dependency files
├── data/                  # Input/output data
│   ├── input/            # Audio files and documents
│   ├── models/           # MLX models
│   └── output/           # Processing results
└── results/              # Test results and benchmarks
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set cache directory
export HF_HOME=./data/models/.cache

# Optional: Set MLX cache
export MLX_CACHE_DIR=./data/models/.cache/mlx
```

### System Requirements

- **Memory**: 16GB+ for large audio files (63+ minutes)
- **Storage**: 5GB+ for models and cache
- **Network**: Required for initial model downloads

## 🧪 Running Tests

```bash
# Integration tests
python tests/test_correction_integration.py

# Noun extraction tests
python tests/test_noun_extraction_docx.py

# Phrase extraction tests
python tests/test_phrase_extraction.py

# LLM fix tests
python tests/test_llm_fix.py
```

## 📊 Performance Benchmarks

The system has been tested with:
- **Audio File**: 115MB, 63.5 minutes
- **Processing Time**: 277 seconds (4.6 minutes)
- **Speed Factor**: 29.5x faster than real-time
- **Speakers**: Multi-speaker diarization
- **Words**: 8,234+ words transcribed with high accuracy

## 🔍 Troubleshooting

### Common Issues

1. **MLX Model Download Fails**
   ```bash
   # Clear cache and retry
   rm -rf data/models/.cache
   python scripts/run_optimized_test.py
   ```

2. **Out of Memory**
   ```bash
   # Reduce batch size in config
   # Use smaller audio chunks
   ```

3. **FFMPEG Not Found**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt install ffmpeg
   ```

4. **SpaCy Model Missing**
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Performance Optimization

1. **For Apple Silicon Macs**: Ensure MPS is available
2. **For Large Files**: Use batch processing mode
3. **Memory Management**: Close other applications during processing

## 🆘 Support

If you encounter issues:

1. Check the `results/` directory for error logs
2. Verify all dependencies are installed correctly
3. Ensure sufficient disk space and memory
4. Check that audio files are in supported formats (WAV, MP3, M4A)

## 🔄 Updates

To update the system:

```bash
git pull origin main
pip install -r requirements/requirements_optimized.txt --upgrade
```

---

*Installation complete! The system is ready for high-performance audio transcription.* 