# Lightning Whisper MLX Transcription Suite

Ultra-fast audio transcription and speaker diarization optimized for Apple Silicon M1/M2/M3 Macs using the Lightning Whisper MLX framework.

## 🚀 Performance Highlights

- **Lightning-fast processing**: 15-31x faster than real-time
- **Sub-10% processing ratios**: Process 63 minutes of audio in ~4 minutes
- **Apple Silicon optimized**: Full MLX framework utilization
- **Advanced speaker diarization**: Automatic speaker identification with MPS GPU acceleration

## 📦 Available Models

### 🌟 Primary Model: Lightning Whisper MLX (Recommended)
- **File**: `lightning_whisper_mlx_transcriber.py`
- **Performance**: 3.2-6.5% processing time ratios
- **Speed**: 15-31x faster than real-time
- **Features**: MLX optimization, batched decoding, distilled models
- **Best for**: Maximum speed with excellent accuracy

### 🔧 Legacy Models (Maintained)
- **`diarisation_hf_turbo.py`**: Hugging Face Turbo Whisper with MPS acceleration
- **`diarisation.py`**: Original diarization implementation

## 🛠️ Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd transcription-suite

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install Lightning MLX dependencies
pip install -r requirements_lightning_mlx.txt
```

### 2. Run Lightning MLX Transcription

```bash
# Process with validation and full test
python lightning_whisper_mlx_transcriber.py
```

The script will automatically:
1. Run validation test on `IsabelleAudio_trimmed_test.wav` (180s)
2. Run full test on `IsabelleAudio.wav` (63 minutes)
3. Generate comprehensive outputs and benchmarks

## 📁 File Structure

### Core Scripts
```
lightning_whisper_mlx_transcriber.py    # 🌟 Primary Lightning MLX implementation
diarisation_hf_turbo.py                 # 🔧 Legacy HF Turbo model
diarisation.py                          # 🔧 Legacy original model
```

### Configuration & Documentation
```
requirements_lightning_mlx.txt          # Lightning MLX dependencies
README_LIGHTNING_MLX.md                 # Detailed Lightning MLX documentation
LIGHTNING_MLX_FINAL_REPORT.md           # Performance analysis report
```

### Audio Files
```
IsabelleAudio_trimmed_test.wav          # 3-minute validation file
IsabelleAudio.wav                       # 63-minute full test file
```

### Output Structure
```
output_lightning_mlx/                   # Lightning MLX results
├── [filename]_lightning_mlx.txt         # Plain transcription
├── [filename]_lightning_mlx_diarized.txt # Speaker-labeled transcript  
├── [filename]_lightning_mlx.srt         # Subtitle format
└── [filename]_lightning_mlx.json        # Detailed metadata
```

## ⚡ Lightning MLX Configuration

### Default Optimized Settings
```python
config = LightningMLXConfig(
    model="base",                   # Balanced speed/quality
    batch_size=12,                  # M1 Max optimized  
    quant=None,                     # Stability over memory
    chunk_length_s=30,              # 30-second chunks
    use_diarization=True            # Speaker identification
)
```

### Model Options
- **`tiny`**: Fastest processing, basic accuracy
- **`base`**: Balanced performance (recommended)  
- **`distil-medium.en`**: Enhanced English accuracy
- **`distil-large-v3`**: Highest quality

### Quantization Options
- **`None`**: Full precision (recommended for stability)
- **`8bit`**: Memory efficient with good quality
- **`4bit`**: Maximum speed, minimal memory

## 📊 Performance Comparison

| Framework | Processing Ratio | Speed | Quality |
|-----------|------------------|-------|---------|
| ⚡ Lightning MLX | 3.2-6.5% | 15-31x RT | Excellent |
| 🔧 HF Turbo | ~31% | 3.3x RT | High |
| 🔧 Original | ~35% | 2.8x RT | High |

*RT = Real-time*

## 🔧 Legacy Model Usage

### Hugging Face Turbo Model
```bash
python diarisation_hf_turbo.py
```

### Original Diarization Model  
```bash
python diarisation.py
```

Both legacy models support:
- MPS GPU acceleration on Apple Silicon
- Speaker diarization with pyannote.audio
- Multiple output formats (TXT, SRT, JSON)

## 📋 System Requirements

### Minimum Requirements
- Apple Silicon M1/M2/M3 Mac
- macOS 12.0 or later
- 8GB unified memory
- Python 3.9+

### Recommended (for Lightning MLX)
- Apple Silicon M1 Max/Pro/Ultra
- 32GB unified memory  
- macOS 14.0+
- Python 3.11+

## 🚨 Prerequisites

1. **FFmpeg Installation:**
   ```bash
   brew install ffmpeg
   ```

2. **Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   ```

3. **Hugging Face Authentication (for diarization):**
   - Create account at https://huggingface.co
   - Accept terms for `pyannote/speaker-diarization-3.1`
   - Login: `huggingface-cli login`

## 🎯 Expected Output

### Lightning MLX Console Output
```
🚀 LIGHTNING WHISPER MLX BENCHMARK
🍎 Apple Silicon M1 Max Optimized
⚡ 10x faster than Whisper CPP claimed
================================================================================
Phase 1: Validation Test
✅ Lightning MLX initialized in 0.26s
🎵 Transcribing: IsabelleAudio_trimmed_test.wav
📊 VALIDATION TEST RESULTS
⏱️  Total time: 5.80s
🎵 Audio duration: 180.0s  
📈 Processing ratio: 0.0322x
🚀 Speed: 31.0x faster than real-time
🏆 EXCELLENT: Sub-5% processing time!
```

### Output Files
- **Plain Text**: Complete transcription
- **Diarized Text**: Speaker-labeled format (`[timestamp] Speaker_X: text`)
- **SRT Subtitles**: Time-coded subtitles with speaker labels
- **JSON Metadata**: Detailed results with performance metrics

## 🔍 Troubleshooting

### Common Issues

1. **MLX Framework Not Available:**
   ```bash
   pip install mlx
   ```

2. **Memory Issues:**
   - Reduce batch size: `batch_size=8` or `batch_size=4`
   - Enable quantization: `quant="8bit"`
   - Use smaller model: `model="tiny"`

3. **Diarization Fails:**
   - Check Hugging Face authentication
   - Verify MPS availability: `torch.backends.mps.is_available()`
   - Fallback to CPU if needed

4. **Audio Format Issues:**
   - Convert to 16kHz mono WAV: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`
   - Ensure FFmpeg is in PATH

### Performance Optimization

**For Maximum Speed:**
```python
config = LightningMLXConfig(
    model="tiny",
    batch_size=20,
    quant="4bit", 
    use_diarization=False
)
```

**For Maximum Quality:**
```python
config = LightningMLXConfig(
    model="distil-large-v3",
    batch_size=8,
    quant=None,
    use_diarization=True
)
```

## 📚 Documentation

- **`README_LIGHTNING_MLX.md`**: Comprehensive Lightning MLX guide
- **`LIGHTNING_MLX_FINAL_REPORT.md`**: Performance analysis and benchmarks

## 🎉 Quick Test

```bash
# Test Lightning MLX on sample file
python lightning_whisper_mlx_transcriber.py

# Expected: ~6 seconds for 180s audio (31x speed)
```

## 📄 License

This project maintains compatibility with the Lightning Whisper MLX framework license and includes optimizations for Apple Silicon performance.

---

**🚀 For maximum performance, use Lightning Whisper MLX. For compatibility or specific requirements, legacy models remain available.**
