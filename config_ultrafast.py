#!/usr/bin/env python3
"""
Ultra-Fast Transcription Configuration File
Modify these settings for your specific use case and hardware
"""

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

# Input/Output Settings
AUDIO_FILE = "IsabelleAudio.wav"  # Path to your audio file
OUTPUT_DIR = "./output_ultrafast/"  # Where to save results
LANGUAGE = "en"  # Audio language code (en, es, fr, de, etc.)

# Hugging Face Authentication (required for pyannote models)
HF_TOKEN = None  # Set to your HF token string, or use 'huggingface-cli login'

# =============================================================================
# MODEL SELECTION (Quality vs Speed Trade-off)
# =============================================================================

# Whisper Model Options (in order of speed vs quality):
# - "openai/whisper-base"           # Fastest, lower quality
# - "openai/whisper-small"          # Fast, good quality
# - "openai/whisper-large-v3-turbo" # Best quality, recommended
# - "openai/whisper-large-v3"       # Highest quality, slower

WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # Recommended for balance

# Use OpenAI Whisper instead of Transformers (often faster)
USE_OPENAI_WHISPER = False  # Set to True for extreme speed script

# =============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================

# Chunking Configuration
CHUNK_LENGTH_S = 15     # Seconds per chunk (smaller = more parallel, higher overhead)
OVERLAP_S = 1.0         # Overlap between chunks (helps with continuity)

# Parallel Processing
MAX_WORKERS = 4         # Number of parallel workers (set to your CPU cores)
BATCH_SIZE = 4          # Batch size for GPU processing

# GPU/Precision Settings
USE_FP16 = True         # Use half precision for speed (GPU only)
OPTIMIZE_MEMORY = True  # Enable memory optimizations

# =============================================================================
# QUALITY VS SPEED TRADE-OFFS
# =============================================================================

# Transcription Quality Settings
BEAM_SIZE = 1                    # Beam search size (1=fastest, 5=best quality)
NO_SPEECH_THRESHOLD = 0.3        # Skip silence detection threshold

# Speaker Diarization Settings
DIARIZATION_MIN_SPEAKERS = 1     # Minimum expected speakers
DIARIZATION_MAX_SPEAKERS = 10    # Maximum expected speakers (limits processing time)
FAST_DIARIZATION = True          # Use faster diarization method

# =============================================================================
# HARDWARE-SPECIFIC OPTIMIZATIONS
# =============================================================================

# Automatically detect and configure for your hardware
import torch
import multiprocessing as mp

def get_optimal_config():
    """Auto-configure based on available hardware"""
    config = {}
    
    # Detect available hardware
    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()
    cpu_count = mp.cpu_count()
    
    if has_mps:
        # Apple Silicon optimizations
        config.update({
            'max_workers': min(8, cpu_count),  # Use efficiency + performance cores
            'batch_size': 6,                   # Higher batch for unified memory
            'use_fp16': True,
            'chunk_length_s': 15,
        })
        print("🍎 Configured for Apple Silicon (MPS)")
        
    elif has_cuda:
        # NVIDIA GPU optimizations
        config.update({
            'max_workers': cpu_count,
            'batch_size': 8,                   # Higher batch for dedicated VRAM
            'use_fp16': True,
            'chunk_length_s': 15,
        })
        print("🔥 Configured for NVIDIA GPU (CUDA)")
        
    else:
        # CPU-only optimizations
        config.update({
            'max_workers': cpu_count,
            'batch_size': 2,                   # Lower batch to prevent memory issues
            'use_fp16': False,                 # FP16 not beneficial on CPU
            'chunk_length_s': 12,              # Smaller chunks for CPU
        })
        print("💻 Configured for CPU-only processing")
    
    return config

# Apply auto-configuration
AUTO_CONFIG = get_optimal_config()

# Override with auto-detected values (comment out lines you want to keep manual)
MAX_WORKERS = AUTO_CONFIG.get('max_workers', MAX_WORKERS)
BATCH_SIZE = AUTO_CONFIG.get('batch_size', BATCH_SIZE)
USE_FP16 = AUTO_CONFIG.get('use_fp16', USE_FP16)
CHUNK_LENGTH_S = AUTO_CONFIG.get('chunk_length_s', CHUNK_LENGTH_S)

# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

class PresetConfigs:
    """Predefined configurations for different use cases"""
    
    @staticmethod
    def maximum_speed():
        """Extreme speed with acceptable quality loss"""
        return {
            'whisper_model': 'openai/whisper-base',
            'use_openai_whisper': True,
            'chunk_length_s': 10,
            'overlap_s': 0.5,
            'beam_size': 1,
            'batch_size': 8,
            'fast_diarization': True,
        }
    
    @staticmethod 
    def balanced():
        """Good balance of speed and quality"""
        return {
            'whisper_model': 'openai/whisper-large-v3-turbo', 
            'use_openai_whisper': False,
            'chunk_length_s': 15,
            'overlap_s': 1.0,
            'beam_size': 1,
            'batch_size': 4,
            'fast_diarization': True,
        }
    
    @staticmethod
    def maximum_quality():
        """Best quality with longer processing time"""
        return {
            'whisper_model': 'openai/whisper-large-v3',
            'use_openai_whisper': False, 
            'chunk_length_s': 20,
            'overlap_s': 2.0,
            'beam_size': 5,
            'batch_size': 2,
            'fast_diarization': False,
        }

# =============================================================================
# EASY PRESET SELECTION
# =============================================================================

# Uncomment one of these to use a preset configuration:

# Use maximum speed preset
# preset = PresetConfigs.maximum_speed()

# Use balanced preset (recommended)
preset = PresetConfigs.balanced()

# Use maximum quality preset  
# preset = PresetConfigs.maximum_quality()

# Apply preset settings
WHISPER_MODEL = preset.get('whisper_model', WHISPER_MODEL)
USE_OPENAI_WHISPER = preset.get('use_openai_whisper', USE_OPENAI_WHISPER)
CHUNK_LENGTH_S = preset.get('chunk_length_s', CHUNK_LENGTH_S)
OVERLAP_S = preset.get('overlap_s', OVERLAP_S)
BEAM_SIZE = preset.get('beam_size', BEAM_SIZE)
BATCH_SIZE = preset.get('batch_size', BATCH_SIZE)
FAST_DIARIZATION = preset.get('fast_diarization', FAST_DIARIZATION)

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config_summary():
    """Print current configuration for verification"""
    print("=" * 60)
    print("📋 ULTRA-FAST TRANSCRIPTION CONFIGURATION")
    print("=" * 60)
    print(f"📁 Audio file: {AUDIO_FILE}")
    print(f"🤖 Whisper model: {WHISPER_MODEL}")
    print(f"🌍 Language: {LANGUAGE}")
    print(f"📊 Chunk length: {CHUNK_LENGTH_S}s")
    print(f"🔄 Overlap: {OVERLAP_S}s")
    print(f"⚡ Workers: {MAX_WORKERS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🎯 Beam size: {BEAM_SIZE}")
    print(f"💾 FP16: {USE_FP16}")
    print(f"🎤 Fast diarization: {FAST_DIARIZATION}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    print_config_summary() 