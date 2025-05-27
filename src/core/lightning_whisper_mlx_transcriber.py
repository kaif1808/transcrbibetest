#!/usr/bin/env python3
"""
Lightning Whisper MLX Transcriber
Ultra-fast transcription using Lightning Whisper MLX framework
Optimized for Apple Silicon M1 Max with 32 GPU cores
Claims: 10x faster than Whisper CPP, 4x faster than current MLX Whisper
"""

import asyncio
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import soundfile as sf
from dataclasses import dataclass
import logging

# Import transcription correction
try:
    from .transcription_corrector import enhance_transcription_result, create_correction_config
    CORRECTION_AVAILABLE = True
    print("✅ Transcription correction module available")
except ImportError:
    CORRECTION_AVAILABLE = False
    print("⚠️  Transcription correction not available. Install with: pip install -r requirements_correction.txt")

# Import noun extraction
try:
    from .noun_extraction_system import extract_nouns_from_transcription, save_noun_analysis
    NOUN_EXTRACTION_AVAILABLE = True
    print("✅ Noun extraction module available")
except ImportError:
    NOUN_EXTRACTION_AVAILABLE = False
    print("⚠️  Noun extraction not available. Install with: pip install -r requirements_correction.txt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LightningMLXConfig:
    """Configuration for Lightning Whisper MLX"""
    audio_file: str = "IsabelleAudio_trimmed_test.wav"
    output_dir: str = "./output_lightning_mlx/"
    
    # Lightning MLX specific settings
    model: str = "distil-large-v3"  # High-quality distilled model for testing
    batch_size: int = 12  # Default recommended batch size
    quant: Optional[str] = None  # Disable quantization for compatibility
    
    # Available models from Lightning MLX
    available_models: List[str] = None
    
    # Chunking for large files
    chunk_length_s: int = 30  # seconds per chunk
    overlap_s: float = 1.0    # overlap between chunks
    max_parallel_chunks: int = 8  # Conservative for MLX
    
    # Diarization settings
    use_diarization: bool = True
    min_speakers: int = 1
    max_speakers: int = 8
    
    # Transcription correction settings
    enable_correction: bool = True
    correction_domain: str = "education"  # education, technology, business, general
    
    # Noun extraction settings
    enable_noun_extraction: bool = True
    noun_extraction_domain: str = "education"  # education, technology, business, general
    extract_phrases: bool = True  # Enable multi-word phrase extraction
    use_gpu_for_extraction: bool = True  # Use GPU acceleration for noun extraction
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                "tiny", "small", "distil-small.en", 
                "base", "medium", "distil-medium.en",
                "large", "large-v2", "distil-large-v2", 
                "large-v3", "distil-large-v3"
            ]


class LightningMLXProcessor:
    """Lightning Whisper MLX processor with M1 Max optimizations"""
    
    def __init__(self, config: LightningMLXConfig):
        self.config = config
        self.whisper = None
        self.model_info = None
        self._initialized = False
        
    async def cleanup(self):
        """Clean up resources"""
        if self.whisper:
            try:
                # Attempt to free MLX resources if possible
                del self.whisper
                self.whisper = None
                print("🧹 Lightning MLX resources cleaned up")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize Lightning Whisper MLX model"""
        try:
            from lightning_whisper_mlx import LightningWhisperMLX
            import mlx.core as mx
            
            print(f"🚀 Initializing Lightning Whisper MLX...")
            print(f"📦 Model: {self.config.model}")
            print(f"🔢 Batch size: {self.config.batch_size}")
            print(f"⚡ Quantization: {self.config.quant}")
            
            # Check MLX availability with proper error handling
            try:
                device = mx.default_device()
                print(f"💻 MLX backend available: {device}")
            except Exception as e:
                raise RuntimeError(f"MLX backend not available: {e}")
            
            init_start = time.time()
            
            # Store original config for fallback reference
            original_model = self.config.model
            original_batch_size = self.config.batch_size
            
            # Initialize with original settings
            try:
                self.whisper = LightningWhisperMLX(
                    model=self.config.model,
                    batch_size=self.config.batch_size,
                    quant=self.config.quant
                )
                print(f"✅ Initialized with requested config: {original_model}, batch_size={original_batch_size}")
                
            except Exception as e:
                print(f"⚠️ Failed with original config: {e}")
                print(f"🔄 Attempting fallback to minimal configuration...")
                
                # Fallback to minimal configuration without modifying original config
                try:
                    self.whisper = LightningWhisperMLX(
                        model="tiny",
                        batch_size=1,
                        quant=None
                    )
                    print(f"✅ Fallback successful: tiny model, batch_size=1")
                    print(f"ℹ️ Note: Using fallback config instead of requested {original_model}")
                    
                except Exception as fallback_error:
                    raise RuntimeError(f"Both original and fallback initialization failed. Original: {e}, Fallback: {fallback_error}")
            
            init_time = time.time() - init_start
            
            # Determine actual model info (may be different from config if fallback was used)
            actual_model = "tiny" if original_model != "tiny" and hasattr(self.whisper, '_model_name') else self.config.model
            actual_batch_size = 1 if original_batch_size != 1 and actual_model == "tiny" else self.config.batch_size
            
            self.model_info = {
                "requested_model": original_model,
                "actual_model": actual_model,
                "requested_batch_size": original_batch_size,
                "actual_batch_size": actual_batch_size,
                "quantization": self.config.quant,
                "device": str(device),
                "init_time": init_time,
                "fallback_used": actual_model != original_model
            }
            
            print(f"✅ Lightning MLX initialized in {init_time:.2f}s")
            print(f"🖥️  Device: {device}")
            
            # Skip warmup for now to avoid compatibility issues
            print("⚡ Skipping warmup due to compatibility mode")
            
            self._initialized = True
            
        except ImportError as e:
            raise ImportError(f"Lightning Whisper MLX not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Lightning MLX: {e}")
    
    async def transcribe_file(self, audio_file: str, start_offset: float = 0.0) -> Dict[str, Any]:
        """Transcribe audio file with Lightning MLX"""
        try:
            print(f"🎵 Transcribing: {audio_file}")
            
            transcribe_start = time.time()
            
            # Use Lightning MLX transcription
            result = self.whisper.transcribe(audio_path=audio_file)
            
            transcribe_time = time.time() - transcribe_start
            
            # Handle different result formats from Lightning MLX
            text = ""
            segments = []
            language = "en"
            
            if isinstance(result, dict):
                text = result.get('text', '').strip()
                language = result.get('language', 'en')
                
                # Process segments if they exist
                if 'segments' in result:
                    segments_data = result['segments']
                    if isinstance(segments_data, list):
                        for segment in segments_data:
                            if isinstance(segment, list) and len(segment) >= 3:
                                # Lightning MLX format: [start_ms, end_ms, text]
                                start_s = segment[0] / 1000.0  # Convert ms to seconds
                                end_s = segment[1] / 1000.0    # Convert ms to seconds
                                segment_text = segment[2].strip()
                                
                                adjusted_segment = {
                                    "start": start_s + start_offset,
                                    "end": end_s + start_offset,
                                    "text": segment_text
                                }
                                if adjusted_segment["text"]:
                                    segments.append(adjusted_segment)
                            elif isinstance(segment, dict):
                                # Standard format fallback
                                adjusted_segment = {
                                    "start": segment.get('start', 0) + start_offset,
                                    "end": segment.get('end', 0) + start_offset,
                                    "text": segment.get('text', '').strip()
                                }
                                if adjusted_segment["text"]:
                                    segments.append(adjusted_segment)
            elif isinstance(result, list):
                # Handle if result is a list of segments
                for segment in result:
                    if isinstance(segment, list) and len(segment) >= 3:
                        # Lightning MLX format: [start_ms, end_ms, text]
                        start_s = segment[0] / 1000.0
                        end_s = segment[1] / 1000.0
                        segment_text = segment[2].strip()
                        
                        if segment_text:
                            text += segment_text + " "
                            adjusted_segment = {
                                "start": start_s + start_offset,
                                "end": end_s + start_offset,
                                "text": segment_text
                            }
                            segments.append(adjusted_segment)
                    elif isinstance(segment, dict):
                        segment_text = segment.get('text', '').strip()
                        if segment_text:
                            text += segment_text + " "
                            adjusted_segment = {
                                "start": segment.get('start', 0) + start_offset,
                                "end": segment.get('end', 0) + start_offset,
                                "text": segment_text
                            }
                            segments.append(adjusted_segment)
                text = text.strip()
            else:
                # Handle if result is just text
                text = str(result).strip()
            
            return {
                "text": text,
                "segments": segments,
                "language": language,
                "transcribe_time": transcribe_time
            }
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_file}: {e}")
            return {
                "text": "",
                "segments": [],
                "language": "unknown",
                "transcribe_time": 0.0,
                "error": str(e)
            }


class LightningMLXChunker:
    """Audio chunking for Lightning MLX processing"""
    
    @staticmethod
    def create_chunks(audio_file: str, config: LightningMLXConfig) -> List[Tuple[str, float, float]]:
        """Create audio chunks using FFmpeg for Lightning MLX"""
        try:
            # Get audio duration
            with sf.SoundFile(audio_file) as f:
                duration = len(f) / f.samplerate
            
            print(f"📊 Audio duration: {duration:.1f}s")
            
            # For smaller files, don't chunk
            if duration <= config.chunk_length_s * 1.5:
                print(f"📄 File small enough, processing without chunking")
                return [(audio_file, 0.0, duration)]
            
            chunk_length = config.chunk_length_s
            overlap = config.overlap_s
            step = chunk_length - overlap
            
            chunks = []
            temp_dir = Path(f"lightning_mlx_chunks_{int(time.time())}")
            temp_dir.mkdir(exist_ok=True)
            
            start_time = 0
            chunk_idx = 0
            
            while start_time < duration:
                end_time = min(start_time + chunk_length, duration)
                
                # Create chunk using FFmpeg
                chunk_file = temp_dir / f"chunk_{chunk_idx:04d}.wav"
                
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-loglevel", "quiet",
                    "-i", audio_file,
                    "-ss", str(start_time),
                    "-t", str(end_time - start_time),
                    "-ar", "16000",  # Lightning MLX works best with 16kHz
                    "-ac", "1",      # Mono
                    "-c:a", "pcm_s16le",
                    str(chunk_file)
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True)
                if result.returncode == 0:
                    chunks.append((str(chunk_file), start_time, end_time))
                    print(f"📦 Created chunk {chunk_idx + 1}: {start_time:.1f}s - {end_time:.1f}s")
                
                start_time += step
                chunk_idx += 1
            
            print(f"✅ Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            return []


class LightningMLXDiarizer:
    """Speaker diarization for Lightning MLX results"""
    
    def __init__(self, config: LightningMLXConfig):
        self.config = config
        self.pipeline = None
        
    async def initialize(self):
        """Initialize diarization pipeline"""
        if not self.config.use_diarization:
            logger.info("Diarization disabled in configuration")
            return
            
        try:
            from pyannote.audio import Pipeline
            import torch
            
            print("🎤 Initializing speaker diarization...")
            
            # Try to load pipeline with proper error handling
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=True
                )
                print("✅ Diarization pipeline loaded successfully")
                
            except Exception as auth_error:
                if "authentication" in str(auth_error).lower() or "token" in str(auth_error).lower():
                    logger.error("❌ Diarization failed: HuggingFace authentication required")
                    logger.error("   Please set up authentication: huggingface-cli login")
                    logger.error("   Or provide auth token in environment: HF_TOKEN=your_token")
                else:
                    logger.error(f"❌ Diarization pipeline load failed: {auth_error}")
                
                self.pipeline = None
                return
            
            # Device optimization with error handling
            try:
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    self.pipeline.to(device)
                    print("✅ Diarization using Apple Silicon MPS")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                    self.pipeline.to(device)
                    print("✅ Diarization using CUDA GPU")
                else:
                    print("ℹ️ Diarization using CPU")
                    
            except Exception as device_error:
                logger.warning(f"⚠️ GPU setup failed, using CPU: {device_error}")
                
        except ImportError as e:
            logger.error(f"❌ Diarization dependencies missing: {e}")
            logger.error("   Install with: pip install pyannote.audio torch")
            self.pipeline = None
        except Exception as e:
            logger.error(f"❌ Diarization initialization failed: {e}")
            self.pipeline = None
    
    async def process(self, audio_file: str) -> List[Tuple[str, float, float]]:
        """Process speaker diarization"""
        if not self.pipeline:
            logger.warning("⚠️ Diarization pipeline not available, skipping speaker identification")
            return []
        
        try:
            print("🎤 Processing speaker diarization...")
            diarization_start = time.time()
            
            # Validate audio file
            if not Path(audio_file).exists():
                logger.error(f"❌ Audio file not found: {audio_file}")
                return []
            
            # Run diarization with progress indication
            diarization = self.pipeline(
                audio_file,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
            
            timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                timeline.append((speaker, turn.start, turn.end))
            
            diarization_time = time.time() - diarization_start
            speaker_count = len(set(s[0] for s in timeline))
            
            if timeline:
                print(f"✅ Diarization completed in {diarization_time:.2f}s")
                print(f"🎤 Detected {speaker_count} speakers with {len(timeline)} speaking turns")
            else:
                logger.warning("⚠️ No speakers detected in audio file")
            
            return timeline
            
        except Exception as e:
            logger.error(f"❌ Diarization processing failed: {e}")
            logger.error(f"   Audio file: {audio_file}")
            logger.error(f"   Config: min_speakers={self.config.min_speakers}, max_speakers={self.config.max_speakers}")
            return []


class LightningMLXResultProcessor:
    """Result processing for Lightning MLX outputs"""
    
    @staticmethod
    def merge_chunks(chunk_results: List[Dict], chunk_info: List[Tuple[str, float, float]]) -> Dict:
        """Merge chunk results efficiently"""
        all_segments = []
        full_text_parts = []
        total_transcribe_time = 0.0
        
        for i, result in enumerate(chunk_results):
            text = result.get("text", "").strip()
            if text:
                full_text_parts.append(text)
            
            all_segments.extend(result.get("segments", []))
            total_transcribe_time += result.get("transcribe_time", 0.0)
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x.get("start", 0))
        
        return {
            "text": " ".join(full_text_parts),
            "segments": all_segments,
            "total_transcribe_time": total_transcribe_time,
            "chunk_count": len(chunk_results)
        }
    
    @staticmethod
    def assign_speakers(segments: List[Dict], speaker_timeline: List[Tuple[str, float, float]]) -> List[Dict]:
        """Assign speakers to segments"""
        if not speaker_timeline:
            return segments
        
        for segment in segments:
            segment_mid = (segment.get("start", 0) + segment.get("end", 0)) / 2
            
            assigned_speaker = "Unknown"
            for speaker, start, end in speaker_timeline:
                if start <= segment_mid <= end:
                    assigned_speaker = speaker
                    break
            
            segment["speaker"] = assigned_speaker
        
        return segments
    
    @staticmethod
    async def apply_corrections(result: Dict[str, Any], config: LightningMLXConfig) -> Dict[str, Any]:
        """Apply transcription corrections if enabled"""
        if not config.enable_correction or not CORRECTION_AVAILABLE:
            return result
        
        try:
            print("🔧 Applying transcription corrections...")
            
            # Create correction configuration
            correction_config = create_correction_config(
                domain=config.correction_domain,
                enable_ai=True,
                custom_terms=["Lightning MLX", "M1 Max", "Apple Silicon", "TVET", "Vietnam", "Vietnamese"]
            )
            
            # Apply corrections
            corrected_result = await enhance_transcription_result(result, correction_config)
            
            print(f"✅ Corrections applied: {corrected_result.get('correction_info', {}).get('corrections_applied', [])}")
            
            return corrected_result
            
        except Exception as e:
            logger.warning(f"Correction failed: {e}")
            return result
    
    @staticmethod
    async def extract_nouns(result: Dict[str, Any], config: LightningMLXConfig) -> Dict[str, Any]:
        """Extract nouns from transcription result if enabled"""
        if not config.enable_noun_extraction:
            logger.debug("Noun extraction disabled in configuration")
            return result
            
        if not NOUN_EXTRACTION_AVAILABLE:
            logger.warning("⚠️ Noun extraction not available - module not found")
            result["noun_extraction_status"] = "unavailable"
            return result
        
        try:
            print("🔍 Extracting nouns and phrases from transcription...")
            
            # Import with error handling
            try:
                from .noun_extraction_system import NounExtractionConfig, extract_nouns_from_transcription
            except ImportError as import_error:
                logger.error(f"❌ Failed to import noun extraction: {import_error}")
                result["noun_extraction_status"] = "import_failed"
                result["noun_extraction_error"] = str(import_error)
                return result
            
            # Create enhanced configuration for noun extraction
            extraction_config = NounExtractionConfig(
                domain_focus=config.noun_extraction_domain,
                extract_phrases=config.extract_phrases,
                use_gpu_acceleration=config.use_gpu_for_extraction,
                gpu_batch_size=64,
                parallel_processing=True,
                max_workers=4,
                min_phrase_length=2,
                max_phrase_length=8,
                use_local_llm=False,  # Disable LLM for reliability
                extract_unusual_nouns=False
            )
            
            # Extract nouns from the transcription
            noun_analysis = await extract_nouns_from_transcription(
                result, 
                domain=config.noun_extraction_domain,
                config=extraction_config
            )
            
            # Check for extraction errors
            if "error" in noun_analysis:
                logger.error(f"❌ Noun extraction failed: {noun_analysis['error']}")
                result["noun_extraction_status"] = "failed"
                result["noun_extraction_error"] = noun_analysis["error"]
                return result
            
            # Add noun analysis to result
            result["noun_analysis"] = noun_analysis
            result["noun_extraction_status"] = "success"
            
            # Summary of extracted nouns and phrases
            noun_data = noun_analysis.get("noun_analysis", {})
            total_nouns = len(noun_data.get("all_nouns", []))
            
            print(f"✅ Noun extraction completed successfully:")
            print(f"   📝 Total nouns found: {total_nouns}")
            
            # Count by category
            for category, nouns in noun_data.items():
                if nouns and category != "all_nouns":
                    count = len(nouns)
                    category_name = category.replace('_', ' ').title()
                    print(f"   {category_name}: {count}")
                    
                    # Show examples for key categories
                    if category in ["technical_terms", "domain_specific"] and count > 0:
                        examples = [noun["text"] if isinstance(noun, dict) else noun.text for noun in nouns[:3]]
                        print(f"      Examples: {', '.join(examples)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Noun extraction failed with exception: {e}")
            import traceback
            logger.debug(f"Noun extraction traceback: {traceback.format_exc()}")
            
            result["noun_extraction_status"] = "exception"
            result["noun_extraction_error"] = str(e)
            return result


class LightningMLXOutputManager:
    """Output management for Lightning MLX results"""
    
    @staticmethod
    def save_results(result: Dict, audio_filename: str, output_dir: str, model_info: Dict):
        """Save Lightning MLX results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(audio_filename).stem
        
        # Save plain text
        txt_path = output_path / f"{base_name}_lightning_mlx.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result.get("text", ""))
        
        # Save diarized text
        diarized_path = output_path / f"{base_name}_lightning_mlx_diarized.txt"
        with open(diarized_path, "w", encoding="utf-8") as f:
            for segment in result.get("segments", []):
                speaker = segment.get("speaker", "Unknown")
                text = segment.get("text", "").strip()
                start = segment.get("start", 0)
                if text:
                    f.write(f"[{start:.1f}s] {speaker}: {text}\n")
        
        # Save SRT
        srt_path = output_path / f"{base_name}_lightning_mlx.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result.get("segments", [])):
                if segment.get("text", "").strip():
                    start_srt = LightningMLXOutputManager._format_srt_time(segment["start"])
                    end_srt = LightningMLXOutputManager._format_srt_time(segment["end"])
                    speaker = segment.get("speaker", "Unknown")
                    text = segment.get("text", "").strip()
                    
                    f.write(f"{i + 1}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(f"({speaker}) {text}\n\n")
        
        # Save detailed JSON with model info
        json_path = output_path / f"{base_name}_lightning_mlx.json"
        output_data = {
            "result": result,
            "model_info": model_info,
            "framework": "Lightning Whisper MLX",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {output_dir}")
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time for SRT files"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


class LightningMLXBenchmark:
    """Comprehensive Lightning MLX benchmark"""
    
    def __init__(self):
        self.config = LightningMLXConfig()
        self.processor = None
        self.diarizer = None
        
    async def run_validation_test(self) -> Dict[str, Any]:
        """Run validation test on short audio file"""
        print("=" * 80)
        print("🧪 LIGHTNING WHISPER MLX - VALIDATION TEST")
        print("📁 File: IsabelleAudio_trimmed_test.wav")
        print("=" * 80)
        
        # Configure for validation
        self.config.audio_file = "IsabelleAudio_trimmed_test.wav"
        self.config.use_diarization = False  # Speed test first
        
        # Check file exists
        if not Path(self.config.audio_file).exists():
            return {"success": False, "error": f"File not found: {self.config.audio_file}"}
        
        try:
            # Get audio info
            with sf.SoundFile(self.config.audio_file) as f:
                audio_duration = len(f) / f.samplerate
            
            print(f"⏱️  Audio duration: {audio_duration:.1f}s")
            
            overall_start = time.time()
            
            # Initialize processor
            self.processor = LightningMLXProcessor(self.config)
            await self.processor.initialize()
            
            # Process audio (direct transcription for validation)
            print(f"🚀 Starting Lightning MLX transcription...")
            transcribe_start = time.time()
            
            result = await self.processor.transcribe_file(self.config.audio_file)
            
            transcribe_time = time.time() - transcribe_start
            total_time = time.time() - overall_start
            
            # Calculate metrics
            processing_ratio = total_time / audio_duration
            speed_multiplier = 1.0 / processing_ratio if processing_ratio > 0 else 0.0
            
            # Apply corrections if enabled
            result = await LightningMLXResultProcessor.apply_corrections(result, self.config)
            
            # Extract nouns if enabled
            result = await LightningMLXResultProcessor.extract_nouns(result, self.config)
            
            # Save results
            LightningMLXOutputManager.save_results(
                result, self.config.audio_file, self.config.output_dir, 
                self.processor.model_info
            )
            
            # Performance summary
            print("\n" + "=" * 60)
            print("📊 VALIDATION TEST RESULTS")
            print("=" * 60)
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"🎵 Audio duration: {audio_duration:.1f}s")
            print(f"📈 Processing ratio: {processing_ratio:.4f}x")
            print(f"🚀 Speed: {speed_multiplier:.1f}x faster than real-time")
            print(f"📝 Words transcribed: {len(result.get('text', '').split())}")
            print(f"💻 Model: {self.processor.model_info['model']}")
            print(f"⚡ Quantization: {self.processor.model_info['quantization']}")
            
            # Performance assessment
            if processing_ratio < 0.05:
                print("🏆 EXCELLENT: Sub-5% processing time!")
            elif processing_ratio < 0.1:
                print("🥇 GREAT: Sub-10% processing time!")
            elif processing_ratio < 0.2:
                print("🥈 GOOD: Sub-20% processing time")
            else:
                print("🥉 ACCEPTABLE: Above 20% processing time")
            
            return {
                "success": True,
                "audio_duration": audio_duration,
                "processing_time": total_time,
                "processing_ratio": processing_ratio,
                "speed_multiplier": speed_multiplier,
                "word_count": len(result.get('text', '').split()),
                "model": self.processor.model_info['model'],
                "quantization": self.processor.model_info['quantization']
            }
            
        except Exception as e:
            print(f"❌ Validation test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    async def run_full_test(self) -> Dict[str, Any]:
        """Run full test on complete audio file"""
        print("\n" + "=" * 80)
        print("🚀 LIGHTNING WHISPER MLX - FULL TEST")
        print("📁 File: IsabelleAudio.wav")
        print("=" * 80)
        
        # Configure for full test
        self.config.audio_file = "IsabelleAudio.wav"
        self.config.use_diarization = True  # Full features
        
        # Check file exists
        if not Path(self.config.audio_file).exists():
            return {"success": False, "error": f"File not found: {self.config.audio_file}"}
        
        try:
            # Get audio info
            with sf.SoundFile(self.config.audio_file) as f:
                audio_duration = len(f) / f.samplerate
            
            print(f"⏱️  Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} minutes)")
            
            overall_start = time.time()
            
            # Initialize components (reuse processor if available)
            if not self.processor:
                self.processor = LightningMLXProcessor(self.config)
                await self.processor.initialize()
            
            # Initialize diarization
            self.diarizer = LightningMLXDiarizer(self.config)
            await self.diarizer.initialize()
            
            # Create chunks for large file
            chunks = LightningMLXChunker.create_chunks(self.config.audio_file, self.config)
            
            if len(chunks) == 1:
                print("📄 Processing as single file (no chunking needed)")
                
                # Start diarization in parallel
                diarization_task = None
                if self.diarizer.pipeline:
                    diarization_task = asyncio.create_task(
                        self.diarizer.process(self.config.audio_file)
                    )
                
                # Transcribe
                result = await self.processor.transcribe_file(self.config.audio_file)
                
                # Wait for diarization
                speaker_timeline = []
                if diarization_task:
                    speaker_timeline = await diarization_task
                    if speaker_timeline:
                        result["segments"] = LightningMLXResultProcessor.assign_speakers(
                            result["segments"], speaker_timeline
                        )
                
            else:
                print(f"📦 Processing {len(chunks)} chunks in parallel...")
                
                # Start diarization in parallel
                diarization_task = None
                if self.diarizer.pipeline:
                    diarization_task = asyncio.create_task(
                        self.diarizer.process(self.config.audio_file)
                    )
                
                # Process chunks with limited parallelism
                semaphore = asyncio.Semaphore(self.config.max_parallel_chunks)
                
                async def process_chunk_with_semaphore(chunk_info):
                    async with semaphore:
                        chunk_file, start_offset, _ = chunk_info
                        return await self.processor.transcribe_file(chunk_file, start_offset)
                
                # Process all chunks
                chunk_results = await asyncio.gather(*[
                    process_chunk_with_semaphore(chunk_info) 
                    for chunk_info in chunks
                ])
                
                # Merge results
                result = LightningMLXResultProcessor.merge_chunks(chunk_results, chunks)
                
                # Wait for diarization
                speaker_timeline = []
                if diarization_task:
                    speaker_timeline = await diarization_task
                    if speaker_timeline:
                        result["segments"] = LightningMLXResultProcessor.assign_speakers(
                            result["segments"], speaker_timeline
                        )
                
                # Cleanup chunks with proper error handling
                try:
                    for chunk_file, _, _ in chunks:
                        chunk_path = Path(chunk_file)
                        if chunk_path.exists():
                            chunk_path.unlink()
                    
                    # Remove temp directory if it exists and is empty
                    if chunks and len(chunks) > 1:
                        temp_dir = Path(chunks[0][0]).parent
                        if temp_dir.exists() and temp_dir.name.startswith("lightning_mlx_chunks_"):
                            try:
                                temp_dir.rmdir()
                                print(f"🧹 Cleaned up temporary directory: {temp_dir}")
                            except OSError:
                                logger.warning(f"Could not remove temp directory: {temp_dir}")
                                
                except Exception as cleanup_error:
                    logger.warning(f"Chunk cleanup failed: {cleanup_error}")
            
            total_time = time.time() - overall_start
            
            # Calculate metrics
            processing_ratio = total_time / audio_duration
            speed_multiplier = 1.0 / processing_ratio if processing_ratio > 0 else 0.0
            
            # Quality metrics
            segments_count = len(result.get("segments", []))
            speakers_count = len(set(s.get("speaker", "") for s in result.get("segments", [])))
            word_count = len(result.get("text", "").split())
            
            # Apply corrections if enabled
            result = await LightningMLXResultProcessor.apply_corrections(result, self.config)
            
            # Save results
            LightningMLXOutputManager.save_results(
                result, self.config.audio_file, self.config.output_dir, 
                self.processor.model_info
            )
            
            # Performance summary
            print("\n" + "=" * 80)
            print("📊 FULL TEST RESULTS")
            print("=" * 80)
            print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            print(f"🎵 Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} minutes)")
            print(f"📈 Processing ratio: {processing_ratio:.4f}x")
            print(f"🚀 Speed: {speed_multiplier:.1f}x faster than real-time")
            print(f"📝 Segments: {segments_count}")
            print(f"🎤 Speakers: {speakers_count}")
            print(f"📖 Words: {word_count}")
            print(f"💻 Model: {self.processor.model_info['model']}")
            print(f"⚡ Quantization: {self.processor.model_info['quantization']}")
            print(f"📊 Chunks processed: {result.get('chunk_count', 1)}")
            
            # Performance assessment
            if processing_ratio < 0.1:
                print("🏆 EXCELLENT: Sub-10% processing time!")
            elif processing_ratio < 0.2:
                print("🥇 GREAT: Sub-20% processing time!")
            elif processing_ratio < 0.3:
                print("🥈 GOOD: Sub-30% processing time")
            else:
                print("🥉 ACCEPTABLE: Above 30% processing time")
            
            return {
                "success": True,
                "audio_duration": audio_duration,
                "processing_time": total_time,
                "processing_ratio": processing_ratio,
                "speed_multiplier": speed_multiplier,
                "segments_count": segments_count,
                "speakers_count": speakers_count,
                "word_count": word_count,
                "model": self.processor.model_info['model'],
                "quantization": self.processor.model_info['quantization'],
                "chunk_count": result.get('chunk_count', 1)
            }
            
        except Exception as e:
            print(f"❌ Full test failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    async def run_document_extraction_test(self) -> Dict[str, Any]:
        """Run document noun extraction test on inputdoc.docx"""
        print("📄 DOCUMENT NOUN EXTRACTION TEST")
        print("📁 File: inputdoc.docx")
        print("=" * 60)
        
        if not NOUN_EXTRACTION_AVAILABLE:
            return {"success": False, "error": "Noun extraction not available"}
        
        # Check if document exists
        doc_file = "inputdoc.docx"
        if not Path(doc_file).exists():
            return {"success": False, "error": f"Document not found: {doc_file}"}
        
        try:
            from noun_extraction_system import extract_nouns_from_document, NounExtractionConfig
            
            print(f"📄 Processing document: {doc_file}")
            start_time = time.time()
            
            # Create configuration for document extraction
            config = NounExtractionConfig(
                domain_focus="education",
                use_local_llm=True,  # Try to use local LLM
                llm_provider="ollama",  # Prefer Ollama
                llm_model="llama3.2",
                extract_unusual_nouns=True,
                min_frequency=2,
                filter_duplicates=True
            )
            
            # Extract nouns from document
            doc_analysis = await extract_nouns_from_document(doc_file, "education", config)
            extraction_time = time.time() - start_time
            
            if "error" in doc_analysis:
                return {"success": False, "error": doc_analysis["error"]}
            
            # Process results
            content_stats = doc_analysis.get("content_statistics", {})
            noun_results = doc_analysis.get("noun_analysis", {})
            insights = doc_analysis.get("insights", {})
            
            # Summary statistics
            total_chars = content_stats.get("total_characters", 0)
            total_words = content_stats.get("total_words", 0)
            total_nouns = insights.get("total_unique_nouns", 0)
            
            print(f"✅ Document extraction completed in {extraction_time:.2f}s")
            print(f"📊 Document Analysis:")
            print(f"   📝 Characters: {total_chars:,}")
            print(f"   📖 Words: {total_words:,}")
            print(f"   🔍 Unique nouns: {total_nouns}")
            
            # Show noun categories
            print(f"\n🔍 Noun extraction results:")
            for category, nouns in noun_results.items():
                if nouns and category != "all_nouns":
                    print(f"   {category.replace('_', ' ').title()}: {len(nouns)}")
                    
                    # Show special attention to unusual nouns
                    if category == "unusual_nouns":
                        print(f"   🌟 Top unusual nouns (LLM identified):")
                        for noun in nouns[:5]:
                            print(f"     • {noun.text} (confidence: {noun.confidence:.2f})")
                    elif len(nouns) > 0:
                        print(f"     Top terms: {', '.join([n.text for n in nouns[:3]])}")
            
            # Save document analysis
            doc_output = f"inputdoc_analysis_{int(time.time())}.json"
            from noun_extraction_system import save_noun_analysis
            save_noun_analysis(doc_analysis, doc_output)
            
            return {
                "success": True,
                "document": doc_file,
                "extraction_time": extraction_time,
                "total_characters": total_chars,
                "total_words": total_words,
                "total_nouns": total_nouns,
                "unusual_nouns_count": len(noun_results.get("unusual_nouns", [])),
                "output_file": doc_output
            }
            
        except Exception as e:
            print(f"❌ Document extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    async def run_complete_benchmark(self):
        """Run complete Lightning MLX benchmark"""
        print("🚀 LIGHTNING WHISPER MLX BENCHMARK")
        print("🍎 Apple Silicon M1 Max Optimized")
        print("⚡ 10x faster than Whisper CPP claimed")
        print("=" * 80)
        
        results = []
        
        # Run validation test
        print("Phase 1: Validation Test")
        validation_result = await self.run_validation_test()
        results.append(("Validation", validation_result))
        
        # Run document noun extraction test
        print("\nPhase 2: Document Noun Extraction")
        doc_result = await self.run_document_extraction_test()
        results.append(("Document Extraction", doc_result))
        
        # Run full test only if validation passes
        if validation_result.get("success", False):
            print("\nPhase 3: Full Test")
            full_result = await self.run_full_test()
            results.append(("Full Test", full_result))
        else:
            print("\n❌ Skipping full test due to validation failure")
        
        # Save benchmark results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        benchmark_file = f"lightning_mlx_benchmark_{timestamp}.json"
        
        benchmark_data = {
            "framework": "Lightning Whisper MLX",
            "timestamp": timestamp,
            "system": "M1 Max",
            "results": {name: result for name, result in results}
        }
        
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"\n💾 Benchmark results saved to: {benchmark_file}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("🏁 LIGHTNING MLX BENCHMARK COMPLETE")
        print("=" * 80)
        
        for test_name, result in results:
            if result.get("success", False):
                ratio = result.get("processing_ratio", 0)
                speed = result.get("speed_multiplier", 0)
                print(f"✅ {test_name}: {ratio:.4f}x ratio, {speed:.1f}x speed")
            else:
                print(f"❌ {test_name}: {result.get('error', 'Unknown error')}")


async def main():
    """Main Lightning MLX execution"""
    benchmark = LightningMLXBenchmark()
    await benchmark.run_complete_benchmark()


if __name__ == "__main__":
    print("🚀 Lightning Whisper MLX Transcriber Starting...")
    asyncio.run(main())
    print("🏁 Lightning MLX processing complete!") 