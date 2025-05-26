#!/usr/bin/env python3
"""
Ultra-Fast Audio Transcription and Diarization Script
Optimized for sub-0.2x processing time ratio (processing time < 20% of audio length)
Target: OpenAI Whisper turbo quality with maximum speed
"""

import torch
import time
import asyncio
import concurrent.futures
from pathlib import Path
import sys
import math
import os
import warnings
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

# Core libraries
import torchaudio
import numpy as np
from scipy import signal
import librosa

# Optimized ML libraries
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Performance monitoring
import psutil
import threading
from queue import Queue

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

print("🚀 Ultra-Fast Diarization & Transcription System Initializing...")

@dataclass
class ProcessingConfig:
    """Configuration for ultra-fast processing"""
    audio_file: str = "IsabelleAudio.wav"
    whisper_model: str = "openai/whisper-large-v3-turbo"
    language: str = "en"
    output_dir: str = "./output_ultrafast/"
    hf_token: Optional[str] = None
    
    # Performance optimizations
    chunk_length_s: int = 15  # Smaller chunks for faster processing
    overlap_s: float = 1.0    # Overlap between chunks
    batch_size: int = 4       # Process multiple chunks in parallel
    max_workers: int = 4      # Number of parallel workers
    use_fp16: bool = True     # Use half precision for speed
    optimize_memory: bool = True
    
    # Quality vs speed trade-offs
    diarization_min_speakers: int = 1
    diarization_max_speakers: int = 10
    fast_diarization: bool = True  # Use faster diarization model


class PerformanceMonitor:
    """Monitor system performance during processing"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
        self.monitoring = False
        self.thread = None
        
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU and memory metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # GPU metrics (if available)
                gpu_usage = 0
                if torch.backends.mps.is_available():
                    # MPS doesn't provide detailed metrics, estimate from CPU
                    gpu_usage = min(cpu_percent * 1.2, 100)
                elif torch.cuda.is_available():
                    gpu_usage = torch.cuda.utilization()
                    
                self.metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'gpu_usage': gpu_usage,
                    'elapsed_time': time.time() - self.start_time
                }
                time.sleep(0.5)
            except Exception:
                pass


class AudioProcessor:
    """Optimized audio processing utilities"""
    
    @staticmethod
    def load_audio_fast(file_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load audio with optimizations"""
        try:
            # Try torchaudio first (fastest)
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
                
            return waveform.squeeze().numpy()
        except Exception:
            # Fallback to librosa
            audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
            return audio
    
    @staticmethod
    def create_chunks(audio: np.ndarray, sr: int, chunk_length_s: int, 
                     overlap_s: float) -> List[Tuple[np.ndarray, float, float]]:
        """Create overlapping audio chunks for parallel processing"""
        chunk_samples = int(chunk_length_s * sr)
        overlap_samples = int(overlap_s * sr)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        for start in range(0, len(audio), step_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Pad short chunks
            if len(chunk) < chunk_samples and len(chunk) > 0:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            if len(chunk) > 0:
                start_time = start / sr
                end_time = end / sr
                chunks.append((chunk, start_time, end_time))
                
        return chunks


class FastDiarization:
    """Optimized speaker diarization"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.pipeline = None
        self.device = self._get_optimal_device()
        
    def _get_optimal_device(self):
        """Select optimal device for diarization"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    async def initialize(self):
        """Initialize diarization pipeline asynchronously"""
        print("🎯 Loading optimized diarization pipeline...")
        
        try:
            # Use faster diarization model if available
            model_name = "pyannote/speaker-diarization-3.1"
            if self.config.fast_diarization:
                # Try segmentation + clustering approach for speed
                model_name = "pyannote/speaker-diarization-3.1"
            
            self.pipeline = DiarizationPipeline.from_pretrained(
                model_name,
                use_auth_token=self.config.hf_token or True
            )
            
            # Move to optimal device
            try:
                self.pipeline.to(self.device)
                print(f"✓ Diarization pipeline loaded on {self.device}")
            except Exception as e:
                print(f"⚠️  Device placement warning: {e}")
                self.device = torch.device("cpu")
                
        except Exception as e:
            print(f"❌ Diarization initialization failed: {e}")
            raise
    
    def process_audio(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """Process audio for speaker diarization with optimizations"""
        print("🎯 Running fast speaker diarization...")
        
        try:
            # Add progress hook for monitoring
            with ProgressHook() as hook:
                diarization = self.pipeline(
                    audio_path,
                    num_speakers=None,
                    min_speakers=self.config.diarization_min_speakers,
                    max_speakers=self.config.diarization_max_speakers,
                    hook=hook
                )
            
            # Convert to speaker timeline
            speaker_timeline = []
            if hasattr(diarization, 'itertracks'):
                # Map raw labels to consistent speaker names
                raw_labels = sorted(list(set([
                    speaker for _, _, speaker in diarization.itertracks(yield_label=True)
                ])))
                speaker_mapping = {label: f"Speaker {i + 1}" for i, label in enumerate(raw_labels)}
                
                for turn, _, raw_speaker in diarization.itertracks(yield_label=True):
                    mapped_speaker = speaker_mapping.get(raw_speaker, raw_speaker)
                    speaker_timeline.append((mapped_speaker, turn.start, turn.end))
                    
                print(f"✓ Found {len(speaker_mapping)} speakers")
            
            return speaker_timeline
            
        except Exception as e:
            print(f"❌ Diarization processing failed: {e}")
            # Return single speaker as fallback
            return [("Speaker 1", 0.0, 999999.0)]


class FastTranscription:
    """Ultra-fast transcription with parallel processing"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.pipeline = None
        self.device = self._get_optimal_device()
        
    def _get_optimal_device(self):
        """Select optimal device for transcription"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    async def initialize(self):
        """Initialize transcription model asynchronously"""
        print(f"🚀 Loading {self.config.whisper_model} for ultra-fast transcription...")
        
        try:
            # Determine optimal dtype
            torch_dtype = torch.float16 if self.config.use_fp16 and self.device.type != 'cpu' else torch.float32
            
            # Load model and processor
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.whisper_model,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                device_map="auto" if self.device.type != 'cpu' else None
            )
            
            self.processor = AutoProcessor.from_pretrained(self.config.whisper_model)
            
            # Create optimized pipeline
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=self.config.chunk_length_s,
                batch_size=self.config.batch_size,
                device=self.device,
                torch_dtype=torch_dtype,
            )
            
            print(f"✓ Transcription model loaded on {self.device} with {torch_dtype}")
            
        except Exception as e:
            print(f"❌ Transcription initialization failed: {e}")
            raise
    
    async def transcribe_chunk(self, chunk_data: Tuple[np.ndarray, float, float]) -> Dict:
        """Transcribe a single audio chunk"""
        chunk, start_time, end_time = chunk_data
        
        try:
            result = self.pipeline(
                chunk,
                generate_kwargs={"language": self.config.language, "task": "transcribe"},
                return_timestamps="word"
            )
            
            # Adjust timestamps to global time
            if "chunks" in result:
                for chunk_result in result["chunks"]:
                    if "timestamp" in chunk_result and chunk_result["timestamp"][0] is not None:
                        chunk_result["timestamp"] = (
                            start_time + chunk_result["timestamp"][0],
                            start_time + chunk_result["timestamp"][1] if chunk_result["timestamp"][1] else end_time
                        )
            
            return {
                "text": result.get("text", ""),
                "chunks": result.get("chunks", []),
                "start_time": start_time,
                "end_time": end_time
            }
            
        except Exception as e:
            print(f"⚠️  Chunk transcription warning: {e}")
            return {
                "text": "",
                "chunks": [],
                "start_time": start_time,
                "end_time": end_time
            }
    
    async def transcribe_parallel(self, audio_chunks: List[Tuple[np.ndarray, float, float]]) -> List[Dict]:
        """Transcribe multiple chunks in parallel"""
        print(f"⚡ Processing {len(audio_chunks)} chunks in parallel...")
        
        # Use ThreadPoolExecutor for CPU-bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self._transcribe_chunk_sync, chunk): i 
                for i, chunk in enumerate(audio_chunks)
            }
            
            results = [None] * len(audio_chunks)
            completed = 0
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    results[chunk_idx] = result
                    completed += 1
                    
                    # Progress update
                    progress = (completed / len(audio_chunks)) * 100
                    print(f"⚡ Transcription progress: {progress:.1f}% ({completed}/{len(audio_chunks)})")
                    
                except Exception as e:
                    print(f"⚠️  Chunk {chunk_idx} failed: {e}")
                    results[chunk_idx] = {
                        "text": "",
                        "chunks": [],
                        "start_time": audio_chunks[chunk_idx][1],
                        "end_time": audio_chunks[chunk_idx][2]
                    }
        
        return [r for r in results if r is not None]
    
    def _transcribe_chunk_sync(self, chunk_data: Tuple[np.ndarray, float, float]) -> Dict:
        """Synchronous wrapper for async transcription"""
        return asyncio.run(self.transcribe_chunk(chunk_data))


class ResultProcessor:
    """Process and merge transcription results"""
    
    @staticmethod
    def merge_results(chunk_results: List[Dict]) -> Dict:
        """Merge parallel transcription results"""
        all_text = []
        all_segments = []
        
        for result in sorted(chunk_results, key=lambda x: x["start_time"]):
            if result["text"].strip():
                all_text.append(result["text"].strip())
            
            if result["chunks"]:
                for chunk in result["chunks"]:
                    if chunk.get("timestamp") and chunk.get("text"):
                        all_segments.append({
                            "start": chunk["timestamp"][0],
                            "end": chunk["timestamp"][1],
                            "text": chunk["text"].strip()
                        })
        
        return {
            "text": " ".join(all_text),
            "segments": all_segments,
            "language": "en"  # Could be detected
        }
    
    @staticmethod
    def assign_speakers(segments: List[Dict], speaker_timeline: List[Tuple[str, float, float]]) -> List[Dict]:
        """Assign speakers to transcription segments"""
        for segment in segments:
            # Find speaker at segment midpoint
            midpoint = segment["start"] + (segment["end"] - segment["start"]) / 2
            assigned_speaker = "Unknown Speaker"
            
            for speaker, start, end in speaker_timeline:
                if start <= midpoint <= end:
                    assigned_speaker = speaker
                    break
            
            segment["speaker"] = assigned_speaker
        
        return segments


class OutputManager:
    """Handle all output file operations"""
    
    @staticmethod
    def save_results(result: Dict, audio_filename: str, output_dir: str):
        """Save transcription results in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(audio_filename).stem
        
        # Diarized transcript
        txt_path = output_path / f"{base_name}_ultrafast_diarized.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                if result.get("segments"):
                    for segment in result["segments"]:
                        speaker = segment.get("speaker", "Unknown Speaker")
                        text = segment.get("text", "").strip()
                        if text:
                            f.write(f"{speaker}: {text}\n")
                else:
                    f.write(f"Full Text: {result.get('text', 'No transcription')}\n")
            print(f"✓ Diarized transcript saved: {txt_path}")
        except Exception as e:
            print(f"❌ Error saving diarized transcript: {e}")
        
        # Plain transcript
        plain_path = output_path / f"{base_name}_ultrafast_plain.txt"
        try:
            with open(plain_path, "w", encoding="utf-8") as f:
                f.write(result.get("text", "No transcription available"))
            print(f"✓ Plain transcript saved: {plain_path}")
        except Exception as e:
            print(f"❌ Error saving plain transcript: {e}")
        
        # SRT file
        srt_path = output_path / f"{base_name}_ultrafast_diarized.srt"
        try:
            with open(srt_path, "w", encoding="utf-8") as f:
                if result.get("segments"):
                    for i, segment in enumerate(result["segments"]):
                        speaker = segment.get("speaker", "Unknown Speaker")
                        text = segment.get("text", "").strip()
                        if text:
                            start_srt = OutputManager._format_srt_time(segment["start"])
                            end_srt = OutputManager._format_srt_time(segment["end"])
                            f.write(f"{i + 1}\n")
                            f.write(f"{start_srt} --> {end_srt}\n")
                            f.write(f"({speaker}) {text}\n\n")
            print(f"✓ SRT file saved: {srt_path}")
        except Exception as e:
            print(f"❌ Error saving SRT file: {e}")
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time for SRT files"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


async def main():
    """Main ultra-fast processing pipeline"""
    
    # Configuration
    config = ProcessingConfig()
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    overall_start = time.time()
    
    try:
        print("=" * 70)
        print("🚀 ULTRA-FAST AUDIO TRANSCRIPTION & DIARIZATION")
        print("🎯 Target: Sub-0.2x processing time ratio")
        print("=" * 70)
        
        # Validate input file
        audio_path = Path(config.audio_file)
        if not audio_path.exists():
            print(f"❌ Audio file not found: {config.audio_file}")
            return
        
        # Get audio duration for ratio calculation
        try:
            import soundfile as sf
            with sf.SoundFile(config.audio_file) as f:
                audio_duration = len(f) / f.samplerate
        except:
            # Fallback duration estimation
            audio = AudioProcessor.load_audio_fast(config.audio_file)
            audio_duration = len(audio) / 16000
        
        print(f"📁 Audio file: {config.audio_file}")
        print(f"⏱️  Audio duration: {audio_duration:.1f}s")
        print(f"🎯 Target processing time: <{audio_duration * 0.2:.1f}s")
        
        # System info
        device_info = "CPU"
        if torch.backends.mps.is_available():
            device_info = "Apple Silicon GPU (MPS)"
        elif torch.cuda.is_available():
            device_info = "NVIDIA GPU (CUDA)"
        print(f"💻 Processing device: {device_info}")
        print()
        
        # Step 1: Load and chunk audio
        print("📊 Step 1: Loading and chunking audio...")
        chunk_start = time.time()
        
        audio = AudioProcessor.load_audio_fast(config.audio_file)
        audio_chunks = AudioProcessor.create_chunks(
            audio, 16000, config.chunk_length_s, config.overlap_s
        )
        
        chunk_time = time.time() - chunk_start
        print(f"✓ Created {len(audio_chunks)} chunks in {chunk_time:.2f}s")
        
        # Step 2: Initialize models in parallel
        print("\n🤖 Step 2: Initializing AI models...")
        init_start = time.time()
        
        # Initialize diarization and transcription models concurrently
        diarizer = FastDiarization(config)
        transcriber = FastTranscription(config)
        
        # Run initialization in parallel
        await asyncio.gather(
            diarizer.initialize(),
            transcriber.initialize()
        )
        
        init_time = time.time() - init_start
        print(f"✓ Models initialized in {init_time:.2f}s")
        
        # Step 3: Run diarization and transcription in parallel
        print("\n⚡ Step 3: Parallel processing...")
        parallel_start = time.time()
        
        # Start both tasks concurrently
        diarization_task = asyncio.create_task(
            asyncio.to_thread(diarizer.process_audio, config.audio_file)
        )
        transcription_task = asyncio.create_task(
            transcriber.transcribe_parallel(audio_chunks)
        )
        
        # Wait for both to complete
        speaker_timeline, chunk_results = await asyncio.gather(
            diarization_task,
            transcription_task
        )
        
        parallel_time = time.time() - parallel_start
        print(f"✓ Parallel processing completed in {parallel_time:.2f}s")
        
        # Step 4: Merge and assign speakers
        print("\n🔄 Step 4: Merging results...")
        merge_start = time.time()
        
        final_result = ResultProcessor.merge_results(chunk_results)
        if final_result["segments"]:
            final_result["segments"] = ResultProcessor.assign_speakers(
                final_result["segments"], speaker_timeline
            )
        
        merge_time = time.time() - merge_start
        print(f"✓ Results merged in {merge_time:.2f}s")
        
        # Step 5: Save outputs
        print("\n💾 Step 5: Saving outputs...")
        save_start = time.time()
        
        OutputManager.save_results(final_result, config.audio_file, config.output_dir)
        
        save_time = time.time() - save_start
        print(f"✓ Files saved in {save_time:.2f}s")
        
        # Performance summary
        total_time = time.time() - overall_start
        processing_ratio = total_time / audio_duration
        
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"📏 Audio duration: {audio_duration:.1f}s")
        print(f"📈 Processing ratio: {processing_ratio:.3f}x")
        print(f"🎯 Target achieved: {'✅ YES' if processing_ratio < 0.2 else '❌ NO'}")
        print(f"🚀 Speed improvement: {1/processing_ratio:.1f}x faster than real-time")
        
        # Detailed breakdown
        print(f"\n📋 Timing breakdown:")
        print(f"   Audio loading & chunking: {chunk_time:.2f}s")
        print(f"   Model initialization: {init_time:.2f}s")
        print(f"   Parallel processing: {parallel_time:.2f}s")
        print(f"   Result merging: {merge_time:.2f}s")
        print(f"   File saving: {save_time:.2f}s")
        
        # Quality metrics
        segments_count = len(final_result.get("segments", []))
        speakers_count = len(set(s.get("speaker", "") for s in final_result.get("segments", [])))
        
        print(f"\n📊 Output quality:")
        print(f"   Transcribed segments: {segments_count}")
        print(f"   Detected speakers: {speakers_count}")
        print(f"   Total words: ~{len(final_result.get('text', '').split())}")
        
        print(f"\n📁 Output directory: {config.output_dir}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    print("🚀 Starting Ultra-Fast Diarization & Transcription System...")
    asyncio.run(main())
    print("🏁 Processing complete!") 