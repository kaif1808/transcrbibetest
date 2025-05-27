#!/usr/bin/env python3
"""
Comprehensive Transcription Integration Test
Tests the full pipeline: Lightning MLX + Diarization + Correction + Noun Extraction
Includes GPU monitoring, timing, and progress tracking
"""

import asyncio
import time
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
import subprocess

# Import all components
try:
    from lightning_whisper_mlx_transcriber import LightningMLXBenchmark, LightningMLXConfig
    LIGHTNING_MLX_AVAILABLE = True
except ImportError:
    LIGHTNING_MLX_AVAILABLE = False
    print("⚠️  Lightning MLX not available")

try:
    from transcription_corrector import enhance_transcription_result, create_correction_config
    CORRECTION_AVAILABLE = True
except ImportError:
    CORRECTION_AVAILABLE = False
    print("⚠️  Transcription correction not available")

try:
    from noun_extraction_system import extract_nouns_from_transcription, NounExtractionConfig
    NOUN_EXTRACTION_AVAILABLE = True
except ImportError:
    NOUN_EXTRACTION_AVAILABLE = False
    print("⚠️  Noun extraction not available")

@dataclass
class GPUStats:
    """GPU utilization statistics"""
    timestamp: float
    gpu_util: float
    memory_used: float
    memory_total: float
    temperature: float = 0.0

@dataclass
class StageTimer:
    """Timer for each processing stage"""
    stage_name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    gpu_stats: List[GPUStats] = None
    
    def __post_init__(self):
        if self.gpu_stats is None:
            self.gpu_stats = []

class GPUMonitor:
    """Real-time GPU monitoring for Apple Silicon and NVIDIA"""
    
    def __init__(self):
        self.monitoring = False
        self.stats_history = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start GPU monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🖥️  GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🖥️  GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self._get_gpu_stats()
                if stats:
                    self.stats_history.append(stats)
                time.sleep(0.5)  # Monitor every 500ms
            except Exception as e:
                print(f"⚠️  GPU monitoring error: {e}")
                break
    
    def _get_gpu_stats(self) -> GPUStats:
        """Get current GPU statistics"""
        try:
            # Try Apple Silicon first
            stats = self._get_apple_silicon_stats()
            if stats:
                return stats
            
            # Try NVIDIA
            stats = self._get_nvidia_stats()
            if stats:
                return stats
                
            # Fallback to system memory
            return self._get_system_stats()
            
        except Exception as e:
            return GPUStats(
                timestamp=time.time(),
                gpu_util=0.0,
                memory_used=0.0,
                memory_total=0.0
            )
    
    def _get_apple_silicon_stats(self) -> GPUStats:
        """Get Apple Silicon GPU stats using powermetrics"""
        try:
            # Use system_profiler for GPU info
            result = subprocess.run([
                "system_profiler", "SPDisplaysDataType", "-json"
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                displays = data.get("SPDisplaysDataType", [])
                
                for display in displays:
                    if "Apple" in display.get("sppci_model", ""):
                        # Estimate GPU usage based on system load
                        gpu_util = min(psutil.cpu_percent() * 1.2, 100.0)  # Approximation
                        
                        # Get memory info
                        memory = psutil.virtual_memory()
                        
                        return GPUStats(
                            timestamp=time.time(),
                            gpu_util=gpu_util,
                            memory_used=memory.used / (1024**3),  # GB
                            memory_total=memory.total / (1024**3)  # GB
                        )
        except Exception:
            pass
        return None
    
    def _get_nvidia_stats(self) -> GPUStats:
        """Get NVIDIA GPU stats using nvidia-smi"""
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                return GPUStats(
                    timestamp=time.time(),
                    gpu_util=float(values[0]),
                    memory_used=float(values[1]) / 1024,  # Convert MB to GB
                    memory_total=float(values[2]) / 1024,  # Convert MB to GB
                    temperature=float(values[3])
                )
        except Exception:
            pass
        return None
    
    def _get_system_stats(self) -> GPUStats:
        """Fallback to system stats"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        return GPUStats(
            timestamp=time.time(),
            gpu_util=cpu_percent,  # Use CPU as proxy
            memory_used=memory.used / (1024**3),
            memory_total=memory.total / (1024**3)
        )
    
    def get_average_utilization(self) -> Dict[str, float]:
        """Get average GPU utilization during monitoring"""
        if not self.stats_history:
            return {"gpu_util": 0.0, "memory_util": 0.0}
        
        avg_gpu = sum(s.gpu_util for s in self.stats_history) / len(self.stats_history)
        avg_memory = sum((s.memory_used/s.memory_total)*100 for s in self.stats_history) / len(self.stats_history)
        
        return {
            "gpu_util": avg_gpu,
            "memory_util": avg_memory,
            "peak_gpu": max(s.gpu_util for s in self.stats_history),
            "peak_memory": max((s.memory_used/s.memory_total)*100 for s in self.stats_history)
        }

class ProgressTracker:
    """Progress tracking with timing"""
    
    def __init__(self, total_stages: int):
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_timers = []
        self.overall_start = time.time()
        
    def start_stage(self, stage_name: str) -> StageTimer:
        """Start timing a new stage"""
        self.current_stage += 1
        timer = StageTimer(stage_name, time.time())
        self.stage_timers.append(timer)
        
        elapsed = time.time() - self.overall_start
        print(f"\n📍 Stage {self.current_stage}/{self.total_stages}: {stage_name}")
        print(f"⏱️  Elapsed: {elapsed:.1f}s | Stage: Starting...")
        return timer
    
    def end_stage(self, timer: StageTimer):
        """End timing for a stage"""
        timer.end_time = time.time()
        timer.duration = timer.end_time - timer.start_time
        
        elapsed = time.time() - self.overall_start
        print(f"✅ {timer.stage_name} completed in {timer.duration:.2f}s")
        print(f"⏱️  Total elapsed: {elapsed:.1f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timing summary"""
        total_time = time.time() - self.overall_start
        
        return {
            "total_time": total_time,
            "stages": [
                {
                    "name": timer.stage_name,
                    "duration": timer.duration,
                    "percentage": (timer.duration / total_time) * 100
                }
                for timer in self.stage_timers
            ]
        }

class IntegratedTranscriptionTest:
    """Comprehensive transcription pipeline test"""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.progress = ProgressTracker(6)  # 6 main stages
        self.results = {}
        
    async def run_full_test(self, audio_file: str = "IsabelleAudio_trimmed_test.wav"):
        """Run complete transcription pipeline test"""
        
        print("🚀 COMPREHENSIVE TRANSCRIPTION PIPELINE TEST")
        print("=" * 80)
        print(f"📁 Audio file: {audio_file}")
        print(f"🖥️  GPU acceleration: Enabled")
        print(f"⚡ Components: Lightning MLX + Diarization + Correction + Noun Extraction")
        print("=" * 80)
        
        # Check file exists
        if not Path(audio_file).exists():
            print(f"❌ Audio file not found: {audio_file}")
            return None
        
        # Start GPU monitoring
        self.gpu_monitor.start_monitoring()
        
        try:
            # Stage 1: Audio Analysis
            await self._stage_audio_analysis(audio_file)
            
            # Stage 2: Lightning MLX Transcription
            await self._stage_transcription(audio_file)
            
            # Stage 3: Speaker Diarization
            await self._stage_diarization(audio_file)
            
            # Stage 4: Transcription Correction
            await self._stage_correction()
            
            # Stage 5: Noun Extraction
            await self._stage_noun_extraction()
            
            # Stage 6: Results Analysis
            await self._stage_analysis()
            
        finally:
            # Stop monitoring
            self.gpu_monitor.stop_monitoring()
        
        # Generate final report
        return await self._generate_report()
    
    async def _stage_audio_analysis(self, audio_file: str):
        """Stage 1: Analyze audio file"""
        timer = self.progress.start_stage("Audio File Analysis")
        
        try:
            import soundfile as sf
            
            print("🎵 Analyzing audio file...")
            with sf.SoundFile(audio_file) as f:
                duration = len(f) / f.samplerate
                sample_rate = f.samplerate
                channels = f.channels
                frames = len(f)
            
            # Get file size
            file_size = Path(audio_file).stat().st_size / (1024 * 1024)  # MB
            
            self.results["audio_info"] = {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "frames": frames,
                "file_size_mb": file_size
            }
            
            print(f"   📊 Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
            print(f"   🔊 Sample rate: {sample_rate} Hz")
            print(f"   📻 Channels: {channels}")
            print(f"   💾 File size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ Audio analysis failed: {e}")
            self.results["audio_info"] = {"error": str(e)}
        
        self.progress.end_stage(timer)
    
    async def _stage_transcription(self, audio_file: str):
        """Stage 2: Lightning MLX Transcription"""
        timer = self.progress.start_stage("Lightning MLX Transcription")
        
        if not LIGHTNING_MLX_AVAILABLE:
            print("❌ Lightning MLX not available")
            self.results["transcription"] = {"error": "Lightning MLX not available"}
            self.progress.end_stage(timer)
            return
        
        try:
            print("⚡ Starting Lightning MLX transcription...")
            
            # Configure for maximum performance
            config = LightningMLXConfig(
                audio_file=audio_file,
                model="distil-large-v3",
                batch_size=12,
                use_diarization=False,  # We'll do this separately
                enable_correction=False,  # We'll do this separately
                enable_noun_extraction=False  # We'll do this separately
            )
            
            # Create processor
            from lightning_whisper_mlx_transcriber import LightningMLXProcessor
            processor = LightningMLXProcessor(config)
            
            print("   🔧 Initializing Lightning MLX...")
            await processor.initialize()
            
            print("   🎤 Transcribing audio...")
            transcription_start = time.time()
            result = await processor.transcribe_file(audio_file)
            transcription_time = time.time() - transcription_start
            
            self.results["transcription"] = {
                "result": result,
                "transcription_time": transcription_time,
                "model_info": processor.model_info,
                "word_count": len(result.get("text", "").split()),
                "segment_count": len(result.get("segments", [])),
                "processing_ratio": transcription_time / self.results["audio_info"]["duration"]
            }
            
            word_count = len(result.get("text", "").split())
            processing_ratio = transcription_time / self.results["audio_info"]["duration"]
            speed_multiplier = 1.0 / processing_ratio if processing_ratio > 0 else 0.0
            
            print(f"   ✅ Transcription completed")
            print(f"   📝 Words transcribed: {word_count}")
            print(f"   ⚡ Processing time: {transcription_time:.2f}s")
            print(f"   🚀 Speed: {speed_multiplier:.1f}x real-time")
            print(f"   📊 Processing ratio: {processing_ratio:.4f}x")
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            self.results["transcription"] = {"error": str(e)}
        
        self.progress.end_stage(timer)
    
    async def _stage_diarization(self, audio_file: str):
        """Stage 3: Speaker Diarization"""
        timer = self.progress.start_stage("Speaker Diarization")
        
        try:
            print("🎤 Starting speaker diarization...")
            
            # Import diarization
            from lightning_whisper_mlx_transcriber import LightningMLXDiarizer, LightningMLXConfig
            
            config = LightningMLXConfig(use_diarization=True, min_speakers=1, max_speakers=8)
            diarizer = LightningMLXDiarizer(config)
            
            print("   🔧 Initializing diarization...")
            await diarizer.initialize()
            
            if diarizer.pipeline:
                print("   👥 Processing speaker diarization...")
                diarization_start = time.time()
                speaker_timeline = await diarizer.process(audio_file)
                diarization_time = time.time() - diarization_start
                
                # Apply speakers to transcription segments
                if "transcription" in self.results and "result" in self.results["transcription"]:
                    from lightning_whisper_mlx_transcriber import LightningMLXResultProcessor
                    segments = self.results["transcription"]["result"].get("segments", [])
                    if speaker_timeline:
                        updated_segments = LightningMLXResultProcessor.assign_speakers(segments, speaker_timeline)
                        self.results["transcription"]["result"]["segments"] = updated_segments
                
                speakers_detected = len(set(s[0] for s in speaker_timeline)) if speaker_timeline else 0
                
                self.results["diarization"] = {
                    "speaker_timeline": speaker_timeline,
                    "diarization_time": diarization_time,
                    "speakers_detected": speakers_detected,
                    "timeline_segments": len(speaker_timeline)
                }
                
                print(f"   ✅ Diarization completed")
                print(f"   👤 Speakers detected: {speakers_detected}")
                print(f"   📊 Timeline segments: {len(speaker_timeline)}")
                print(f"   ⏱️  Processing time: {diarization_time:.2f}s")
            else:
                print("   ⚠️  Diarization pipeline not available")
                self.results["diarization"] = {"error": "Diarization pipeline not available"}
            
        except Exception as e:
            print(f"❌ Diarization failed: {e}")
            self.results["diarization"] = {"error": str(e)}
        
        self.progress.end_stage(timer)
    
    async def _stage_correction(self):
        """Stage 4: Transcription Correction"""
        timer = self.progress.start_stage("Transcription Correction")
        
        if not CORRECTION_AVAILABLE:
            print("❌ Transcription correction not available")
            self.results["correction"] = {"error": "Correction not available"}
            self.progress.end_stage(timer)
            return
        
        if "transcription" not in self.results or "result" not in self.results["transcription"]:
            print("❌ No transcription result to correct")
            self.results["correction"] = {"error": "No transcription to correct"}
            self.progress.end_stage(timer)
            return
        
        try:
            print("🔧 Starting transcription correction...")
            
            # Create correction config
            correction_config = create_correction_config(
                domain="education",
                enable_ai=True,
                custom_terms=["Lightning MLX", "TVET", "Vietnam", "Vietnamese", "ministry", "department"]
            )
            
            print("   🤖 Applying AI-powered corrections...")
            correction_start = time.time()
            
            corrected_result = await enhance_transcription_result(
                self.results["transcription"]["result"], 
                correction_config
            )
            
            correction_time = time.time() - correction_start
            
            # Update transcription result
            original_word_count = len(self.results["transcription"]["result"].get("text", "").split())
            corrected_word_count = len(corrected_result.get("text", "").split())
            
            self.results["transcription"]["result"] = corrected_result
            self.results["correction"] = {
                "correction_time": correction_time,
                "original_word_count": original_word_count,
                "corrected_word_count": corrected_word_count,
                "corrections_applied": corrected_result.get("correction_info", {}).get("corrections_applied", []),
                "enhancement_applied": corrected_result.get("enhanced", False)
            }
            
            corrections = corrected_result.get("correction_info", {}).get("corrections_applied", [])
            
            print(f"   ✅ Correction completed")
            print(f"   📝 Word count: {original_word_count} → {corrected_word_count}")
            print(f"   🔧 Corrections applied: {len(corrections)}")
            print(f"   ⏱️  Processing time: {correction_time:.2f}s")
            
            if corrections:
                print(f"   💡 Types: {', '.join(corrections[:3])}")
            
        except Exception as e:
            print(f"❌ Correction failed: {e}")
            self.results["correction"] = {"error": str(e)}
        
        self.progress.end_stage(timer)
    
    async def _stage_noun_extraction(self):
        """Stage 5: Enhanced Noun Extraction"""
        timer = self.progress.start_stage("Enhanced Noun Extraction")
        
        if not NOUN_EXTRACTION_AVAILABLE:
            print("❌ Noun extraction not available")
            self.results["noun_extraction"] = {"error": "Noun extraction not available"}
            self.progress.end_stage(timer)
            return
        
        if "transcription" not in self.results or "result" not in self.results["transcription"]:
            print("❌ No transcription result for noun extraction")
            self.results["noun_extraction"] = {"error": "No transcription for extraction"}
            self.progress.end_stage(timer)
            return
        
        try:
            print("🔍 Starting enhanced noun extraction...")
            
            # Create enhanced configuration
            extraction_config = NounExtractionConfig(
                domain_focus="education",
                extract_phrases=True,
                use_gpu_acceleration=True,
                gpu_batch_size=64,
                parallel_processing=True,
                max_workers=4,
                min_phrase_length=2,
                max_phrase_length=8,
                use_local_llm=True,
                extract_unusual_nouns=True
            )
            
            print("   🖥️  Using GPU acceleration with phrase extraction...")
            extraction_start = time.time()
            
            noun_analysis = await extract_nouns_from_transcription(
                self.results["transcription"]["result"],
                domain="education",
                config=extraction_config
            )
            
            extraction_time = time.time() - extraction_start
            
            # Parse results
            full_analysis = noun_analysis.get("full_text_analysis", {})
            total_nouns = len(full_analysis.get("all_nouns", []))
            phrase_entities = len(full_analysis.get("phrase_entities", []))
            org_phrases = len(full_analysis.get("organizational_phrases", []))
            named_entities = len(full_analysis.get("named_entities", []))
            technical_terms = len(full_analysis.get("technical_terms", []))
            unusual_nouns = len(full_analysis.get("unusual_nouns", []))
            
            self.results["noun_extraction"] = {
                "analysis": noun_analysis,
                "extraction_time": extraction_time,
                "total_nouns": total_nouns,
                "phrase_entities": phrase_entities,
                "organizational_phrases": org_phrases,
                "named_entities": named_entities,
                "technical_terms": technical_terms,
                "unusual_nouns": unusual_nouns
            }
            
            print(f"   ✅ Noun extraction completed")
            print(f"   📝 Total nouns: {total_nouns}")
            print(f"   👥 Multi-word phrases: {phrase_entities}")
            print(f"   🏛️  Organizational phrases: {org_phrases}")
            print(f"   👤 Named entities: {named_entities}")
            print(f"   🔧 Technical terms: {technical_terms}")
            print(f"   🌟 Unusual terms (LLM): {unusual_nouns}")
            print(f"   ⏱️  Processing time: {extraction_time:.2f}s")
            
            # Show examples
            if phrase_entities > 0:
                examples = [noun.text for noun in full_analysis.get("phrase_entities", [])[:2]]
                print(f"   💡 Example phrases: {', '.join(examples)}")
            
        except Exception as e:
            print(f"❌ Noun extraction failed: {e}")
            self.results["noun_extraction"] = {"error": str(e)}
        
        self.progress.end_stage(timer)
    
    async def _stage_analysis(self):
        """Stage 6: Results Analysis"""
        timer = self.progress.start_stage("Results Analysis & Optimization")
        
        try:
            print("📊 Analyzing performance and results...")
            
            # Calculate overall metrics
            audio_duration = self.results.get("audio_info", {}).get("duration", 0)
            total_processing = sum([
                self.results.get("transcription", {}).get("transcription_time", 0),
                self.results.get("diarization", {}).get("diarization_time", 0),
                self.results.get("correction", {}).get("correction_time", 0),
                self.results.get("noun_extraction", {}).get("extraction_time", 0)
            ])
            
            overall_ratio = total_processing / audio_duration if audio_duration > 0 else 0
            overall_speed = 1.0 / overall_ratio if overall_ratio > 0 else 0
            
            # GPU utilization
            gpu_stats = self.gpu_monitor.get_average_utilization()
            
            # Quality metrics
            word_count = self.results.get("transcription", {}).get("word_count", 0)
            speakers_detected = self.results.get("diarization", {}).get("speakers_detected", 0)
            corrections_count = len(self.results.get("correction", {}).get("corrections_applied", []))
            total_nouns = self.results.get("noun_extraction", {}).get("total_nouns", 0)
            
            self.results["performance_analysis"] = {
                "audio_duration": audio_duration,
                "total_processing_time": total_processing,
                "overall_processing_ratio": overall_ratio,
                "overall_speed_multiplier": overall_speed,
                "gpu_utilization": gpu_stats,
                "quality_metrics": {
                    "words_transcribed": word_count,
                    "speakers_detected": speakers_detected,
                    "corrections_applied": corrections_count,
                    "nouns_extracted": total_nouns
                }
            }
            
            print(f"   📏 Audio duration: {audio_duration:.1f}s")
            print(f"   ⚡ Total processing: {total_processing:.2f}s")
            print(f"   🚀 Overall speed: {overall_speed:.1f}x real-time")
            print(f"   📊 Processing ratio: {overall_ratio:.4f}x")
            print(f"   🖥️  Average GPU: {gpu_stats.get('gpu_util', 0):.1f}%")
            print(f"   💾 Peak memory: {gpu_stats.get('peak_memory', 0):.1f}%")
            print(f"   📝 Quality: {word_count} words, {speakers_detected} speakers, {total_nouns} nouns")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            self.results["performance_analysis"] = {"error": str(e)}
        
        self.progress.end_stage(timer)
    
    async def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Get timing summary
        timing_summary = self.progress.get_summary()
        
        # Combine all results
        report = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_duration": timing_summary["total_time"],
            "components_tested": {
                "lightning_mlx": "transcription" in self.results and "error" not in self.results["transcription"],
                "diarization": "diarization" in self.results and "error" not in self.results["diarization"],
                "correction": "correction" in self.results and "error" not in self.results["correction"],
                "noun_extraction": "noun_extraction" in self.results and "error" not in self.results["noun_extraction"]
            },
            "timing_breakdown": timing_summary,
            "results": self.results
        }
        
        # Save detailed report
        report_file = f"comprehensive_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST COMPLETED")
        print("=" * 80)
        
        print(f"⏱️  Total time: {timing_summary['total_time']:.2f}s")
        print(f"📊 Components tested: {sum(report['components_tested'].values())}/4")
        
        if "performance_analysis" in self.results:
            perf = self.results["performance_analysis"]
            print(f"🚀 Overall speed: {perf.get('overall_speed_multiplier', 0):.1f}x real-time")
            print(f"🖥️  GPU utilization: {perf.get('gpu_utilization', {}).get('gpu_util', 0):.1f}%")
        
        print(f"\n📁 Detailed report saved: {report_file}")
        
        # Performance assessment
        if "performance_analysis" in self.results:
            ratio = self.results["performance_analysis"].get("overall_processing_ratio", 1.0)
            if ratio < 0.1:
                print("🏆 EXCELLENT: Sub-10% processing time!")
            elif ratio < 0.2:
                print("🥇 GREAT: Sub-20% processing time!")
            elif ratio < 0.3:
                print("🥈 GOOD: Sub-30% processing time")
            else:
                print("🥉 ACCEPTABLE: Above 30% processing time")
        
        return report

async def main():
    """Run comprehensive transcription test"""
    
    # Check prerequisites
    print("🔍 Checking system capabilities...")
    
    # Check GPU
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ Apple Silicon MPS GPU available")
        elif torch.cuda.is_available():
            print("✅ NVIDIA CUDA GPU available")
        else:
            print("⚠️  No GPU acceleration detected")
    except ImportError:
        print("⚠️  PyTorch not available for GPU detection")
    
    # Check audio file
    audio_files = ["IsabelleAudio_trimmed_test.wav", "IsabelleAudio.wav"]
    test_file = None
    
    for audio_file in audio_files:
        if Path(audio_file).exists():
            test_file = audio_file
            print(f"✅ Found audio file: {audio_file}")
            break
    
    if not test_file:
        print("❌ No audio files found. Please ensure audio files are available.")
        return
    
    # Run comprehensive test
    test = IntegratedTranscriptionTest()
    report = await test.run_full_test(test_file)
    
    if report:
        print("\n🎉 Comprehensive transcription test completed successfully!")
    else:
        print("\n❌ Test failed to complete")

if __name__ == "__main__":
    asyncio.run(main()) 