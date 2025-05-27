#!/usr/bin/env python3
"""
Optimized Transcription System
Advanced system combining:
- Noun extraction from inputdoc.docx for transcription correction enhancement
- Parallelized speaker diarization with batch processing  
- GPU optimization and concurrent processing
- Multi-audio file batch processing capabilities
"""

import asyncio
import time
import json
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging

# Core imports
try:
    from ..core.lightning_whisper_mlx_transcriber import LightningMLXProcessor, LightningMLXConfig, LightningMLXDiarizer
    LIGHTNING_MLX_AVAILABLE = True
except ImportError:
    LIGHTNING_MLX_AVAILABLE = False
    print("⚠️  Lightning MLX not available")

try:
    from ..core.transcription_corrector import enhance_transcription_result, create_correction_config
    CORRECTION_AVAILABLE = True
except ImportError:
    CORRECTION_AVAILABLE = False
    print("⚠️  Transcription correction not available")

try:
    from ..core.noun_extraction_system import (
        extract_nouns_from_document, 
        extract_nouns_from_transcription,
        NounExtractionConfig,
        DocumentNounExtractor
    )
    NOUN_EXTRACTION_AVAILABLE = True
except ImportError:
    NOUN_EXTRACTION_AVAILABLE = False
    print("⚠️  Noun extraction not available")

try:
    from ..core.context_analyzer import AdvancedContextAnalyzer, ContextAnalysisResult
    CONTEXT_ANALYSIS_AVAILABLE = True
except ImportError:
    CONTEXT_ANALYSIS_AVAILABLE = False
    print("⚠️  Context analysis not available")

# Advanced diarization imports
try:
    from pyannote.audio import Pipeline
    import torch
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("⚠️  Advanced diarization not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedConfig:
    """Configuration for optimized transcription system"""
    # Input files
    audio_files: List[str] = field(default_factory=list)
    reference_document: str = "inputdoc.docx"
    output_dir: str = "./output_optimized/"
    
    # Processing optimization
    max_parallel_workers: int = 4
    enable_gpu_acceleration: bool = True
    batch_size: int = 8
    chunk_parallel_limit: int = 6
    
    # Diarization optimization
    enable_parallel_diarization: bool = True
    diarization_batch_size: int = 4
    max_speakers: int = 8
    min_speakers: int = 1
    
    # Noun extraction enhancement
    apply_document_noun_correction: bool = True
    extract_document_phrases: bool = True
    use_llm_for_unusual_nouns: bool = True
    noun_confidence_threshold: float = 0.8
    
    # Context analysis settings
    enable_context_analysis: bool = True
    context_output_dir: str = "output"
    generate_executive_summary: bool = True
    analyze_speaker_interactions: bool = True
    extract_action_items: bool = True
    sentiment_analysis: bool = True
    
    # Quality settings
    domain_focus: str = "education"
    enable_ai_correction: bool = True
    preserve_timestamps: bool = True
    
    def __post_init__(self):
        if not self.audio_files:
            self.audio_files = ["IsabelleAudio_trimmed_test.wav"]


class DocumentNounCache:
    """Caches document noun extraction for reuse across audio files"""
    
    def __init__(self):
        self.cached_nouns = None
        self.custom_vocabulary = set()
        self.domain_terms = {}
        self.technical_phrases = set()
        
    async def load_document_nouns(self, document_path: str, config: OptimizedConfig) -> Dict[str, Any]:
        """Extract and cache nouns from reference document"""
        if self.cached_nouns is not None:
            return self.cached_nouns
            
        print(f"📄 Loading reference document: {document_path}")
        
        if not Path(document_path).exists():
            print(f"⚠️  Reference document not found: {document_path}")
            return {}
        
        try:
            # Enhanced noun extraction configuration
            noun_config = NounExtractionConfig(
                domain_focus=config.domain_focus,
                extract_phrases=config.extract_document_phrases,
                use_local_llm=config.use_llm_for_unusual_nouns,
                llm_provider="ollama",
                llm_model="llama3.2",
                extract_unusual_nouns=True,
                use_gpu_acceleration=config.enable_gpu_acceleration,
                parallel_processing=True,
                max_workers=config.max_parallel_workers,
                min_frequency=2,
                confidence_threshold=config.noun_confidence_threshold
            )
            
            # Extract nouns from document
            start_time = time.time()
            document_analysis = await extract_nouns_from_document(
                document_path, 
                config.domain_focus, 
                noun_config
            )
            extraction_time = time.time() - start_time
            
            if "error" in document_analysis:
                print(f"❌ Document extraction error: {document_analysis['error']}")
                return {}
            
            # Process and cache results
            noun_results = document_analysis.get("noun_analysis", {})
            self._build_vocabulary_cache(noun_results)
            self.cached_nouns = document_analysis
            
            print(f"✅ Document nouns extracted in {extraction_time:.2f}s")
            print(f"🔍 Cached {len(self.custom_vocabulary)} vocabulary terms")
            print(f"🔍 Cached {len(self.technical_phrases)} technical phrases")
            
            return document_analysis
            
        except Exception as e:
            print(f"❌ Error extracting document nouns: {e}")
            return {}
    
    def _build_vocabulary_cache(self, noun_results: Dict[str, Any]):
        """Build vocabulary caches from extracted nouns"""
        for category, nouns in noun_results.items():
            if not isinstance(nouns, list):
                continue
                
            for noun in nouns:
                if hasattr(noun, 'text'):
                    text = noun.text.lower()
                    self.custom_vocabulary.add(text)
                    
                    # Cache technical terms
                    if category in ['technical_terms', 'unusual_nouns']:
                        self.technical_phrases.add(noun.text)
                    
                    # Cache domain-specific terms
                    if hasattr(noun, 'domain') and noun.domain:
                        if noun.domain not in self.domain_terms:
                            self.domain_terms[noun.domain] = set()
                        self.domain_terms[noun.domain].add(noun.text)
    
    def get_correction_vocabulary(self) -> List[str]:
        """Get vocabulary for transcription correction"""
        return list(self.custom_vocabulary)
    
    def get_technical_phrases(self) -> List[str]:
        """Get technical phrases for correction"""
        return list(self.technical_phrases)


class ParallelDiarizer:
    """Parallelized speaker diarization processor"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.pipeline = None
        self.device = None
        
    async def initialize(self):
        """Initialize diarization pipeline with GPU optimization"""
        if not DIARIZATION_AVAILABLE:
            raise ImportError("Diarization dependencies not available")
        
        print("🎙️  Initializing parallel diarization pipeline...")
        
        # Determine optimal device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✅ Using CUDA GPU for diarization")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ Using Apple Silicon GPU (MPS) for diarization")
        else:
            self.device = torch.device("cpu")
            print("✅ Using CPU for diarization")
        
        # Load diarization pipeline
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=True
            )
            self.pipeline.to(self.device)
            print("✅ Diarization pipeline loaded successfully")
        except Exception as e:
            print(f"⚠️  Diarization pipeline load failed: {e}")
            raise
    
    async def process_batch(self, audio_files: List[str]) -> Dict[str, List[Tuple[str, float, float]]]:
        """Process multiple audio files for diarization in parallel"""
        print(f"🎙️  Processing {len(audio_files)} files for diarization...")
        
        results = {}
        
        # Process files in parallel batches
        batch_size = min(self.config.diarization_batch_size, len(audio_files))
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Create tasks for each audio file
            future_to_file = {
                executor.submit(self._process_single_file, audio_file): audio_file
                for audio_file in audio_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    speaker_timeline = future.result()
                    results[audio_file] = speaker_timeline
                    print(f"✅ Diarization completed for: {Path(audio_file).name}")
                except Exception as e:
                    print(f"❌ Diarization failed for {audio_file}: {e}")
                    results[audio_file] = []
        
        return results
    
    def _process_single_file(self, audio_file: str) -> List[Tuple[str, float, float]]:
        """Process single audio file for diarization"""
        try:
            if not Path(audio_file).exists():
                print(f"⚠️  Audio file not found: {audio_file}")
                return []
            
            # Run diarization
            diarization_result = self.pipeline(
                audio_file,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
            
            # Process results
            speaker_timeline = []
            if diarization_result and hasattr(diarization_result, 'itertracks'):
                # Map raw labels to speaker names
                raw_labels = sorted(list(set([
                    speaker for _, _, speaker in diarization_result.itertracks(yield_label=True)
                ])))
                speaker_mapping = {label: f"Speaker {i + 1}" for i, label in enumerate(raw_labels)}
                
                # Extract timeline
                for turn, _, raw_speaker_label in diarization_result.itertracks(yield_label=True):
                    mapped_speaker = speaker_mapping.get(raw_speaker_label, raw_speaker_label)
                    speaker_timeline.append((mapped_speaker, turn.start, turn.end))
            
            return speaker_timeline
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            return []


class BatchTranscriptionProcessor:
    """Batch processor for multiple audio files with optimization"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.document_cache = DocumentNounCache()
        self.diarizer = ParallelDiarizer(config)
        self.transcriber = None
        self.context_analyzer = None
        
    async def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing optimized transcription system...")
        
        # Initialize document noun cache
        await self.document_cache.load_document_nouns(
            self.config.reference_document, 
            self.config
        )
        
        # Initialize context analyzer
        if CONTEXT_ANALYSIS_AVAILABLE and self.config.enable_context_analysis:
            print("🔍 Initializing context analyzer...")
            self.context_analyzer = AdvancedContextAnalyzer()
            print("✅ Context analyzer initialized")
        
        # Initialize diarization
        if self.config.enable_parallel_diarization:
            await self.diarizer.initialize()
        
        # Initialize transcriber
        if LIGHTNING_MLX_AVAILABLE:
            transcriber_config = LightningMLXConfig(
                batch_size=self.config.batch_size,
                use_diarization=False,  # We handle diarization separately
                enable_correction=True,
                enable_noun_extraction=True,
                correction_domain=self.config.domain_focus,
                noun_extraction_domain=self.config.domain_focus
            )
            self.transcriber = LightningMLXProcessor(transcriber_config)
            await self.transcriber.initialize()
        
        print("✅ System initialization complete")
    
    async def process_batch(self) -> Dict[str, Any]:
        """Process all audio files in batch with full optimization"""
        results = {}
        start_time = time.time()
        
        print(f"🎵 Starting batch processing of {len(self.config.audio_files)} files")
        print("=" * 60)
        
        # Verify all files exist
        valid_files = []
        for audio_file in self.config.audio_files:
            if Path(audio_file).exists():
                valid_files.append(audio_file)
            else:
                print(f"⚠️  File not found: {audio_file}")
        
        if not valid_files:
            return {"error": "No valid audio files found"}
        
        # Step 1: Parallel diarization for all files
        diarization_results = {}
        if self.config.enable_parallel_diarization and self.diarizer.pipeline:
            print("🎙️  STEP 1: Parallel Speaker Diarization")
            diarization_start = time.time()
            diarization_results = await self.diarizer.process_batch(valid_files)
            diarization_time = time.time() - diarization_start
            print(f"✅ Batch diarization completed in {diarization_time:.2f}s")
        
        # Step 2: Parallel transcription with optimization
        print("🎤 STEP 2: Optimized Transcription Processing")
        transcription_start = time.time()
        
        # Process files in parallel
        max_workers = min(self.config.max_parallel_workers, len(valid_files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self._process_single_audio_optimized, 
                    audio_file, 
                    diarization_results.get(audio_file, [])
                ): audio_file
                for audio_file in valid_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    result = future.result()
                    results[audio_file] = result
                    print(f"✅ Completed processing: {Path(audio_file).name}")
                except Exception as e:
                    print(f"❌ Error processing {audio_file}: {e}")
                    results[audio_file] = {"error": str(e)}
        
        transcription_time = time.time() - transcription_start
        total_time = time.time() - start_time
        
        # Create comprehensive summary
        summary = self._create_batch_summary(results, {
            "total_time": total_time,
            "diarization_time": diarization_results and diarization_time or 0,
            "transcription_time": transcription_time,
            "files_processed": len(results),
            "files_successful": len([r for r in results.values() if "error" not in r])
        })
        
        # Save results
        self._save_batch_results(results, summary)
        
        print("🎉 BATCH PROCESSING COMPLETE")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Files processed: {len(results)}")
        
        return {"results": results, "summary": summary}
    
    def _process_single_audio_optimized(self, audio_file: str, speaker_timeline: List[Tuple[str, float, float]]) -> Dict[str, Any]:
        """Process single audio file with all optimizations applied"""
        try:
            file_start = time.time()
            print(f"🎵 Processing: {Path(audio_file).name}")
            
            # Step 1: Transcription
            if self.transcriber:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                transcription_result = loop.run_until_complete(
                    self.transcriber.transcribe_file(audio_file)
                )
                loop.close()
            else:
                transcription_result = {"text": "", "segments": []}
            
            # Step 2: Apply speaker diarization
            if speaker_timeline and transcription_result.get("segments"):
                transcription_result["segments"] = self._assign_speakers_optimized(
                    transcription_result["segments"], 
                    speaker_timeline
                )
            
            # Step 3: Enhanced correction with document nouns
            if CORRECTION_AVAILABLE and self.config.apply_document_noun_correction:
                correction_config = create_correction_config(
                    domain=self.config.domain_focus,
                    enable_ai=self.config.enable_ai_correction,
                    custom_terms=self.document_cache.get_correction_vocabulary()
                )
                
                # Add technical phrases to correction
                correction_config.custom_vocabulary.extend(
                    self.document_cache.get_technical_phrases()
                )
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                transcription_result = loop.run_until_complete(
                    enhance_transcription_result(transcription_result, correction_config)
                )
                loop.close()
            
            # Step 4: Noun extraction for transcription
            if NOUN_EXTRACTION_AVAILABLE:
                noun_config = NounExtractionConfig(
                    domain_focus=self.config.domain_focus,
                    use_gpu_acceleration=self.config.enable_gpu_acceleration,
                    confidence_threshold=self.config.noun_confidence_threshold
                )
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                noun_analysis = loop.run_until_complete(
                    extract_nouns_from_transcription(transcription_result, self.config.domain_focus, noun_config)
                )
                loop.close()
                
                transcription_result["noun_analysis"] = noun_analysis
            
            # Step 5: Advanced context analysis
            if self.context_analyzer and self.config.enable_context_analysis:
                print(f"🔍 Performing context analysis for {Path(audio_file).name}")
                context_start = time.time()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                context_analysis = loop.run_until_complete(
                    self.context_analyzer.analyze_transcription(
                        transcription_result, 
                        self.config.context_output_dir
                    )
                )
                loop.close()
                
                context_time = time.time() - context_start
                transcription_result["context_analysis"] = {
                    "executive_summary": context_analysis.executive_summary,
                    "speaker_profiles": {
                        speaker_id: {
                            "total_duration": profile.total_duration,
                            "word_count": profile.word_count,
                            "speaking_rate": profile.speaking_rate,
                            "key_phrases": profile.key_phrases[:5],
                            "sentiment_scores": profile.sentiment_scores,
                            "dominant_emotions": profile.dominant_emotions
                        }
                        for speaker_id, profile in context_analysis.speaker_profiles.items()
                    },
                    "conversation_insights": {
                        "main_topics": context_analysis.conversation_insights.main_topics[:10],
                        "organizations": context_analysis.conversation_insights.organizations,
                        "mentioned_people": context_analysis.conversation_insights.mentioned_people,
                        "action_items": context_analysis.conversation_insights.action_items[:5],
                        "technical_terms": context_analysis.conversation_insights.technical_terms[:10]
                    },
                    "quality_metrics": context_analysis.quality_metrics,
                    "analysis_time": context_time
                }
                print(f"✅ Context analysis completed in {context_time:.2f}s")
            
            processing_time = time.time() - file_start
            transcription_result["processing_time"] = processing_time
            transcription_result["optimizations_applied"] = {
                "document_noun_correction": self.config.apply_document_noun_correction,
                "parallel_diarization": len(speaker_timeline) > 0,
                "gpu_acceleration": self.config.enable_gpu_acceleration,
                "context_analysis": self.config.enable_context_analysis,
                "ai_correction": self.config.enable_ai_correction,
                "noun_extraction": NOUN_EXTRACTION_AVAILABLE
            }
            
            return transcription_result
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}
    
    def _assign_speakers_optimized(self, segments: List[Dict], speaker_timeline: List[Tuple[str, float, float]]) -> List[Dict]:
        """Optimized speaker assignment algorithm"""
        for segment in segments:
            segment_midpoint = segment.get('start', 0) + (segment.get('end', 0) - segment.get('start', 0)) / 2
            assigned_speaker = "Unknown_Speaker"
            
            # Find best matching speaker segment
            best_overlap = 0
            for speaker_label, start_time, end_time in speaker_timeline:
                # Calculate overlap
                overlap_start = max(segment.get('start', 0), start_time)
                overlap_end = min(segment.get('end', 0), end_time)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    assigned_speaker = speaker_label
            
            segment['speaker'] = assigned_speaker
        
        return segments
    
    def _create_batch_summary(self, results: Dict[str, Any], timing: Dict[str, float]) -> Dict[str, Any]:
        """Create comprehensive batch processing summary"""
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        # Calculate statistics
        total_transcription_time = sum(
            result.get("processing_time", 0) for result in successful_results.values()
        )
        
        total_audio_duration = sum(
            len(result.get("segments", [])) * 30  # Estimate from segment count
            for result in successful_results.values()
        )
        
        # Document noun enhancement stats
        document_stats = {}
        if self.document_cache.cached_nouns:
            document_stats = {
                "vocabulary_terms": len(self.document_cache.custom_vocabulary),
                "technical_phrases": len(self.document_cache.technical_phrases),
                "domain_categories": len(self.document_cache.domain_terms)
            }
        
        return {
            "timing": timing,
            "statistics": {
                "files_processed": len(results),
                "files_successful": len(successful_results),
                "total_transcription_time": total_transcription_time,
                "estimated_audio_duration": total_audio_duration,
                "average_processing_time": total_transcription_time / max(len(successful_results), 1)
            },
            "optimizations": {
                "parallel_diarization_enabled": self.config.enable_parallel_diarization,
                "gpu_acceleration_enabled": self.config.enable_gpu_acceleration,
                "document_noun_enhancement": self.config.apply_document_noun_correction,
                "batch_processing_workers": self.config.max_parallel_workers
            },
            "document_enhancement": document_stats,
            "quality_metrics": {
                "domain_focus": self.config.domain_focus,
                "noun_confidence_threshold": self.config.noun_confidence_threshold,
                "ai_correction_enabled": self.config.enable_ai_correction
            }
        }
    
    def _save_batch_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Save batch processing results"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save individual results
        for audio_file, result in results.items():
            filename = Path(audio_file).stem
            output_file = output_dir / f"{filename}_optimized_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        # Save summary
        summary_file = output_dir / f"batch_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Results saved to: {output_dir}")


async def run_optimized_test(audio_files: List[str] = None, reference_doc: str = "inputdoc.docx"):
    """Run the complete optimized transcription test"""
    print("🚀 OPTIMIZED TRANSCRIPTION SYSTEM TEST")
    print("=" * 60)
    print("🔧 Optimizations enabled:")
    print("   • Document noun extraction for correction enhancement")
    print("   • Parallelized speaker diarization")
    print("   • Batch processing with GPU acceleration")
    print("   • Multi-worker concurrent processing")
    print("   • Advanced AI correction with domain vocabulary")
    print("=" * 60)
    
    # Default to test file if none provided
    if not audio_files:
        audio_files = ["IsabelleAudio_trimmed_test.wav"]
    
    # Create optimized configuration
    config = OptimizedConfig(
        audio_files=audio_files,
        reference_document=reference_doc,
        max_parallel_workers=mp.cpu_count(),
        enable_gpu_acceleration=True,
        batch_size=12,
        enable_parallel_diarization=True,
        diarization_batch_size=4,
        apply_document_noun_correction=True,
        extract_document_phrases=True,
        use_llm_for_unusual_nouns=True,
        domain_focus="education",
        enable_ai_correction=True
    )
    
    # Initialize and run batch processor
    processor = BatchTranscriptionProcessor(config)
    
    try:
        await processor.initialize()
        results = await processor.process_batch()
        
        print("\n🎉 OPTIMIZATION TEST COMPLETE!")
        
        if "summary" in results:
            summary = results["summary"]
            stats = summary.get("statistics", {})
            timing = summary.get("timing", {})
            doc_enhancement = summary.get("document_enhancement", {})
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   ⏱️  Total time: {timing.get('total_time', 0):.2f}s")
            print(f"   🎙️  Diarization time: {timing.get('diarization_time', 0):.2f}s")
            print(f"   🎤 Transcription time: {timing.get('transcription_time', 0):.2f}s")
            print(f"   📁 Files processed: {stats.get('files_successful', 0)}/{stats.get('files_processed', 0)}")
            print(f"   ⚡ Avg processing time: {stats.get('average_processing_time', 0):.2f}s/file")
            
            print(f"\n📄 DOCUMENT ENHANCEMENT:")
            print(f"   📚 Vocabulary terms cached: {doc_enhancement.get('vocabulary_terms', 0)}")
            print(f"   🔧 Technical phrases: {doc_enhancement.get('technical_phrases', 0)}")
            print(f"   🏷️  Domain categories: {doc_enhancement.get('domain_categories', 0)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(run_optimized_test()) 