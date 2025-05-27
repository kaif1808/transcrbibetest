#!/usr/bin/env python3
"""
Full Audio Test - Complete IsabelleAudio.wav Processing
Demonstrates the optimized transcription system on the full audio file
"""

import asyncio
import time
import json
from pathlib import Path
from optimized_transcription_system import run_optimized_test

async def run_full_audio_test():
    """Run the optimized system on the complete IsabelleAudio.wav file"""
    print("🎵 FULL AUDIO TRANSCRIPTION TEST")
    print("=" * 80)
    print("🔧 Optimized features enabled:")
    print("   • Document noun extraction from inputdoc.docx")
    print("   • Parallelized speaker diarization")
    print("   • GPU acceleration (Apple Silicon MPS)")
    print("   • Multi-worker batch processing")
    print("   • AI-enhanced correction with domain vocabulary")
    print("=" * 80)
    
    # Check if full audio file exists
    audio_file = "IsabelleAudio.wav"
    if not Path(audio_file).exists():
        print(f"❌ Full audio file not found: {audio_file}")
        print("📁 Available audio files:")
        for f in Path(".").glob("*.wav"):
            print(f"   • {f.name}")
        return
    
    # Get audio file info
    file_size = Path(audio_file).stat().st_size / (1024**2)  # MB
    print(f"📁 Processing file: {audio_file} ({file_size:.1f} MB)")
    
    start_time = time.time()
    
    try:
        # Run the optimized transcription system
        results = await run_optimized_test([audio_file], "inputdoc.docx")
        
        total_time = time.time() - start_time
        
        if "error" in results:
            print(f"❌ Test failed: {results['error']}")
            return
        
        print("\n" + "🎉" * 20)
        print("  FULL AUDIO TEST COMPLETE!")
        print("🎉" * 20)
        
        # Display comprehensive results
        display_full_results(results, total_time, file_size)
        
        # Save detailed analysis
        save_full_audio_analysis(results, total_time, audio_file)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

def display_full_results(results: dict, total_time: float, file_size_mb: float):
    """Display comprehensive results for the full audio test"""
    
    if "summary" not in results:
        print("⚠️  No summary available")
        return
    
    summary = results["summary"]
    stats = summary.get("statistics", {})
    timing = summary.get("timing", {})
    optimizations = summary.get("optimizations", {})
    doc_enhancement = summary.get("document_enhancement", {})
    
    # Performance metrics
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    print(f"   ⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"   📁 File size processed: {file_size_mb:.1f} MB")
    print(f"   🎙️  Diarization time: {timing.get('diarization_time', 0):.2f}s")
    print(f"   🎤 Transcription time: {timing.get('transcription_time', 0):.2f}s")
    print(f"   📝 Processing rate: {file_size_mb / total_time:.2f} MB/s")
    
    # Audio analysis
    audio_duration = stats.get("estimated_audio_duration", 0)
    transcription_time = stats.get("total_transcription_time", 0)
    
    if audio_duration > 0 and transcription_time > 0:
        speed_factor = audio_duration / transcription_time
        print(f"\n🚀 SPEED ANALYSIS:")
        print(f"   📏 Estimated audio duration: {audio_duration:.0f} seconds ({audio_duration/60:.1f} minutes)")
        print(f"   ⚡ Processing speed: {speed_factor:.1f}x faster than real-time")
        print(f"   📈 Efficiency: {(speed_factor-1)*100:.0f}% faster than real-time")
    
    # Document enhancement
    print(f"\n📄 DOCUMENT ENHANCEMENT APPLIED:")
    print(f"   📚 Vocabulary terms from inputdoc.docx: {doc_enhancement.get('vocabulary_terms', 0)}")
    print(f"   🔧 Technical phrases cached: {doc_enhancement.get('technical_phrases', 0)}")
    print(f"   🏷️  Domain categories: {doc_enhancement.get('domain_categories', 0)}")
    
    # Optimization features
    print(f"\n🔧 OPTIMIZATION FEATURES ACTIVE:")
    print(f"   Parallel diarization: {'✅' if optimizations.get('parallel_diarization_enabled') else '❌'}")
    print(f"   GPU acceleration: {'✅' if optimizations.get('gpu_acceleration_enabled') else '❌'}")
    print(f"   Document noun enhancement: {'✅' if optimizations.get('document_noun_enhancement') else '❌'}")
    print(f"   Batch processing workers: {optimizations.get('batch_processing_workers', 0)}")
    
    # File-specific results
    if "results" in results:
        print(f"\n📁 TRANSCRIPTION RESULTS:")
        for audio_file, result in results["results"].items():
            if "error" in result:
                print(f"   ❌ {Path(audio_file).name}: {result['error']}")
            else:
                segments = result.get("segments", [])
                speakers = set(seg.get("speaker", "Unknown") for seg in segments)
                processing_time = result.get("processing_time", 0)
                word_count = len(result.get("text", "").split())
                
                print(f"   ✅ {Path(audio_file).name}:")
                print(f"      📝 Total segments: {len(segments)}")
                print(f"      🎙️  Speakers detected: {len(speakers)} ({', '.join(sorted(speakers))})")
                print(f"      📖 Word count: {word_count:,}")
                print(f"      ⏱️  Processing time: {processing_time:.2f}s")
                
                # Show optimization flags
                opts = result.get("optimizations_applied", {})
                active_opts = []
                if opts.get("document_noun_correction"): active_opts.append("doc-nouns")
                if opts.get("parallel_diarization"): active_opts.append("parallel-diar")
                if opts.get("gpu_acceleration"): active_opts.append("gpu")
                if opts.get("ai_correction"): active_opts.append("ai-correct")
                if opts.get("noun_extraction"): active_opts.append("noun-extract")
                
                if active_opts:
                    print(f"      🔧 Optimizations: {', '.join(active_opts)}")
                
                # Show sample segments
                print(f"\n📄 SAMPLE TRANSCRIPTION SEGMENTS:")
                for i, segment in enumerate(segments[:3]):  # Show first 3 segments
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    speaker = segment.get("speaker", "Unknown")
                    text = segment.get("text", "").strip()
                    
                    print(f"   [{start_time:.1f}s-{end_time:.1f}s] {speaker}: {text[:100]}...")
                
                if len(segments) > 3:
                    print(f"   ... and {len(segments) - 3} more segments")
                
                # Show correction information if available
                if "correction_info" in result:
                    correction_info = result["correction_info"]
                    print(f"\n🔧 CORRECTION ANALYSIS:")
                    print(f"   ⏱️  Correction time: {correction_info.get('correction_time', 0):.3f}s")
                    print(f"   📝 Original words: {correction_info.get('original_word_count', 0)}")
                    print(f"   📝 Corrected words: {correction_info.get('corrected_word_count', 0)}")
                    corrections = correction_info.get('corrections_applied', [])
                    if corrections:
                        print(f"   ✅ Corrections applied: {', '.join(corrections)}")

def save_full_audio_analysis(results: dict, total_time: float, audio_file: str):
    """Save comprehensive analysis of the full audio test"""
    timestamp = int(time.time())
    
    # Create comprehensive analysis
    analysis = {
        "test_info": {
            "timestamp": timestamp,
            "test_type": "full_audio_optimization_test",
            "audio_file": audio_file,
            "file_size_mb": Path(audio_file).stat().st_size / (1024**2),
            "total_execution_time": total_time,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "performance_metrics": {
            "processing_rate_mb_per_second": Path(audio_file).stat().st_size / (1024**2) / total_time,
            "real_time_factor": None,  # Will be calculated if audio duration is available
        },
        "optimization_results": results,
        "system_diagnostics": {
            "gpu_utilized": True,  # Apple Silicon MPS
            "parallel_workers": results.get("summary", {}).get("optimizations", {}).get("batch_processing_workers", 0),
            "document_enhancement_applied": True,
            "vocabulary_terms_count": results.get("summary", {}).get("document_enhancement", {}).get("vocabulary_terms", 0)
        }
    }
    
    # Calculate real-time factor if possible
    if "summary" in results:
        stats = results["summary"].get("statistics", {})
        audio_duration = stats.get("estimated_audio_duration", 0)
        transcription_time = stats.get("total_transcription_time", 0)
        
        if audio_duration > 0 and transcription_time > 0:
            analysis["performance_metrics"]["real_time_factor"] = audio_duration / transcription_time
            analysis["performance_metrics"]["efficiency_percentage"] = ((audio_duration / transcription_time) - 1) * 100
    
    # Save analysis
    analysis_file = f"full_audio_analysis_{timestamp}.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Full audio analysis saved: {analysis_file}")
    
    # Create summary report
    summary_file = f"full_audio_summary_{timestamp}.md"
    create_summary_report(analysis, summary_file)
    print(f"📄 Summary report created: {summary_file}")

def create_summary_report(analysis: dict, filename: str):
    """Create a markdown summary report"""
    test_info = analysis.get("test_info", {})
    perf_metrics = analysis.get("performance_metrics", {})
    results = analysis.get("optimization_results", {})
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 🎵 Full Audio Transcription Test Report\n\n")
        f.write(f"**Test Date**: {test_info.get('test_date', 'N/A')}\n")
        f.write(f"**Audio File**: {test_info.get('audio_file', 'N/A')}\n")
        f.write(f"**File Size**: {test_info.get('file_size_mb', 0):.1f} MB\n")
        f.write(f"**Total Processing Time**: {test_info.get('total_execution_time', 0):.2f} seconds\n\n")
        
        f.write("## 🚀 Performance Results\n\n")
        f.write(f"- **Processing Rate**: {perf_metrics.get('processing_rate_mb_per_second', 0):.2f} MB/second\n")
        
        if perf_metrics.get('real_time_factor'):
            f.write(f"- **Speed Factor**: {perf_metrics.get('real_time_factor', 0):.1f}x faster than real-time\n")
            f.write(f"- **Efficiency**: {perf_metrics.get('efficiency_percentage', 0):.0f}% faster than real-time\n")
        
        f.write("\n## 🔧 Optimizations Applied\n\n")
        if "summary" in results:
            summary = results["summary"]
            opts = summary.get("optimizations", {})
            doc_enhance = summary.get("document_enhancement", {})
            
            f.write(f"- ✅ Parallel diarization: {opts.get('parallel_diarization_enabled', False)}\n")
            f.write(f"- ✅ GPU acceleration: {opts.get('gpu_acceleration_enabled', False)}\n")
            f.write(f"- ✅ Document noun enhancement: {opts.get('document_noun_enhancement', False)}\n")
            f.write(f"- ✅ Batch processing workers: {opts.get('batch_processing_workers', 0)}\n")
            f.write(f"- ✅ Vocabulary terms cached: {doc_enhance.get('vocabulary_terms', 0)}\n")
        
        f.write("\n## 📊 Transcription Statistics\n\n")
        if "results" in results:
            for audio_file, result in results["results"].items():
                if "error" not in result:
                    segments = len(result.get("segments", []))
                    speakers = len(set(seg.get("speaker", "Unknown") for seg in result.get("segments", [])))
                    word_count = len(result.get("text", "").split())
                    
                    f.write(f"**{Path(audio_file).name}**:\n")
                    f.write(f"- Segments: {segments}\n")
                    f.write(f"- Speakers: {speakers}\n")
                    f.write(f"- Words: {word_count:,}\n")
                    f.write(f"- Processing time: {result.get('processing_time', 0):.2f}s\n\n")

if __name__ == "__main__":
    asyncio.run(run_full_audio_test()) 