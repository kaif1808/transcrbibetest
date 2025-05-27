#!/usr/bin/env python3
"""
Comprehensive Optimized Transcription Test Runner
Demonstrates all optimization features:
- Document noun extraction from inputdoc.docx for correction enhancement
- Parallelized speaker diarization with batch processing
- GPU acceleration and multi-worker processing
- Complete integration test with performance metrics
"""

import asyncio
import time
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Import the optimized system
try:
    from optimized_transcription_system import run_optimized_test, OptimizedConfig, BatchTranscriptionProcessor
    OPTIMIZED_SYSTEM_AVAILABLE = True
except ImportError:
    OPTIMIZED_SYSTEM_AVAILABLE = False
    print("❌ Optimized transcription system not available")

def print_banner():
    """Print test banner"""
    print("🚀" * 20)
    print("  OPTIMIZED TRANSCRIPTION SYSTEM")
    print("  🔧 Advanced Features Test")
    print("🚀" * 20)
    print()
    print("📋 Test Features:")
    print("   ✅ Document noun extraction (inputdoc.docx)")
    print("   ✅ Parallelized speaker diarization")
    print("   ✅ Batch audio file processing")
    print("   ✅ GPU acceleration optimization")
    print("   ✅ Multi-worker concurrent processing")
    print("   ✅ Advanced AI correction with domain vocabulary")
    print("   ✅ Performance monitoring and metrics")
    print()

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    checks = {
        "Optimized System": OPTIMIZED_SYSTEM_AVAILABLE,
        "Lightning MLX": False,
        "Transcription Corrector": False,
        "Noun Extraction": False,
        "Diarization": False,
        "Input Document": Path("inputdoc.docx").exists(),
        "Test Audio": Path("IsabelleAudio_trimmed_test.wav").exists()
    }
    
    # Test imports
    try:
        from lightning_whisper_mlx_transcriber import LightningMLXProcessor
        checks["Lightning MLX"] = True
    except ImportError:
        pass
    
    try:
        from transcription_corrector import enhance_transcription_result
        checks["Transcription Corrector"] = True
    except ImportError:
        pass
    
    try:
        from noun_extraction_system import extract_nouns_from_document
        checks["Noun Extraction"] = True
    except ImportError:
        pass
    
    try:
        from pyannote.audio import Pipeline
        checks["Diarization"] = True
    except ImportError:
        pass
    
    # Display results
    all_good = True
    for component, available in checks.items():
        status = "✅" if available else "❌"
        print(f"   {status} {component}")
        if not available:
            all_good = False
    
    if not all_good:
        print("\n⚠️  Some dependencies are missing. Install with:")
        print("   pip install -r requirements_optimized.txt")
        print("   python -m spacy download en_core_web_sm")
        return False
    
    print("✅ All dependencies available!")
    return True

async def run_comprehensive_test():
    """Run comprehensive optimization test"""
    print_banner()
    
    if not check_dependencies():
        return False
    
    print("\n🚀 Starting comprehensive optimization test...")
    print("=" * 60)
    
    # Test configuration
    audio_files = ["IsabelleAudio_trimmed_test.wav"]
    reference_doc = "inputdoc.docx"
    
    # Verify files exist
    missing_files = []
    if not Path(reference_doc).exists():
        missing_files.append(reference_doc)
    
    for audio_file in audio_files:
        if not Path(audio_file).exists():
            missing_files.append(audio_file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Run the optimized test
    try:
        start_time = time.time()
        
        print("📄 Phase 1: Document noun extraction and caching...")
        print("🎙️  Phase 2: Parallel speaker diarization...")
        print("🎤 Phase 3: Batch transcription with optimization...")
        print("🔧 Phase 4: Enhanced correction with document vocabulary...")
        print("📊 Phase 5: Performance analysis and reporting...")
        print()
        
        # Run the test
        results = await run_optimized_test(audio_files, reference_doc)
        
        total_time = time.time() - start_time
        
        if "error" in results:
            print(f"❌ Test failed: {results['error']}")
            return False
        
        # Display detailed results
        print_detailed_results(results, total_time)
        
        # Save comprehensive report
        save_comprehensive_report(results, total_time)
        
        print("\n🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_detailed_results(results: Dict[str, Any], total_time: float):
    """Print detailed test results"""
    print("\n" + "📊" * 20)
    print("  DETAILED RESULTS")
    print("📊" * 20)
    
    if "summary" not in results:
        print("⚠️  No summary available in results")
        return
    
    summary = results["summary"]
    stats = summary.get("statistics", {})
    timing = summary.get("timing", {})
    optimizations = summary.get("optimizations", {})
    doc_enhancement = summary.get("document_enhancement", {})
    quality = summary.get("quality_metrics", {})
    
    # Performance metrics
    print(f"\n⏱️  PERFORMANCE METRICS:")
    print(f"   Total execution time: {total_time:.2f}s")
    print(f"   Document processing: {timing.get('total_time', 0) - timing.get('transcription_time', 0):.2f}s")
    print(f"   Diarization time: {timing.get('diarization_time', 0):.2f}s")
    print(f"   Transcription time: {timing.get('transcription_time', 0):.2f}s")
    print(f"   Average per file: {stats.get('average_processing_time', 0):.2f}s")
    
    # Processing statistics
    print(f"\n📈 PROCESSING STATISTICS:")
    print(f"   Files processed: {stats.get('files_successful', 0)}/{stats.get('files_processed', 0)}")
    print(f"   Total transcription time: {stats.get('total_transcription_time', 0):.2f}s")
    print(f"   Estimated audio duration: {stats.get('estimated_audio_duration', 0):.2f}s")
    
    if stats.get('estimated_audio_duration', 0) > 0:
        speed_factor = stats.get('estimated_audio_duration', 0) / stats.get('total_transcription_time', 1)
        print(f"   Speed factor: {speed_factor:.1f}x real-time")
    
    # Optimization features
    print(f"\n🔧 OPTIMIZATION FEATURES:")
    print(f"   Parallel diarization: {'✅' if optimizations.get('parallel_diarization_enabled') else '❌'}")
    print(f"   GPU acceleration: {'✅' if optimizations.get('gpu_acceleration_enabled') else '❌'}")
    print(f"   Document noun enhancement: {'✅' if optimizations.get('document_noun_enhancement') else '❌'}")
    print(f"   Batch processing workers: {optimizations.get('batch_processing_workers', 0)}")
    
    # Document enhancement details
    print(f"\n📄 DOCUMENT ENHANCEMENT:")
    print(f"   Vocabulary terms cached: {doc_enhancement.get('vocabulary_terms', 0)}")
    print(f"   Technical phrases extracted: {doc_enhancement.get('technical_phrases', 0)}")
    print(f"   Domain categories identified: {doc_enhancement.get('domain_categories', 0)}")
    
    # Quality metrics
    print(f"\n🎯 QUALITY METRICS:")
    print(f"   Domain focus: {quality.get('domain_focus', 'N/A')}")
    print(f"   Noun confidence threshold: {quality.get('noun_confidence_threshold', 0)}")
    print(f"   AI correction enabled: {'✅' if quality.get('ai_correction_enabled') else '❌'}")
    
    # Individual file results
    if "results" in results:
        print(f"\n📁 FILE-BY-FILE RESULTS:")
        for audio_file, result in results["results"].items():
            filename = Path(audio_file).name
            if "error" in result:
                print(f"   ❌ {filename}: {result['error']}")
            else:
                processing_time = result.get("processing_time", 0)
                segments = len(result.get("segments", []))
                speaker_count = len(set(seg.get("speaker", "Unknown") for seg in result.get("segments", [])))
                
                print(f"   ✅ {filename}:")
                print(f"      Processing time: {processing_time:.2f}s")
                print(f"      Segments: {segments}")
                print(f"      Speakers detected: {speaker_count}")
                
                # Show optimizations applied
                opts = result.get("optimizations_applied", {})
                optimizations_list = []
                if opts.get("document_noun_correction"):
                    optimizations_list.append("doc-nouns")
                if opts.get("parallel_diarization"):
                    optimizations_list.append("parallel-diar")
                if opts.get("gpu_acceleration"):
                    optimizations_list.append("gpu")
                if opts.get("ai_correction"):
                    optimizations_list.append("ai-correct")
                if opts.get("noun_extraction"):
                    optimizations_list.append("noun-extract")
                
                if optimizations_list:
                    print(f"      Optimizations: {', '.join(optimizations_list)}")

def save_comprehensive_report(results: Dict[str, Any], total_time: float):
    """Save comprehensive test report"""
    timestamp = int(time.time())
    report_file = f"optimized_test_report_{timestamp}.json"
    
    # Create comprehensive report
    report = {
        "test_info": {
            "timestamp": timestamp,
            "test_type": "comprehensive_optimization_test",
            "total_execution_time": total_time,
            "python_version": sys.version,
            "platform": sys.platform
        },
        "results": results,
        "system_info": {
            "available_cpus": None,
            "memory_info": None
        }
    }
    
    # Add system info if available
    try:
        import psutil
        import multiprocessing as mp
        report["system_info"]["available_cpus"] = mp.cpu_count()
        memory = psutil.virtual_memory()
        report["system_info"]["memory_info"] = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent_used": memory.percent
        }
    except ImportError:
        pass
    
    # Save report
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Comprehensive report saved: {report_file}")

async def main():
    """Main test execution function"""
    print("🎯 Starting optimized transcription system test...")
    
    success = await run_comprehensive_test()
    
    if success:
        print("\n✅ All tests passed successfully!")
        print("🎉 Optimized transcription system is fully functional!")
        return 0
    else:
        print("\n❌ Tests failed!")
        print("🔧 Check dependencies and file availability.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 