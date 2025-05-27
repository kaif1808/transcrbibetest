#!/usr/bin/env python3
"""
Context Analysis Test Runner
Demonstrates the advanced context analysis capabilities including:
- Speaker identification and profiling
- Executive summary generation
- Action item extraction
- Sentiment analysis
- Conversation insights
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from optimizations.optimized_transcription_system import (
        run_optimized_test, 
        OptimizedConfig, 
        BatchTranscriptionProcessor
    )
    OPTIMIZED_SYSTEM_AVAILABLE = True
except ImportError as e:
    OPTIMIZED_SYSTEM_AVAILABLE = False
    print(f"❌ Optimized system not available: {e}")

try:
    from core.context_analyzer import AdvancedContextAnalyzer
    CONTEXT_ANALYZER_AVAILABLE = True
except ImportError as e:
    CONTEXT_ANALYZER_AVAILABLE = False
    print(f"❌ Context analyzer not available: {e}")

def print_banner():
    """Print test banner"""
    print("🔍" * 25)
    print("  ADVANCED CONTEXT ANALYSIS TEST")
    print("🔍" * 25)
    print()
    print("Features demonstrated:")
    print("• 🎯 Speaker identification and profiling")
    print("• 📝 Executive summary generation")
    print("• 📊 Conversation insights extraction")
    print("• 💭 Sentiment analysis")
    print("• 🎤 Speaker interaction analysis")
    print("• 📋 Action item extraction")
    print("• 📁 Output to 'output' folder")
    print()

async def run_context_analysis_test():
    """Run comprehensive context analysis test"""
    
    print_banner()
    
    if not OPTIMIZED_SYSTEM_AVAILABLE:
        print("❌ Cannot run test - optimized system not available")
        return
    
    # Configure for context analysis
    config = OptimizedConfig(
        audio_files=["data/input/IsabelleAudio.wav", "data/input/IsabelleAudio_trimmed_test.wav"],
        reference_document="data/input/inputdoc.docx",
        output_dir="output_context_analysis",
        
        # Enable all context analysis features
        enable_context_analysis=True,
        context_output_dir="output",
        generate_executive_summary=True,
        analyze_speaker_interactions=True,
        extract_action_items=True,
        sentiment_analysis=True,
        
        # Optimization settings
        enable_parallel_diarization=True,
        max_parallel_workers=4,
        enable_gpu_acceleration=True,
        apply_document_noun_correction=True,
        enable_ai_correction=True,
        
        # Domain settings
        domain_focus="education"
    )
    
    # Check if audio files exist
    available_files = []
    for audio_file in config.audio_files:
        if Path(audio_file).exists():
            available_files.append(audio_file)
            print(f"✅ Found audio file: {audio_file}")
        else:
            print(f"⚠️  Audio file not found: {audio_file}")
    
    if not available_files:
        print("❌ No audio files available for testing")
        return
    
    config.audio_files = available_files
    
    # Create output directory
    Path(config.context_output_dir).mkdir(exist_ok=True)
    print(f"📁 Output directory: {config.context_output_dir}/")
    
    print("\n🚀 Starting advanced context analysis test...")
    start_time = time.time()
    
    # Initialize and run processor
    processor = BatchTranscriptionProcessor(config)
    
    print("⚙️  Initializing system components...")
    await processor.initialize()
    
    # Process files
    print("🎵 Processing audio files with context analysis...")
    results = await processor.process_batch()
    
    total_time = time.time() - start_time
    
    # Display results
    print("\n" + "🎉" * 20)
    print("  CONTEXT ANALYSIS TEST COMPLETE!")
    print("🎉" * 20)
    
    if "error" in results:
        print(f"❌ Test failed: {results['error']}")
        return
    
    print_detailed_results(results, total_time)
    
    # Save comprehensive test report
    save_test_report(results, total_time, config)

def print_detailed_results(results: dict, total_time: float):
    """Print detailed analysis results"""
    
    summary = results.get("summary", {})
    file_results = results.get("results", {})
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   ⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"   📁 Files processed: {len(file_results)}")
    print(f"   ✅ Success rate: {summary.get('success_rate', 0):.1%}")
    
    # Process each file result
    for audio_file, result in file_results.items():
        print(f"\n🎵 FILE: {Path(audio_file).name}")
        print("=" * 50)
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            continue
        
        # Basic transcription info
        segments = result.get("segments", [])
        print(f"   📝 Segments: {len(segments)}")
        print(f"   📖 Words: {len(result.get('text', '').split())}")
        print(f"   ⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
        
        # Context analysis results
        context = result.get("context_analysis", {})
        if context:
            print(f"\n🔍 CONTEXT ANALYSIS RESULTS:")
            print(f"   ⏱️  Analysis time: {context.get('analysis_time', 0):.2f}s")
            
            # Speaker profiles
            speaker_profiles = context.get("speaker_profiles", {})
            if speaker_profiles:
                print(f"\n👥 SPEAKER PROFILES ({len(speaker_profiles)} identified):")
                for speaker_id, profile in speaker_profiles.items():
                    duration_min = profile.get("total_duration", 0) / 60
                    word_count = profile.get("word_count", 0)
                    speaking_rate = profile.get("speaking_rate", 0)
                    
                    print(f"   🎤 {speaker_id}:")
                    print(f"      • Duration: {duration_min:.1f} minutes")
                    print(f"      • Words: {word_count}")
                    print(f"      • Rate: {speaking_rate:.1f} words/min")
                    
                    # Key phrases
                    key_phrases = profile.get("key_phrases", [])
                    if key_phrases:
                        print(f"      • Key phrases: {', '.join(key_phrases[:3])}")
                    
                    # Emotions
                    emotions = profile.get("dominant_emotions", [])
                    if emotions:
                        print(f"      • Emotions: {', '.join(emotions[:2])}")
            
            # Conversation insights
            insights = context.get("conversation_insights", {})
            if insights:
                print(f"\n💡 CONVERSATION INSIGHTS:")
                
                topics = insights.get("main_topics", [])
                if topics:
                    print(f"   📚 Main topics: {', '.join(topics[:5])}")
                
                orgs = insights.get("organizations", [])
                if orgs:
                    print(f"   🏢 Organizations: {', '.join(orgs[:3])}")
                
                people = insights.get("mentioned_people", [])
                if people:
                    print(f"   👤 People mentioned: {', '.join(people[:3])}")
                
                actions = insights.get("action_items", [])
                if actions:
                    print(f"   📋 Action items: {len(actions)} identified")
                    for i, action in enumerate(actions[:2], 1):
                        print(f"      {i}. {action[:50]}...")
            
            # Quality metrics
            quality = context.get("quality_metrics", {})
            if quality:
                print(f"\n📊 QUALITY METRICS:")
                for metric, value in quality.items():
                    if isinstance(value, float):
                        print(f"   • {metric.replace('_', ' ').title()}: {value:.2%}")
                    else:
                        print(f"   • {metric.replace('_', ' ').title()}: {value}")
            
            # Executive summary preview
            exec_summary = context.get("executive_summary", "")
            if exec_summary:
                print(f"\n📝 EXECUTIVE SUMMARY (Preview):")
                lines = exec_summary.split('\n')[:5]
                for line in lines:
                    if line.strip():
                        print(f"   {line.strip()}")
                if len(exec_summary.split('\n')) > 5:
                    print("   ...")
        
        # Show optimization flags
        optimizations = result.get("optimizations_applied", {})
        if optimizations:
            active_opts = [opt for opt, enabled in optimizations.items() if enabled]
            print(f"\n🔧 OPTIMIZATIONS APPLIED: {', '.join(active_opts)}")

def save_test_report(results: dict, total_time: float, config: OptimizedConfig):
    """Save comprehensive test report"""
    timestamp = int(time.time())
    
    # Create test report
    test_report = {
        "test_info": {
            "test_name": "Advanced Context Analysis Test",
            "timestamp": timestamp,
            "execution_time": total_time,
            "audio_files": config.audio_files,
            "output_directory": config.context_output_dir
        },
        "configuration": {
            "context_analysis_enabled": config.enable_context_analysis,
            "executive_summary": config.generate_executive_summary,
            "speaker_interactions": config.analyze_speaker_interactions,
            "sentiment_analysis": config.sentiment_analysis,
            "action_items": config.extract_action_items,
            "parallel_workers": config.max_parallel_workers,
            "gpu_acceleration": config.enable_gpu_acceleration
        },
        "results": results,
        "performance_summary": {
            "total_execution_time": total_time,
            "files_processed": len(results.get("results", {})),
            "average_processing_time": total_time / max(1, len(results.get("results", {}))),
            "success_rate": results.get("summary", {}).get("success_rate", 0)
        }
    }
    
    # Save test report
    report_file = f"context_analysis_test_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Test report saved: {report_file}")
    print(f"📁 Context analysis outputs: {config.context_output_dir}/")
    
    # List generated files
    output_path = Path(config.context_output_dir)
    if output_path.exists():
        generated_files = list(output_path.glob("*"))
        if generated_files:
            print(f"\n📄 Generated context analysis files:")
            for file_path in sorted(generated_files):
                print(f"   • {file_path.name}")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        "Optimized System": OPTIMIZED_SYSTEM_AVAILABLE,
        "Context Analyzer": CONTEXT_ANALYZER_AVAILABLE
    }
    
    all_available = True
    for dep_name, available in dependencies.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dep_name}")
        if not available:
            all_available = False
    
    if not all_available:
        print("\n⚠️  Some dependencies are missing. Install with:")
        print("   pip install -r requirements/requirements_optimized.txt")
        print("   python -m spacy download en_core_web_sm")
    
    return all_available

async def main():
    """Main test execution"""
    print("🔍 Advanced Context Analysis Test Runner")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print()
    
    # Run the test
    await run_context_analysis_test()
    
    print("\n🎯 Test completed!")
    print("Check the 'output/' directory for detailed analysis results.")

if __name__ == "__main__":
    asyncio.run(main()) 