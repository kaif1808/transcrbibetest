#!/usr/bin/env python3
"""
Context Analysis Demo
Demonstrates the advanced context analysis capabilities using sample data
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from core.context_analyzer import AdvancedContextAnalyzer, ContextAnalysisResult
    CONTEXT_ANALYZER_AVAILABLE = True
except ImportError as e:
    CONTEXT_ANALYZER_AVAILABLE = False
    print(f"❌ Context analyzer not available: {e}")

def create_sample_transcription_data():
    """Create sample transcription data for demonstration"""
    return {
        "text": """Hello everyone, welcome to today's educational meeting. I'm Dr. Sarah Johnson, the lead researcher on this project. We're here to discuss the findings from our recent study on student engagement in online learning environments.

As you can see from the data we collected over the past semester, there's been a significant increase in student participation when we implemented interactive elements. John, could you share your observations from the computer science department?

Thank you, Sarah. From what we've observed in CS101 and CS102, students are more engaged when they have hands-on coding exercises. We need to implement more interactive labs and perhaps consider using VR technology for complex algorithm visualization.

That's an excellent point, John. Maria, what about from the psychology perspective? How are students responding emotionally to these changes?

Well, Dr. Johnson, our psychological assessments show that students report feeling more confident and less anxious about the material when they can interact with it directly. We should definitely continue this approach and maybe expand it to other departments.

I agree completely. Let's make this our action item for next quarter: we need to develop a comprehensive interactive learning framework that we can pilot in three departments - computer science, psychology, and mathematics. We should have a follow-up meeting in two weeks to discuss implementation details.

Before we wrap up, I want to mention that the National Education Foundation has expressed interest in our work. They might provide additional funding for expanding this research. This could be a great opportunity for our university.

That's fantastic news! We should prepare a detailed proposal for them. I'll coordinate with the grants office to make sure we have all the necessary documentation ready.""",
        
        "segments": [
            {
                "text": "Hello everyone, welcome to today's educational meeting. I'm Dr. Sarah Johnson, the lead researcher on this project.",
                "start": 0.0,
                "end": 6.5,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "We're here to discuss the findings from our recent study on student engagement in online learning environments.",
                "start": 7.0,
                "end": 13.2,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "As you can see from the data we collected over the past semester, there's been a significant increase in student participation when we implemented interactive elements.",
                "start": 14.0,
                "end": 22.5,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "John, could you share your observations from the computer science department?",
                "start": 23.0,
                "end": 27.8,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "Thank you, Sarah. From what we've observed in CS101 and CS102, students are more engaged when they have hands-on coding exercises.",
                "start": 28.5,
                "end": 36.2,
                "speaker": "Professor John Miller"
            },
            {
                "text": "We need to implement more interactive labs and perhaps consider using VR technology for complex algorithm visualization.",
                "start": 37.0,
                "end": 44.5,
                "speaker": "Professor John Miller"
            },
            {
                "text": "That's an excellent point, John. Maria, what about from the psychology perspective?",
                "start": 45.0,
                "end": 50.2,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "How are students responding emotionally to these changes?",
                "start": 50.5,
                "end": 54.8,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "Well, Dr. Johnson, our psychological assessments show that students report feeling more confident and less anxious about the material when they can interact with it directly.",
                "start": 55.5,
                "end": 65.2,
                "speaker": "Dr. Maria Rodriguez"
            },
            {
                "text": "We should definitely continue this approach and maybe expand it to other departments.",
                "start": 66.0,
                "end": 71.5,
                "speaker": "Dr. Maria Rodriguez"
            },
            {
                "text": "I agree completely. Let's make this our action item for next quarter: we need to develop a comprehensive interactive learning framework that we can pilot in three departments.",
                "start": 72.0,
                "end": 82.5,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "Computer science, psychology, and mathematics. We should have a follow-up meeting in two weeks to discuss implementation details.",
                "start": 83.0,
                "end": 91.2,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "Before we wrap up, I want to mention that the National Education Foundation has expressed interest in our work.",
                "start": 92.0,
                "end": 99.5,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "They might provide additional funding for expanding this research. This could be a great opportunity for our university.",
                "start": 100.0,
                "end": 108.2,
                "speaker": "Dr. Sarah Johnson"
            },
            {
                "text": "That's fantastic news! We should prepare a detailed proposal for them.",
                "start": 109.0,
                "end": 114.5,
                "speaker": "Professor John Miller"
            },
            {
                "text": "I'll coordinate with the grants office to make sure we have all the necessary documentation ready.",
                "start": 115.0,
                "end": 121.8,
                "speaker": "Professor John Miller"
            }
        ]
    }

async def run_context_analysis_demo():
    """Run the context analysis demonstration"""
    
    print("🔍" * 25)
    print("  CONTEXT ANALYSIS DEMONSTRATION")
    print("🔍" * 25)
    print()
    
    if not CONTEXT_ANALYZER_AVAILABLE:
        print("❌ Context analyzer not available")
        return
    
    # Create output directory
    output_dir = "output"
    Path(output_dir).mkdir(exist_ok=True)
    
    print("🎯 Initializing advanced context analyzer...")
    analyzer = AdvancedContextAnalyzer()
    
    print("📄 Creating sample educational meeting transcription...")
    sample_data = create_sample_transcription_data()
    
    print("🔍 Performing comprehensive context analysis...")
    start_time = time.time()
    
    # Run the analysis
    result = await analyzer.analyze_transcription(sample_data, output_dir)
    
    analysis_time = time.time() - start_time
    
    print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
    print()
    
    # Display results
    display_analysis_results(result)
    
    # Show generated files
    show_generated_files(output_dir)

def display_analysis_results(result: ContextAnalysisResult):
    """Display the analysis results in a formatted way"""
    
    print("📊 ANALYSIS RESULTS")
    print("=" * 50)
    
    # Executive Summary
    print("\n📝 EXECUTIVE SUMMARY:")
    print("-" * 30)
    print(result.executive_summary)
    
    # Speaker Profiles
    print(f"\n👥 SPEAKER PROFILES ({len(result.speaker_profiles)} identified):")
    print("-" * 40)
    
    for speaker_id, profile in result.speaker_profiles.items():
        duration_min = profile.total_duration / 60
        print(f"\n🎤 {speaker_id}:")
        print(f"   • Speaking time: {duration_min:.1f} minutes")
        print(f"   • Word count: {profile.word_count}")
        print(f"   • Speaking rate: {profile.speaking_rate:.1f} words/minute")
        print(f"   • Segments: {profile.segment_count}")
        
        if profile.key_phrases:
            print(f"   • Key phrases: {', '.join(profile.key_phrases[:3])}")
        
        if profile.expertise_areas:
            print(f"   • Expertise areas: {', '.join(profile.expertise_areas[:3])}")
        
        if profile.dominant_emotions:
            print(f"   • Dominant emotions: {', '.join(profile.dominant_emotions)}")
    
    # Conversation Insights
    print(f"\n💡 CONVERSATION INSIGHTS:")
    print("-" * 30)
    
    insights = result.conversation_insights
    
    if insights.main_topics:
        print(f"📚 Main topics: {', '.join(insights.main_topics[:5])}")
    
    if insights.organizations:
        print(f"🏢 Organizations: {', '.join(insights.organizations)}")
    
    if insights.mentioned_people:
        print(f"👤 People mentioned: {', '.join(insights.mentioned_people)}")
    
    if insights.action_items:
        print(f"📋 Action items ({len(insights.action_items)}):")
        for i, item in enumerate(insights.action_items[:3], 1):
            print(f"   {i}. {item}")
    
    if insights.technical_terms:
        print(f"🔬 Technical terms: {', '.join(insights.technical_terms[:5])}")
    
    # Content Structure
    print(f"\n📊 CONTENT STRUCTURE:")
    print("-" * 25)
    
    structure = result.content_structure
    duration_min = structure.get('total_duration', 0) / 60
    print(f"📅 Total duration: {duration_min:.1f} minutes")
    
    speaking_dist = structure.get('speaking_distribution', {})
    if speaking_dist:
        print("🎤 Speaking distribution:")
        for speaker, percentage in speaking_dist.items():
            print(f"   • {speaker}: {percentage:.1f}%")
    
    # Quality Metrics
    print(f"\n📈 QUALITY METRICS:")
    print("-" * 20)
    
    metrics = result.quality_metrics
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"• {metric.replace('_', ' ').title()}: {value:.1%}")
    
    # Processing Information
    metadata = result.processing_metadata
    print(f"\n⚙️  PROCESSING METADATA:")
    print(f"• Analysis duration: {metadata.get('analysis_duration', 0):.2f}s")
    print(f"• Total segments: {metadata.get('total_segments', 0)}")
    print(f"• Models used: {sum(metadata.get('models_used', {}).values())} of {len(metadata.get('models_used', {}))}")

def show_generated_files(output_dir: str):
    """Show the files generated by the context analysis"""
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    generated_files = list(output_path.glob("*"))
    if not generated_files:
        return
    
    print(f"\n📁 GENERATED FILES IN '{output_dir}/' DIRECTORY:")
    print("-" * 45)
    
    for file_path in sorted(generated_files):
        file_size = file_path.stat().st_size
        size_kb = file_size / 1024
        
        print(f"📄 {file_path.name} ({size_kb:.1f} KB)")
        
        # Show preview for smaller files
        if file_path.suffix == '.md' and size_kb < 10:
            print("   Preview:")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:3]
                    for line in lines:
                        if line.strip():
                            print(f"   {line.strip()}")
                    if len(lines) >= 3:
                        print("   ...")
            except:
                pass
            print()

def print_usage_instructions():
    """Print instructions for using the context analysis system"""
    
    print("\n" + "🎯" * 25)
    print("  USAGE INSTRUCTIONS")
    print("🎯" * 25)
    print()
    print("To use context analysis with your own audio transcriptions:")
    print()
    print("1. 📁 Place your audio files in data/input/")
    print("2. 📄 Add reference documents (like inputdoc.docx) to data/input/")
    print("3. 🚀 Run: python scripts/run_context_analysis_test.py")
    print("4. 📊 Check the 'output/' directory for detailed analysis results")
    print()
    print("Generated files include:")
    print("• 📝 executive_summary.md - Executive summary")
    print("• 👥 speaker_profiles.json - Detailed speaker analysis")
    print("• 💡 conversation_insights.json - Key insights and topics")
    print("• 📊 complete_analysis_report.json - Full analysis data")
    print("• 📋 detailed_analysis_report.md - Comprehensive report")
    print()
    print("Features available:")
    print("• 🎯 Multi-speaker identification and profiling")
    print("• 📝 AI-powered executive summary generation")
    print("• 💭 Sentiment and emotion analysis")
    print("• 🔗 Speaker interaction network analysis")
    print("• 📋 Automatic action item extraction")
    print("• 📊 Comprehensive quality metrics")

async def main():
    """Main demo execution"""
    
    print("🔍 Context Analysis Demonstration")
    print("=" * 40)
    print()
    
    if not CONTEXT_ANALYZER_AVAILABLE:
        print("❌ Context analyzer not available")
        print("Install requirements: pip install -r requirements/requirements_context_analysis.txt")
        return
    
    # Run the demonstration
    await run_context_analysis_demo()
    
    # Show usage instructions
    print_usage_instructions()
    
    print("\n✅ Demonstration completed!")
    print("Check the 'output/' directory for generated files.")

if __name__ == "__main__":
    asyncio.run(main()) 