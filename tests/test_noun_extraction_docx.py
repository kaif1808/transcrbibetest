#!/usr/bin/env python3
"""
Test Enhanced Noun Extraction with DOCX and Local LLM
Demonstrates noun extraction from inputdoc.docx with local LLM for unusual nouns
"""

import asyncio
import time
import json
from pathlib import Path

# Import the enhanced noun extraction system
try:
    from noun_extraction_system import (
        extract_nouns_from_document,
        save_noun_analysis,
        NounExtractionConfig,
        DocumentNounExtractor
    )
    NOUN_EXTRACTION_AVAILABLE = True
except ImportError:
    NOUN_EXTRACTION_AVAILABLE = False
    print("❌ Noun extraction module not available")

async def test_docx_noun_extraction():
    """Test noun extraction from inputdoc.docx"""
    print("📄 Testing DOCX Noun Extraction")
    print("=" * 50)
    
    if not NOUN_EXTRACTION_AVAILABLE:
        print("❌ Please install requirements: pip install -r requirements_correction.txt")
        return
    
    docx_file = "inputdoc.docx"
    
    if not Path(docx_file).exists():
        print(f"❌ File not found: {docx_file}")
        return
    
    # Test basic extraction without LLM first
    print("🔍 Phase 1: Basic noun extraction (no LLM)")
    config_basic = NounExtractionConfig(
        domain_focus="education",
        use_local_llm=False,  # Disable LLM for initial test
        extract_unusual_nouns=False,
        extract_phrases=True,  # Enable phrase extraction
        use_gpu_acceleration=True,  # Use GPU acceleration
        gpu_batch_size=64,
        parallel_processing=True,
        max_workers=4,
        min_phrase_length=2,
        max_phrase_length=8,
        min_frequency=2,  # Only show nouns that appear multiple times
        filter_duplicates=True
    )
    
    start_time = time.time()
    basic_analysis = await extract_nouns_from_document(docx_file, "education", config_basic)
    basic_time = time.time() - start_time
    
    if "error" in basic_analysis:
        print(f"❌ Error: {basic_analysis['error']}")
        return
    
    # Display basic results
    noun_results = basic_analysis.get("noun_analysis", {})
    content_stats = basic_analysis.get("content_statistics", {})
    
    print(f"✅ Basic extraction completed in {basic_time:.2f}s")
    print(f"📊 Document statistics:")
    print(f"   📝 Characters: {content_stats.get('total_characters', 0):,}")
    print(f"   📖 Words: {content_stats.get('total_words', 0):,}")
    print(f"   📄 Sentences: {content_stats.get('total_sentences', 0):,}")
    print(f"   📋 Paragraphs: {content_stats.get('total_paragraphs', 0):,}")
    
    print(f"\n🔍 Basic noun extraction results:")
    for category, nouns in noun_results.items():
        if nouns and category != "all_nouns":
            print(f"   {category.replace('_', ' ').title()}: {len(nouns)} found")
            # Show top 5 for each category
            for noun in nouns[:5]:
                print(f"     • {noun.text} (freq: {noun.frequency}, conf: {noun.confidence:.2f})")
    
    # Save basic results
    basic_output = f"inputdoc_basic_nouns_{int(time.time())}.json"
    save_noun_analysis(basic_analysis, basic_output)
    
    return basic_analysis

async def test_llm_enhanced_extraction():
    """Test LLM-enhanced noun extraction"""
    print("\n🤖 Testing LLM-Enhanced Noun Extraction")
    print("=" * 50)
    
    # Test with local LLM for unusual nouns
    print("🔍 Phase 2: LLM-enhanced extraction for unusual nouns")
    
    # Try Ollama first, then fallback to MLX-LM
    for provider in ["ollama", "mlx-lm"]:
        print(f"\n🤖 Trying {provider.upper()} for local LLM...")
        
        config_llm = NounExtractionConfig(
            domain_focus="education",
            use_local_llm=True,
            llm_provider=provider,
            llm_model="llama3.2" if provider == "ollama" else "Llama-3.2-3B-Instruct",
            extract_unusual_nouns=True,
            extract_phrases=True,  # Enable phrase extraction
            use_gpu_acceleration=True,  # Use GPU acceleration
            gpu_batch_size=64,
            parallel_processing=True,
            max_workers=4,
            min_phrase_length=2,
            max_phrase_length=8,
            unusual_noun_threshold=0.7,
            min_frequency=1,  # Lower threshold for LLM extraction
            filter_duplicates=True
        )
        
        try:
            start_time = time.time()
            llm_analysis = await extract_nouns_from_document("inputdoc.docx", "education", config_llm)
            llm_time = time.time() - start_time
            
            if "error" not in llm_analysis:
                # Display LLM results
                noun_results = llm_analysis.get("noun_analysis", {})
                
                print(f"✅ LLM extraction completed in {llm_time:.2f}s")
                print(f"\n🔍 LLM-enhanced noun extraction results:")
                
                for category, nouns in noun_results.items():
                    if nouns and category != "all_nouns":
                        print(f"   {category.replace('_', ' ').title()}: {len(nouns)} found")
                        
                        # Show detailed results for unusual nouns
                        if category == "unusual_nouns":
                            print(f"   🌟 Unusual nouns identified by LLM:")
                            for noun in nouns[:10]:  # Show top 10 unusual nouns
                                contexts = noun.contexts[:1] if noun.contexts else ["No context"]
                                print(f"     • {noun.text} (conf: {noun.confidence:.2f})")
                                print(f"       Context: {contexts[0]}")
                        else:
                            # Show top 3 for other categories
                            for noun in nouns[:3]:
                                print(f"     • {noun.text} (freq: {noun.frequency}, conf: {noun.confidence:.2f})")
                
                # Save LLM results
                llm_output = f"inputdoc_llm_nouns_{provider}_{int(time.time())}.json"
                save_noun_analysis(llm_analysis, llm_output)
                
                # Compare with basic extraction
                await compare_extraction_methods(llm_analysis)
                
                return llm_analysis
            else:
                print(f"⚠️  {provider.upper()} extraction failed: {llm_analysis.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"⚠️  {provider.upper()} failed: {e}")
            continue
    
    print("⚠️  No local LLM providers available. Install Ollama or MLX-LM for enhanced extraction.")
    return None

async def compare_extraction_methods(llm_analysis):
    """Compare different extraction methods"""
    print("\n📊 Extraction Method Comparison")
    print("=" * 40)
    
    noun_results = llm_analysis.get("noun_analysis", {})
    
    # Count by extraction method/category
    method_counts = {}
    for category, nouns in noun_results.items():
        if category != "all_nouns":
            method_counts[category] = len(nouns)
    
    print("📈 Noun counts by method:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {method.replace('_', ' ').title()}: {count}")
    
    # Highlight unusual nouns found by LLM
    unusual_nouns = noun_results.get("unusual_nouns", [])
    if unusual_nouns:
        print(f"\n🌟 Top unusual nouns identified by LLM:")
        for noun in unusual_nouns[:15]:
            print(f"   • {noun.text} ({noun.confidence:.2f}) - {noun.contexts[0] if noun.contexts else 'LLM identified'}")

async def test_integration_with_transcription():
    """Test integration with Lightning MLX transcription"""
    print("\n🎤 Testing Integration with Transcription")
    print("=" * 50)
    
    # Create a mock transcription result for testing
    mock_transcription = {
        "text": "The Ministry of Education in Vietnam is implementing TVET programs with competency-based assessment frameworks.",
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "The Ministry of Education in Vietnam is implementing TVET programs",
                "speaker": "SPEAKER_01"
            },
            {
                "start": 5.0,
                "end": 10.0,
                "text": "with competency-based assessment frameworks.",
                "speaker": "SPEAKER_01"
            }
        ],
        "audio_file": "test_audio.wav"
    }
    
    try:
        from noun_extraction_system import extract_nouns_from_transcription
        
        config = NounExtractionConfig(
            domain_focus="education",
            use_local_llm=False,  # Disable for quick test
            min_frequency=1
        )
        
        transcription_analysis = await extract_nouns_from_transcription(
            mock_transcription, 
            "education", 
            config
        )
        
        summary = transcription_analysis.get("summary", {})
        print(f"✅ Transcription noun extraction completed")
        print(f"   📝 Total nouns: {summary.get('total_nouns_extracted', 0)}")
        print(f"   👤 Named entities: {len(summary.get('top_named_entities', []))}")
        print(f"   🔧 Technical terms: {len(summary.get('top_technical_terms', []))}")
        
        return transcription_analysis
        
    except Exception as e:
        print(f"⚠️  Transcription integration test failed: {e}")
        return None

async def main():
    """Run comprehensive noun extraction tests"""
    print("🔍 ENHANCED NOUN EXTRACTION TEST SUITE")
    print("📄 Document: inputdoc.docx")
    print("🤖 Local LLM: Enabled for unusual noun detection")
    print("=" * 70)
    
    # Test 1: Basic DOCX extraction
    basic_result = await test_docx_noun_extraction()
    
    # Test 2: LLM-enhanced extraction
    llm_result = await test_llm_enhanced_extraction()
    
    # Test 3: Integration with transcription
    transcription_result = await test_integration_with_transcription()
    
    print("\n🏁 Test Suite Completed!")
    print("\n💡 Next Steps:")
    print("  1. Review the generated JSON files for detailed analysis")
    print("  2. Install Ollama or MLX-LM for local LLM features:")
    print("     - Ollama: https://ollama.ai/ then 'ollama pull llama3.2'")
    print("     - MLX-LM: pip install mlx-lm")
    print("  3. Run Lightning MLX transcription with noun extraction enabled")
    print("  4. Use the enhanced system for document analysis and transcription")

if __name__ == "__main__":
    asyncio.run(main()) 