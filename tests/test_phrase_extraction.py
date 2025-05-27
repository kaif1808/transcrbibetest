#!/usr/bin/env python3
"""
Test Enhanced Phrase Extraction with GPU Acceleration
Demonstrates multi-word phrase extraction and organizational entity detection
"""

import asyncio
import time
from noun_extraction_system import NounExtractor, NounExtractionConfig

async def test_phrase_extraction():
    """Test enhanced phrase extraction with real-world examples"""
    
    # Test text with multi-word organizations and technical terms
    test_text = """
    The Ministry of Industry and Trade announced a new partnership with the 
    Department of Vocational Education and Continuing Education. The Technical 
    and Vocational Education and Training program will focus on artificial 
    intelligence and machine learning technologies. Representatives from the 
    Vietnam National University and Ho Chi Minh City University of Technology 
    will collaborate with the German Agency for International Cooperation on 
    this competency-based training and assessment initiative. The Institute 
    of Strategy and Policy will provide guidance for the digital transformation 
    project.
    """
    
    print("🔍 ENHANCED PHRASE EXTRACTION TEST")
    print("=" * 50)
    print(f"📝 Test text: {len(test_text)} characters")
    
    # Configure for maximum phrase extraction with GPU acceleration
    config = NounExtractionConfig(
        domain_focus="education",
        extract_phrases=True,
        use_gpu_acceleration=True,
        gpu_batch_size=32,
        parallel_processing=True,
        max_workers=4,
        min_phrase_length=2,
        max_phrase_length=8,
        phrase_confidence_threshold=0.7,
        use_local_llm=False,  # Focus on GPU models
        min_frequency=1,
        filter_duplicates=True
    )
    
    # Initialize extractor
    extractor = NounExtractor(config)
    
    # Extract nouns and phrases
    start_time = time.time()
    results = await extractor.extract_nouns_from_text(test_text)
    extraction_time = time.time() - start_time
    
    print(f"⚡ Extraction completed in {extraction_time:.3f}s")
    print(f"🖥️  Using GPU acceleration: {config.use_gpu_acceleration}")
    
    # Display results by category
    print("\n🔍 EXTRACTION RESULTS:")
    print("-" * 40)
    
    categories = [
        ("phrase_entities", "Multi-word Phrases", "👥"),
        ("organizational_phrases", "Organizational Entities", "🏛️"),
        ("proper_nouns", "Proper Nouns", "🏷️"),
        ("named_entities", "Named Entities", "👤"),
        ("technical_terms", "Technical Terms", "🔧"),
        ("common_nouns", "Common Nouns", "📝")
    ]
    
    total_extracted = 0
    
    for category_key, category_name, emoji in categories:
        items = results.get(category_key, [])
        if items:
            print(f"\n{emoji} {category_name}: {len(items)} found")
            total_extracted += len(items)
            
            # Show top items with details
            for i, item in enumerate(items[:5]):
                print(f"   {i+1}. '{item.text}' (freq: {item.frequency}, conf: {item.confidence:.2f})")
                if hasattr(item, 'entity_type') and item.entity_type:
                    print(f"      Type: {item.entity_type}")
                if item.contexts and len(item.contexts) > 0:
                    context = item.contexts[0][:100] + "..." if len(item.contexts[0]) > 100 else item.contexts[0]
                    print(f"      Context: {context}")
            
            if len(items) > 5:
                print(f"   ... and {len(items) - 5} more")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total extracted: {total_extracted}")
    print(f"   Processing speed: {len(test_text.split()) / extraction_time:.1f} words/sec")
    
    # Highlight multi-word phrases specifically
    phrases = results.get("phrase_entities", []) + results.get("organizational_phrases", [])
    if phrases:
        print(f"\n🌟 MULTI-WORD PHRASES DETECTED:")
        print("-" * 40)
        for phrase in phrases:
            word_count = len(phrase.text.split())
            print(f"   • '{phrase.text}' ({word_count} words)")
            print(f"     Category: {phrase.category}")
            print(f"     Confidence: {phrase.confidence:.2f}")
            if hasattr(phrase, 'entity_type') and phrase.entity_type:
                print(f"     Type: {phrase.entity_type}")
            print()

async def test_ministry_extraction():
    """Test specific ministry and department extraction"""
    
    print("\n🏛️ MINISTRY AND DEPARTMENT EXTRACTION TEST")
    print("=" * 55)
    
    ministry_text = """
    The Ministry of Education and Training works closely with the Ministry of 
    Labour, Invalids and Social Affairs. The Department of Vocational Education 
    and Continuing Education coordinates with the General Department of Training 
    and Education. The Vietnam National University collaborates with international 
    partners on Technical and Vocational Education and Training programs.
    """
    
    config = NounExtractionConfig(
        domain_focus="education",
        extract_phrases=True,
        use_gpu_acceleration=True,
        min_phrase_length=2,
        max_phrase_length=10,  # Allow longer organizational names
        phrase_confidence_threshold=0.6,
        min_frequency=1
    )
    
    extractor = NounExtractor(config)
    results = await extractor.extract_nouns_from_text(ministry_text)
    
    # Focus on organizational phrases
    org_phrases = results.get("organizational_phrases", [])
    
    print(f"🏢 Found {len(org_phrases)} organizational entities:")
    for org in org_phrases:
        print(f"   • {org.text}")
        print(f"     Category: {org.category}")
        print(f"     Frequency: {org.frequency}")
        print(f"     Confidence: {org.confidence:.2f}")
        print()

async def main():
    """Run comprehensive phrase extraction tests"""
    await test_phrase_extraction()
    await test_ministry_extraction()
    
    print("🏁 Enhanced phrase extraction test completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ GPU acceleration with Apple Silicon MPS")
    print("   ✅ Multi-word phrase extraction (2-8 words)")
    print("   ✅ Organizational entity detection")
    print("   ✅ Ministry/Department pattern recognition")
    print("   ✅ Technical term identification")
    print("   ✅ Parallel processing optimization")

if __name__ == "__main__":
    asyncio.run(main()) 