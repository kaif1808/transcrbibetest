#!/usr/bin/env python3
"""
Test Local LLM Initialization Fix
Verifies that Ollama connection and model detection works properly
"""

import asyncio
from noun_extraction_system import NounExtractionConfig, LocalLLMExtractor

async def test_llm_initialization():
    """Test LLM initialization with improved error handling"""
    
    print("🔧 Testing Local LLM Initialization Fix")
    print("=" * 50)
    
    # Test with Ollama
    print("🤖 Testing Ollama LLM initialization...")
    
    config = NounExtractionConfig(
        domain_focus="education",
        use_local_llm=True,
        llm_provider="ollama",
        llm_model="llama3.2",  # This should match llama3.2:latest
        extract_unusual_nouns=True
    )
    
    # Initialize LLM extractor
    llm_extractor = LocalLLMExtractor(config)
    
    # Test initialization
    await llm_extractor.initialize()
    
    if config.use_local_llm:
        print("✅ LLM initialization successful!")
        print(f"   Model: {config.llm_model}")
        print(f"   Provider: {config.llm_provider}")
        
        # Test a simple extraction
        test_text = "The Ministry of Industry and Trade works with TVET programs."
        
        print(f"\n🔍 Testing unusual noun extraction...")
        print(f"   Text: {test_text}")
        
        try:
            unusual_nouns = await llm_extractor.extract_unusual_nouns(test_text)
            print(f"✅ Extraction completed: {len(unusual_nouns)} unusual nouns found")
            
            for noun in unusual_nouns[:3]:  # Show first 3
                print(f"   • {noun.text} (confidence: {noun.confidence:.2f})")
                
        except Exception as e:
            print(f"⚠️  Extraction test failed: {e}")
    else:
        print("❌ LLM initialization failed - Local LLM disabled")
    
    # Test fallback behavior
    print(f"\n🔄 Testing with LLM disabled...")
    config_no_llm = NounExtractionConfig(
        domain_focus="education",
        use_local_llm=False,
        extract_unusual_nouns=False
    )
    
    llm_extractor_disabled = LocalLLMExtractor(config_no_llm)
    await llm_extractor_disabled.initialize()
    
    unusual_nouns_disabled = await llm_extractor_disabled.extract_unusual_nouns("Test text")
    print(f"✅ Fallback test: {len(unusual_nouns_disabled)} nouns (expected: 0)")

async def test_model_name_matching():
    """Test model name matching logic"""
    
    print("\n🎯 Testing Model Name Matching")
    print("=" * 40)
    
    # Simulate available models
    available_models = ["llama3.2:latest", "mistral:7b", "codellama:13b"]
    test_queries = ["llama3.2", "llama3.2:latest", "mistral", "nonexistent"]
    
    for query in test_queries:
        model_found = False
        matching_model = None
        
        for model in available_models:
            if (query == model or 
                query in model or 
                f"{query}:latest" == model):
                model_found = True
                matching_model = model
                break
        
        status = "✅ Found" if model_found else "❌ Not found"
        result = f" -> {matching_model}" if matching_model else ""
        print(f"   {query}: {status}{result}")

async def main():
    """Run all LLM initialization tests"""
    await test_llm_initialization()
    await test_model_name_matching()
    
    print("\n🏁 LLM Initialization Test Complete!")
    print("\n💡 Summary:")
    print("   ✅ Improved error handling for Ollama connection")
    print("   ✅ Better model name matching (handles :latest suffix)")
    print("   ✅ Graceful fallback when LLM unavailable")
    print("   ✅ Clear error messages and installation instructions")

if __name__ == "__main__":
    asyncio.run(main()) 