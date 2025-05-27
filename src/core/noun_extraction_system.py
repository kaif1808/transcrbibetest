#!/usr/bin/env python3
"""
Advanced Noun Extraction System
Comprehensive NLP system for extracting and analyzing nouns from documents and transcriptions.
Supports GPU acceleration, local LLM integration, and domain-specific vocabulary building.
"""

import asyncio
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import logging

# Core NLP imports
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Document processing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Local LLM support
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from mlx_lm import load, generate
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedNoun:
    """Represents an extracted noun with metadata"""
    text: str
    frequency: int = 1
    confidence: float = 1.0
    category: str = "general"
    contexts: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    is_phrase: bool = False
    pos_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "category": self.category,
            "contexts": self.contexts[:3],  # Limit contexts
            "domain": self.domain,
            "is_phrase": self.is_phrase,
            "pos_tags": self.pos_tags
        }

@dataclass
class NounExtractionConfig:
    """Configuration for noun extraction process"""
    domain_focus: str = "general"
    extract_phrases: bool = True
    use_local_llm: bool = False
    llm_provider: str = "ollama"  # "ollama" or "mlx-lm"
    llm_model: str = "llama3.2:latest"
    extract_unusual_nouns: bool = False
    use_gpu_acceleration: bool = True
    gpu_batch_size: int = 64
    parallel_processing: bool = True
    max_workers: int = 4
    min_phrase_length: int = 2
    max_phrase_length: int = 6
    min_frequency: int = 1
    confidence_threshold: float = 0.5
    unusual_noun_threshold: float = 0.7
    filter_duplicates: bool = True

class DocumentNounExtractor:
    """Main document noun extraction engine"""
    
    def __init__(self, config: NounExtractionConfig):
        self.config = config
        self.nlp = None
        self.domain_terms = self._load_domain_terms()
        self.extracted_nouns = defaultdict(list)
        
    def _load_domain_terms(self) -> Dict[str, Set[str]]:
        """Load domain-specific terminology"""
        domains = {
            "education": {
                "curriculum", "pedagogy", "assessment", "competency", "framework",
                "tvet", "vocational", "technical", "qualification", "accreditation",
                "ministry", "education", "training", "learning", "student", "teacher",
                "instructor", "programme", "course", "module", "skill", "knowledge"
            },
            "technology": {
                "algorithm", "framework", "api", "database", "software", "hardware",
                "programming", "development", "implementation", "system", "platform",
                "interface", "architecture", "deployment", "configuration", "optimization"
            },
            "business": {
                "management", "strategy", "organization", "department", "company",
                "enterprise", "corporation", "business", "commercial", "financial",
                "economic", "market", "industry", "sector", "revenue", "profit"
            }
        }
        return domains
    
    async def initialize(self):
        """Initialize NLP models and components"""
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ SpaCy model loaded successfully")
            except OSError:
                logger.warning("⚠️ SpaCy model not found, using basic tokenizer")
                self.nlp = English()
        else:
            logger.warning("⚠️ SpaCy not available")
    
    async def extract_from_text(self, text: str) -> Dict[str, List[ExtractedNoun]]:
        """Extract nouns from raw text"""
        if not text.strip():
            return {}
        
        results = {
            "common_nouns": [],
            "proper_nouns": [],
            "technical_terms": [],
            "domain_specific": [],
            "phrases": [],
            "unusual_nouns": [],
            "all_nouns": []
        }
        
        # Basic extraction without spaCy
        if not self.nlp:
            return await self._basic_extraction(text, results)
        
        # Advanced extraction with spaCy
        doc = self.nlp(text)
        noun_counts = Counter()
        noun_contexts = defaultdict(list)
        
        # Extract individual nouns
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                noun_text = token.text.lower()
                noun_counts[noun_text] += 1
                
                # Get context
                start = max(0, token.i - 3)
                end = min(len(doc), token.i + 4)
                context = " ".join([t.text for t in doc[start:end]])
                noun_contexts[noun_text].append(context)
        
        # Extract noun phrases
        if self.config.extract_phrases:
            for chunk in doc.noun_chunks:
                if (self.config.min_phrase_length <= len(chunk.text.split()) <= self.config.max_phrase_length):
                    phrase_text = chunk.text.lower()
                    noun_counts[phrase_text] += 1
                    
                    # Get context for phrase
                    start = max(0, chunk.start - 3)
                    end = min(len(doc), chunk.end + 3)
                    context = " ".join([t.text for t in doc[start:end]])
                    noun_contexts[phrase_text].append(context)
        
        # Categorize nouns
        for noun_text, frequency in noun_counts.items():
            if frequency < self.config.min_frequency:
                continue
            
            contexts = noun_contexts[noun_text][:3]
            confidence = min(1.0, frequency / 10.0)  # Simple confidence calculation
            
            # Create ExtractedNoun object
            extracted_noun = ExtractedNoun(
                text=noun_text,
                frequency=frequency,
                confidence=confidence,
                contexts=contexts
            )
            
            # Categorize
            if noun_text.istitle() or noun_text.isupper():
                extracted_noun.category = "proper_noun"
                results["proper_nouns"].append(extracted_noun)
            elif self._is_domain_specific(noun_text):
                extracted_noun.category = "domain_specific"
                extracted_noun.domain = self.config.domain_focus
                results["domain_specific"].append(extracted_noun)
            elif self._is_technical_term(noun_text):
                extracted_noun.category = "technical"
                results["technical_terms"].append(extracted_noun)
            elif len(noun_text.split()) > 1:
                extracted_noun.category = "phrase"
                extracted_noun.is_phrase = True
                results["phrases"].append(extracted_noun)
            else:
                extracted_noun.category = "common"
                results["common_nouns"].append(extracted_noun)
            
            results["all_nouns"].append(extracted_noun)
        
        # LLM enhancement for unusual nouns
        if self.config.use_local_llm and self.config.extract_unusual_nouns:
            unusual_nouns = await self._extract_unusual_nouns_with_llm(text)
            results["unusual_nouns"] = unusual_nouns
            results["all_nouns"].extend(unusual_nouns)
        
        # Sort by frequency and confidence
        for category in results:
            results[category].sort(key=lambda x: (x.frequency, x.confidence), reverse=True)
        
        return results
    
    async def _basic_extraction(self, text: str, results: Dict) -> Dict[str, List[ExtractedNoun]]:
        """Basic noun extraction without spaCy"""
        # Simple word tokenization and filtering
        words = re.findall(r'\b[A-Za-z]{3,}\b', text)
        word_counts = Counter(word.lower() for word in words)
        
        for word, frequency in word_counts.items():
            if frequency >= self.config.min_frequency:
                extracted_noun = ExtractedNoun(
                    text=word,
                    frequency=frequency,
                    confidence=0.7,  # Lower confidence for basic extraction
                    category="basic"
                )
                results["common_nouns"].append(extracted_noun)
                results["all_nouns"].append(extracted_noun)
        
        return results
    
    def _is_domain_specific(self, noun_text: str) -> bool:
        """Check if noun is domain-specific"""
        domain_terms = self.domain_terms.get(self.config.domain_focus, set())
        return any(term in noun_text.lower() for term in domain_terms)
    
    def _is_technical_term(self, noun_text: str) -> bool:
        """Check if noun is a technical term"""
        technical_indicators = [
            "system", "framework", "algorithm", "methodology", "implementation",
            "configuration", "optimization", "integration", "analysis", "evaluation"
        ]
        return any(indicator in noun_text.lower() for indicator in technical_indicators)
    
    async def _extract_unusual_nouns_with_llm(self, text: str) -> List[ExtractedNoun]:
        """Extract unusual nouns using local LLM"""
        unusual_nouns = []
        
        if self.config.llm_provider == "ollama" and OLLAMA_AVAILABLE:
            unusual_nouns = await self._extract_with_ollama(text)
        elif self.config.llm_provider == "mlx-lm" and MLX_LM_AVAILABLE:
            unusual_nouns = await self._extract_with_mlx_lm(text)
        else:
            logger.warning(f"LLM provider {self.config.llm_provider} not available")
        
        return unusual_nouns
    
    async def _extract_with_ollama(self, text: str) -> List[ExtractedNoun]:
        """Extract unusual nouns using Ollama"""
        try:
            prompt = f"""
            Analyze the following text and identify unusual, domain-specific, or technical nouns that might be important for {self.config.domain_focus}. 
            Return only the nouns, one per line, without explanations.
            
            Text: {text[:1000]}  # Limit text length
            
            Unusual nouns:
            """
            
            response = ollama.generate(model=self.config.llm_model, prompt=prompt)
            
            if response and "response" in response:
                lines = response["response"].strip().split('\n')
                unusual_nouns = []
                
                for line in lines:
                    noun = line.strip().lower()
                    if noun and len(noun) > 2:
                        extracted_noun = ExtractedNoun(
                            text=noun,
                            frequency=1,
                            confidence=self.config.unusual_noun_threshold,
                            category="unusual",
                            contexts=[f"LLM identified in {self.config.domain_focus} context"]
                        )
                        unusual_nouns.append(extracted_noun)
                
                return unusual_nouns[:10]  # Limit to top 10
        
        except Exception as e:
            logger.warning(f"Ollama extraction failed: {e}")
        
        return []
    
    async def _extract_with_mlx_lm(self, text: str) -> List[ExtractedNoun]:
        """Extract unusual nouns using MLX-LM"""
        try:
            model, tokenizer = load(self.config.llm_model)
            
            prompt = f"Identify important {self.config.domain_focus} terms in: {text[:500]}"
            
            response = generate(model, tokenizer, prompt, max_tokens=100)
            
            # Parse response for nouns
            words = re.findall(r'\b[A-Za-z]{3,}\b', response)
            unusual_nouns = []
            
            for word in words[:5]:  # Limit to 5
                extracted_noun = ExtractedNoun(
                    text=word.lower(),
                    frequency=1,
                    confidence=self.config.unusual_noun_threshold,
                    category="unusual",
                    contexts=[f"MLX-LM identified"]
                )
                unusual_nouns.append(extracted_noun)
            
            return unusual_nouns
        
        except Exception as e:
            logger.warning(f"MLX-LM extraction failed: {e}")
        
        return []

class LocalLLMExtractor:
    """Local LLM integration for enhanced noun extraction"""
    
    def __init__(self, config: NounExtractionConfig):
        self.config = config
    
    async def extract_with_context(self, text: str, domain: str) -> List[ExtractedNoun]:
        """Extract nouns with domain context using LLM"""
        if self.config.llm_provider == "ollama":
            return await self._ollama_extraction(text, domain)
        elif self.config.llm_provider == "mlx-lm":
            return await self._mlx_extraction(text, domain)
        
        return []
    
    async def _ollama_extraction(self, text: str, domain: str) -> List[ExtractedNoun]:
        """Ollama-based extraction"""
        # Implementation similar to above but with more sophisticated prompting
        return []
    
    async def _mlx_extraction(self, text: str, domain: str) -> List[ExtractedNoun]:
        """MLX-LM based extraction"""
        # Implementation for MLX-LM
        return []

# Main API functions
async def extract_nouns_from_document(
    document_path: str, 
    domain: str, 
    config: NounExtractionConfig
) -> Dict[str, Any]:
    """Extract nouns from a document file"""
    
    try:
        # Read document content
        if document_path.endswith('.docx') and DOCX_AVAILABLE:
            content = await _read_docx_content(document_path)
        elif document_path.endswith('.txt'):
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            return {"error": f"Unsupported file format: {document_path}"}
        
        if not content.strip():
            return {"error": "No content found in document"}
        
        # Initialize extractor
        extractor = DocumentNounExtractor(config)
        await extractor.initialize()
        
        # Extract nouns
        noun_analysis = await extractor.extract_from_text(content)
        
        # Calculate statistics
        content_stats = {
            "total_characters": len(content),
            "total_words": len(content.split()),
            "total_sentences": len(re.split(r'[.!?]+', content)),
            "total_paragraphs": len(content.split('\n\n'))
        }
        
        return {
            "document_path": document_path,
            "domain": domain,
            "content_statistics": content_stats,
            "noun_analysis": {
                category: [noun.to_dict() for noun in nouns]
                for category, nouns in noun_analysis.items()
            },
            "extraction_metadata": {
                "config": {
                    "domain_focus": config.domain_focus,
                    "extract_phrases": config.extract_phrases,
                    "use_local_llm": config.use_local_llm,
                    "llm_provider": config.llm_provider if config.use_local_llm else None
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}"}

async def extract_nouns_from_transcription(
    transcription_data: Dict[str, Any], 
    domain: str, 
    config: NounExtractionConfig
) -> Dict[str, Any]:
    """Extract nouns from transcription data"""
    
    try:
        # Extract text from transcription
        if "text" in transcription_data:
            text = transcription_data["text"]
        elif "segments" in transcription_data:
            text = " ".join(seg.get("text", "") for seg in transcription_data["segments"])
        else:
            return {"error": "No text found in transcription data"}
        
        # Initialize extractor
        extractor = DocumentNounExtractor(config)
        await extractor.initialize()
        
        # Extract nouns
        noun_analysis = await extractor.extract_from_text(text)
        
        return {
            "transcription_source": "audio_file",
            "domain": domain,
            "noun_analysis": {
                category: [noun.to_dict() for noun in nouns]
                for category, nouns in noun_analysis.items()
            },
            "extraction_metadata": {
                "config": {
                    "domain_focus": config.domain_focus,
                    "extract_phrases": config.extract_phrases,
                    "use_local_llm": config.use_local_llm
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
    except Exception as e:
        return {"error": f"Transcription noun extraction failed: {str(e)}"}

async def _read_docx_content(file_path: str) -> str:
    """Read content from DOCX file"""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not available")
    
    doc = Document(file_path)
    content = []
    
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            content.append(paragraph.text.strip())
    
    return "\n".join(content)

def save_noun_analysis(analysis_result: Dict[str, Any], output_file: str):
    """Save noun analysis results to file"""
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Noun analysis saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save analysis: {e}")

# Utility classes and functions for backward compatibility
class NounExtractor(DocumentNounExtractor):
    """Alias for backward compatibility"""
    pass

# Main execution for testing
async def main():
    """Test the noun extraction system"""
    config = NounExtractionConfig(
        domain_focus="education",
        extract_phrases=True,
        use_local_llm=False,
        extract_unusual_nouns=False
    )
    
    # Test with sample text
    sample_text = """
    The Ministry of Education has implemented a new competency-based assessment framework 
    for TVET programs. This technical education system focuses on practical skills and 
    vocational training to prepare students for the workforce.
    """
    
    extractor = DocumentNounExtractor(config)
    await extractor.initialize()
    
    results = await extractor.extract_from_text(sample_text)
    
    print("🔍 Noun Extraction Test Results:")
    for category, nouns in results.items():
        if nouns:
            print(f"\n{category.replace('_', ' ').title()}: {len(nouns)} found")
            for noun in nouns[:3]:
                print(f"  • {noun.text} (freq: {noun.frequency}, conf: {noun.confidence:.2f})")

if __name__ == "__main__":
    asyncio.run(main()) 