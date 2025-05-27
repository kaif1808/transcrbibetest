#!/usr/bin/env python3
"""
Transcription Corrector & Accuracy Enhancer
Advanced post-processing for Lightning MLX and other transcription outputs
Features: Noun correction, context awareness, speaker consistency, and quality improvements
"""

import re
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
import asyncio

# Optional dependencies for enhanced correction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  SpaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available for advanced correction. Install with: pip install transformers")

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not available. Install with: pip install nltk")

logger = logging.getLogger(__name__)

@dataclass
class CorrectionConfig:
    """Configuration for transcription correction"""
    # Basic correction settings
    enable_noun_correction: bool = True
    enable_context_correction: bool = True
    enable_speaker_consistency: bool = True
    enable_grammar_correction: bool = True
    
    # Advanced AI correction
    enable_ai_correction: bool = True
    ai_correction_model: str = "facebook/bart-large-cnn"  # For text improvement
    
    # Domain-specific corrections
    custom_vocabulary: List[str] = None
    domain_terms: Dict[str, List[str]] = None
    
    # Quality thresholds
    min_confidence_threshold: float = 0.7
    max_segment_length: int = 500
    
    # Processing options
    preserve_timestamps: bool = True
    fix_repetitions: bool = True
    normalize_punctuation: bool = True
    
    def __post_init__(self):
        if self.custom_vocabulary is None:
            self.custom_vocabulary = []
        if self.domain_terms is None:
            self.domain_terms = {
                "education": ["TVET", "vocational", "education", "training", "assessment", "competency"],
                "technology": ["semiconductor", "digital", "transformation", "AI", "machine learning"],
                "business": ["ministry", "government", "policy", "implementation", "framework"],
                "general": ["Vietnam", "Vietnamese", "Ho Chi Minh", "Hanoi"]
            }


class CommonCorrections:
    """Common transcription errors and their corrections"""
    
    # Common ASR errors
    COMMON_ERRORS = {
        # Technical terms
        "AI": ["A.I.", "ay", "aye", "eye"],
        "API": ["A.P.I.", "epi"],
        "GPU": ["G.P.U.", "gee pee you"],
        "CPU": ["C.P.U.", "see pee you"],
        "MLX": ["M.L.X.", "emm el ex"],
        "TVET": ["T.V.E.T.", "tee vee eee tee"],
        
        # Countries and proper nouns
        "Vietnam": ["vietnam", "viet nam", "vietnan", "vietnum"],
        "Vietnamese": ["vietnamese", "viet namese"],
        "Ho Chi Minh": ["ho chi min", "ho chi ming", "hochi minh"],
        "Hanoi": ["ha noi", "hanoy"],
        
        # Education terms
        "vocational": ["vocation", "vocational"],
        "education": ["educasion", "educaton"],
        "training": ["trainin", "traning"],
        "competency": ["competancy", "competensy"],
        "assessment": ["assesment", "asessment"],
        
        # Business terms
        "ministry": ["ministy", "ministery"],
        "government": ["goverment", "govermment"],
        "implementation": ["implementasion", "implimentation"],
        "framework": ["framwork", "framewerk"],
        
        # Numbers and quantities
        "one": ["wan", "wun"],
        "two": ["too", "tow"],
        "three": ["tree", "thre"],
        "four": ["for", "fore"],
        "five": ["fiv", "fyve"],
        "six": ["siks", "six"],
        "seven": ["sevn", "sevan"],
        "eight": ["eght", "aight"],
        "nine": ["nyne", "nin"],
        "ten": ["tan", "tin"],
    }
    
    # Context-specific corrections
    CONTEXT_CORRECTIONS = {
        "education": {
            "teacher training": ["teachr training", "techer training"],
            "curriculum": ["curiculum", "curriculm"],
            "qualification": ["qualifcation", "qualificaton"],
            "certification": ["certifcation", "certificaton"],
        },
        "technology": {
            "artificial intelligence": ["artifical intelligence", "artificial inteligence"],
            "machine learning": ["machne learning", "machin learning"],
            "deep learning": ["dep learning", "deap learning"],
            "neural network": ["neurl network", "nural network"],
        }
    }
    
    # Filler word patterns to remove or reduce
    FILLER_PATTERNS = [
        r'\b(um|uh|er|ah|eh)\b',
        r'\b(like|you know|I mean|sort of|kind of)\b',
        r'\b(actually|basically|literally)\b(?=.*\b\1\b)',  # Repetitive adverbs
    ]
    
    # Punctuation normalization
    PUNCTUATION_FIXES = {
        r'\.{2,}': '.',  # Multiple dots to single
        r'\?{2,}': '?',  # Multiple question marks
        r'!{2,}': '!',   # Multiple exclamation marks
        r',{2,}': ',',   # Multiple commas
        r'\s+([,.!?;:])': r'\1',  # Remove space before punctuation
        r'([,.!?;:])\s*([,.!?;:])': r'\1\2',  # Remove duplicate punctuation
    }


class TranscriptionCorrector:
    """Advanced transcription correction engine"""
    
    def __init__(self, config: CorrectionConfig = None):
        self.config = config or CorrectionConfig()
        self.nlp = None
        self.grammar_corrector = None
        self.word_frequency = Counter()
        self.speaker_vocabulary = defaultdict(set)
        self._initialized = False
        
    async def initialize(self):
        """Initialize language models explicitly"""
        if not self._initialized:
            await self._initialize_models()
            self._initialized = True
    
    async def _initialize_models(self):
        """Initialize language processing models"""
        print("🔧 Initializing correction models...")
        
        # Initialize SpaCy for advanced processing
        if SPACY_AVAILABLE and self.config.enable_context_correction:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ SpaCy model loaded for context analysis")
            except OSError:
                print("⚠️  SpaCy model not found. Run: python -m spacy download en_core_web_sm")
        
        # Initialize grammar correction
        if TRANSFORMERS_AVAILABLE and self.config.enable_ai_correction:
            try:
                self.grammar_corrector = pipeline(
                    "text2text-generation",
                    model="pszemraj/flan-t5-large-grammar-synthesis",
                    device=0 if self._has_gpu() else -1
                )
                print("✅ AI grammar corrector loaded")
            except Exception as e:
                print(f"⚠️  AI corrector failed to load: {e}")
        
        # Initialize NLTK resources
        if NLTK_AVAILABLE:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                print("✅ NLTK resources downloaded")
            except Exception as e:
                print(f"⚠️  NLTK initialization failed: {e}")
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            # Check CUDA availability with error handling
            cuda_available = False
            try:
                cuda_available = torch.cuda.is_available()
            except Exception as e:
                logger.debug(f"CUDA check failed: {e}")
            
            # Check MPS availability with error handling
            mps_available = False
            try:
                mps_available = torch.backends.mps.is_available()
            except Exception as e:
                logger.debug(f"MPS check failed: {e}")
            
            return cuda_available or mps_available
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")
            return False
    
    async def correct_transcription(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Main correction pipeline for transcription results"""
        # Ensure models are initialized
        if not self._initialized:
            await self.initialize()
            
        print("🔧 Starting transcription correction...")
        start_time = time.time()
        
        # Extract text and segments
        original_text = result.get("text", "")
        segments = result.get("segments", [])
        
        if not original_text and not segments:
            print("⚠️  No text content to correct")
            return result
        
        # Correction pipeline
        corrected_result = result.copy()
        
        # 1. Basic text cleaning
        if segments:
            corrected_segments = await self._correct_segments(segments)
            corrected_result["segments"] = corrected_segments
            corrected_result["text"] = self._reconstruct_text_from_segments(corrected_segments)
        else:
            corrected_text = await self._correct_text(original_text)
            corrected_result["text"] = corrected_text
        
        # 2. Speaker consistency improvements
        if self.config.enable_speaker_consistency and segments:
            corrected_result["segments"] = self._improve_speaker_consistency(
                corrected_result["segments"]
            )
        
        # 3. Add correction metadata
        correction_time = time.time() - start_time
        corrected_result["correction_info"] = {
            "correction_time": correction_time,
            "corrections_applied": self._get_applied_corrections(),
            "original_word_count": len(original_text.split()),
            "corrected_word_count": len(corrected_result["text"].split()),
        }
        
        print(f"✅ Correction completed in {correction_time:.2f}s")
        return corrected_result
    
    async def _correct_segments(self, segments: List[Dict]) -> List[Dict]:
        """Correct individual segments"""
        corrected_segments = []
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            # Apply corrections
            corrected_text = await self._correct_text(text)
            
            # Create corrected segment
            corrected_segment = segment.copy()
            corrected_segment["text"] = corrected_text
            corrected_segment["original_text"] = text  # Keep original for reference
            
            corrected_segments.append(corrected_segment)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"🔧 Correcting segments: {i+1}/{len(segments)}")
        
        return corrected_segments
    
    async def _correct_text(self, text: str) -> str:
        """Apply comprehensive text corrections"""
        if not text.strip():
            return text
        
        corrected = text
        
        # 1. Basic cleaning
        corrected = self._basic_cleaning(corrected)
        
        # 2. Common error corrections
        if self.config.enable_noun_correction:
            corrected = self._correct_common_errors(corrected)
        
        # 3. Context-aware corrections
        if self.config.enable_context_correction:
            corrected = self._context_aware_correction(corrected)
        
        # 4. Repetition removal
        if self.config.fix_repetitions:
            corrected = self._fix_repetitions(corrected)
        
        # 5. Punctuation normalization
        if self.config.normalize_punctuation:
            corrected = self._normalize_punctuation(corrected)
        
        # 6. AI-powered grammar correction
        if self.config.enable_ai_correction and self.grammar_corrector:
            corrected = await self._ai_grammar_correction(corrected)
        
        # 7. Domain-specific corrections
        corrected = self._apply_domain_corrections(corrected)
        
        return corrected.strip()
    
    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove filler words (optional, configurable)
        for pattern in CommonCorrections.FILLER_PATTERNS[:1]:  # Only remove um, uh, etc.
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _correct_common_errors(self, text: str) -> str:
        """Correct common transcription errors"""
        for correct, errors in CommonCorrections.COMMON_ERRORS.items():
            for error in errors:
                # Case-insensitive replacement with word boundaries
                pattern = r'\b' + re.escape(error) + r'\b'
                text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
        
        return text
    
    def _context_aware_correction(self, text: str) -> str:
        """Apply context-aware corrections"""
        if not self.nlp:
            return text
        
        try:
            doc = self.nlp(text)
            corrected_tokens = []
            
            for token in doc:
                corrected_token = self._correct_token_in_context(token, doc)
                corrected_tokens.append(corrected_token)
            
            return " ".join(corrected_tokens)
        except Exception as e:
            logger.warning(f"Context correction failed: {e}")
            return text
    
    def _correct_token_in_context(self, token, doc) -> str:
        """Correct individual token based on context"""
        # Check if token is a common error
        for correct, errors in CommonCorrections.COMMON_ERRORS.items():
            if token.text.lower() in [e.lower() for e in errors]:
                return correct
        
        # Check context-specific corrections
        for domain, corrections in CommonCorrections.CONTEXT_CORRECTIONS.items():
            for correct_phrase, error_phrases in corrections.items():
                if any(error.lower() in token.text.lower() for error in error_phrases):
                    return correct_phrase.split()[0] if token.i == 0 else correct_phrase.split()[-1]
        
        return token.text
    
    def _fix_repetitions(self, text: str) -> str:
        """Remove repetitive phrases and words"""
        # Remove word repetitions (e.g., "the the" -> "the")
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        
        # Remove phrase repetitions
        words = text.split()
        if len(words) < 4:
            return text
        
        cleaned_words = []
        i = 0
        while i < len(words):
            # Check for 2-3 word phrase repetitions
            found_repetition = False
            for phrase_len in [3, 2]:
                if i + phrase_len * 2 <= len(words):
                    phrase1 = words[i:i+phrase_len]
                    phrase2 = words[i+phrase_len:i+phrase_len*2]
                    if phrase1 == phrase2:
                        cleaned_words.extend(phrase1)
                        i += phrase_len * 2
                        found_repetition = True
                        break
            
            if not found_repetition:
                cleaned_words.append(words[i])
                i += 1
        
        return " ".join(cleaned_words)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation"""
        for pattern, replacement in CommonCorrections.PUNCTUATION_FIXES.items():
            text = re.sub(pattern, replacement, text)
        
        # Ensure proper sentence capitalization
        sentences = re.split(r'[.!?]+', text)
        capitalized_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                capitalized_sentences.append(sentence)
        
        return '. '.join(capitalized_sentences) + '.' if capitalized_sentences else text
    
    async def _ai_grammar_correction(self, text: str) -> str:
        """Apply AI-powered grammar correction"""
        if not self.grammar_corrector or len(text) > self.config.max_segment_length:
            return text
        
        try:
            prompt = f"Correct grammar and improve clarity: {text}"
            result = self.grammar_corrector(prompt, max_length=len(text) + 50)
            return result[0]["generated_text"] if result else text
        except Exception as e:
            logger.warning(f"AI grammar correction failed: {e}")
            return text
    
    def _apply_domain_corrections(self, text: str) -> str:
        """Apply domain-specific corrections"""
        for domain, terms in self.config.domain_terms.items():
            for term in terms:
                # Ensure proper capitalization of domain terms
                pattern = r'\b' + re.escape(term) + r'\b'
                text = re.sub(pattern, term, text, flags=re.IGNORECASE)
        
        # Apply custom vocabulary
        for term in self.config.custom_vocabulary:
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            text = re.sub(pattern, term, text, flags=re.IGNORECASE)
        
        return text
    
    def _improve_speaker_consistency(self, segments: List[Dict]) -> List[Dict]:
        """Improve speaker label consistency"""
        # Build speaker vocabulary
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
            if speaker != "Unknown":
                self.speaker_vocabulary[speaker].update(text.lower().split())
        
        # Apply speaker-specific corrections
        improved_segments = []
        for segment in segments:
            improved_segment = segment.copy()
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
            
            if speaker in self.speaker_vocabulary:
                # Apply speaker-specific vocabulary preferences
                improved_text = self._apply_speaker_vocabulary(text, speaker)
                improved_segment["text"] = improved_text
            
            improved_segments.append(improved_segment)
        
        return improved_segments
    
    def _apply_speaker_vocabulary(self, text: str, speaker: str) -> str:
        """Apply speaker-specific vocabulary corrections"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            # If word is commonly used by this speaker, prefer their version
            if word.lower() in self.speaker_vocabulary[speaker]:
                corrected_words.append(word)
            else:
                # Check if there's a similar word in speaker's vocabulary
                similar = self._find_similar_word(word, self.speaker_vocabulary[speaker])
                corrected_words.append(similar if similar else word)
        
        return " ".join(corrected_words)
    
    def _find_similar_word(self, word: str, vocabulary: Set[str]) -> Optional[str]:
        """Find similar word in vocabulary"""
        word_lower = word.lower()
        for vocab_word in vocabulary:
            if self._is_similar(word_lower, vocab_word):
                return vocab_word
        return None
    
    def _is_similar(self, word1: str, word2: str, threshold: float = 0.8) -> bool:
        """Check if two words are similar"""
        if len(word1) != len(word2):
            return False
        
        matches = sum(c1 == c2 for c1, c2 in zip(word1, word2))
        similarity = matches / len(word1)
        return similarity >= threshold
    
    def _reconstruct_text_from_segments(self, segments: List[Dict]) -> str:
        """Reconstruct full text from corrected segments"""
        texts = []
        for segment in segments:
            text = segment.get("text", "").strip()
            if text:
                texts.append(text)
        return " ".join(texts)
    
    def _get_applied_corrections(self) -> List[str]:
        """Get list of correction types that were applied"""
        corrections = []
        if self.config.enable_noun_correction:
            corrections.append("noun_correction")
        if self.config.enable_context_correction:
            corrections.append("context_correction")
        if self.config.enable_speaker_consistency:
            corrections.append("speaker_consistency")
        if self.config.enable_grammar_correction:
            corrections.append("grammar_correction")
        if self.config.enable_ai_correction:
            corrections.append("ai_correction")
        return corrections


class CorrectionIntegrator:
    """Integration layer for transcription correction"""
    
    def __init__(self, config: CorrectionConfig = None):
        self.corrector = TranscriptionCorrector(config)
    
    async def enhance_lightning_mlx_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance Lightning MLX transcription results"""
        print("🎯 Enhancing Lightning MLX transcription with corrections...")
        
        # Apply corrections
        enhanced_result = await self.corrector.correct_transcription(result)
        
        # Add enhancement metadata
        enhanced_result["enhanced"] = True
        enhanced_result["enhancement_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return enhanced_result
    
    async def enhance_segments(self, segments: List[Dict]) -> List[Dict]:
        """Enhance segment-based transcription"""
        dummy_result = {"segments": segments, "text": ""}
        enhanced = await self.corrector.correct_transcription(dummy_result)
        return enhanced["segments"]
    
    async def enhance_text(self, text: str) -> str:
        """Enhance plain text transcription"""
        dummy_result = {"text": text, "segments": []}
        enhanced = await self.corrector.correct_transcription(dummy_result)
        return enhanced["text"]


# Integration functions for existing scripts
async def enhance_transcription_result(result: Dict[str, Any], 
                                     config: CorrectionConfig = None) -> Dict[str, Any]:
    """
    Main function to enhance any transcription result
    Compatible with Lightning MLX, Whisper, and other transcription outputs
    """
    integrator = CorrectionIntegrator(config)
    return await integrator.enhance_lightning_mlx_result(result)


def create_correction_config(domain: str = "education", 
                           enable_ai: bool = True,
                           custom_terms: List[str] = None) -> CorrectionConfig:
    """Create a correction configuration for specific domains"""
    config = CorrectionConfig()
    
    # Domain-specific settings
    if domain == "education":
        config.domain_terms["education"].extend([
            "CBTA", "competency-based", "vocational", "TVET", "assessment"
        ])
    elif domain == "technology":
        config.domain_terms["technology"].extend([
            "semiconductor", "AI", "MLX", "GPU", "CPU", "digital transformation"
        ])
    elif domain == "business":
        config.domain_terms["business"].extend([
            "ministry", "policy", "implementation", "framework", "nationwide"
        ])
    
    # Add custom terms
    if custom_terms:
        config.custom_vocabulary.extend(custom_terms)
    
    # AI settings
    config.enable_ai_correction = enable_ai and TRANSFORMERS_AVAILABLE
    
    return config


if __name__ == "__main__":
    # Example usage
    async def test_corrector():
        config = create_correction_config("education")
        corrector = TranscriptionCorrector(config)
        
        # Test text with common errors
        test_text = "um, the ministy of educasion in vietnam is working on um vocational trainin"
        
        corrected = await corrector._correct_text(test_text)
        print(f"Original: {test_text}")
        print(f"Corrected: {corrected}")
    
    asyncio.run(test_corrector()) 