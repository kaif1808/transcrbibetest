#!/usr/bin/env python3
"""
Advanced Context Analysis Module
Provides comprehensive analysis of transcribed audio content including:
- Speaker identification and profiling
- Executive summaries and key insights
- Topic modeling and theme extraction
- Sentiment analysis and emotional context
- Timeline analysis and content structure
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import logging

# Core NLP and analysis imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

@dataclass
class SpeakerProfile:
    """Detailed speaker profile with characteristics and patterns"""
    speaker_id: str
    total_duration: float = 0.0
    word_count: int = 0
    segment_count: int = 0
    speaking_rate: float = 0.0  # words per minute
    topics: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    sentiment_scores: Dict[str, float] = field(default_factory=dict)
    dominant_emotions: List[str] = field(default_factory=list)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    role_indicators: List[str] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)

@dataclass
class ConversationInsight:
    """Key insights extracted from conversation analysis"""
    main_topics: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    important_dates: List[str] = field(default_factory=list)
    mentioned_people: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    sentiment_timeline: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ContextAnalysisResult:
    """Complete context analysis results"""
    executive_summary: str = ""
    speaker_profiles: Dict[str, SpeakerProfile] = field(default_factory=dict)
    conversation_insights: ConversationInsight = field(default_factory=ConversationInsight)
    timeline_analysis: List[Dict[str, Any]] = field(default_factory=list)
    interaction_network: Dict[str, Any] = field(default_factory=dict)
    content_structure: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedContextAnalyzer:
    """Advanced context analysis engine for transcribed audio content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self.sentiment_analyzer = None
        self.emotion_analyzer = None
        self.summarizer = None
        
        # Initialize NLP models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize NLP models and analyzers"""
        try:
            # Load spaCy model
            if SPACY_AVAILABLE:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("✅ SpaCy model loaded successfully")
            
            # Load sentiment analysis model
            if TRANSFORMERS_AVAILABLE:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Load emotion analysis model
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Load summarization model
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.logger.info("✅ Transformer models loaded successfully")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Some models could not be loaded: {e}")
    
    async def analyze_transcription(
        self,
        transcription_data: Dict[str, Any],
        output_dir: str = "output"
    ) -> ContextAnalysisResult:
        """
        Perform comprehensive context analysis on transcription data
        
        Args:
            transcription_data: Complete transcription with segments and speakers
            output_dir: Directory to save analysis results
            
        Returns:
            ContextAnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        
        self.logger.info("🔍 Starting comprehensive context analysis...")
        
        # Extract segments and basic info
        segments = transcription_data.get("segments", [])
        if not segments:
            self.logger.warning("⚠️ No segments found in transcription data")
            return ContextAnalysisResult()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Initialize result object
        result = ContextAnalysisResult()
        
        # Perform analysis steps
        await self._analyze_speakers(segments, result)
        await self._extract_conversation_insights(segments, result)
        await self._analyze_content_structure(segments, result)
        await self._analyze_interactions(segments, result)
        await self._generate_executive_summary(segments, result)
        await self._calculate_quality_metrics(segments, result)
        
        # Save detailed results
        await self._save_analysis_results(result, output_path)
        
        # Update processing metadata
        result.processing_metadata = {
            "analysis_duration": time.time() - start_time,
            "total_segments": len(segments),
            "models_used": self._get_models_status(),
            "output_directory": str(output_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.logger.info(f"✅ Context analysis completed in {result.processing_metadata['analysis_duration']:.2f}s")
        
        return result
    
    async def _analyze_speakers(self, segments: List[Dict], result: ContextAnalysisResult):
        """Analyze speaker characteristics and patterns"""
        self.logger.info("👥 Analyzing speaker profiles...")
        
        speaker_data = defaultdict(list)
        
        # Group segments by speaker
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            speaker_data[speaker].append(segment)
        
        # Analyze each speaker
        for speaker_id, speaker_segments in speaker_data.items():
            profile = SpeakerProfile(speaker_id=speaker_id)
            
            # Calculate basic metrics
            profile.segment_count = len(speaker_segments)
            profile.total_duration = sum(
                seg.get("end", 0) - seg.get("start", 0) 
                for seg in speaker_segments
            )
            
            # Combine all text for analysis
            speaker_text = " ".join(seg.get("text", "") for seg in speaker_segments)
            profile.word_count = len(speaker_text.split())
            
            # Calculate speaking rate (words per minute)
            if profile.total_duration > 0:
                profile.speaking_rate = (profile.word_count / profile.total_duration) * 60
            
            # Extract topics and key phrases
            if self.nlp and speaker_text:
                doc = self.nlp(speaker_text)
                
                # Extract key phrases (noun phrases)
                profile.key_phrases = [
                    chunk.text.strip() 
                    for chunk in doc.noun_chunks 
                    if len(chunk.text.strip()) > 3
                ][:10]  # Top 10 phrases
                
                # Extract entities as expertise areas
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                profile.expertise_areas = [
                    ent[0] for ent in entities 
                    if ent[1] in ["ORG", "PRODUCT", "EVENT", "LAW", "GPE"]
                ][:5]
                
                # Identify role indicators
                role_patterns = [
                    r"\b(manager|director|CEO|president|lead|head|senior|junior)\b",
                    r"\b(responsible for|in charge of|managing|overseeing)\b",
                    r"\b(expert in|specialist in|consultant)\b"
                ]
                
                for pattern in role_patterns:
                    matches = re.findall(pattern, speaker_text, re.IGNORECASE)
                    profile.role_indicators.extend(matches)
            
            # Sentiment analysis
            if self.sentiment_analyzer and speaker_text:
                try:
                    sentiment_results = self.sentiment_analyzer(speaker_text[:512])  # Truncate for model
                    if sentiment_results:
                        profile.sentiment_scores = {
                            result["label"]: result["score"] 
                            for result in sentiment_results
                        }
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis failed for {speaker_id}: {e}")
            
            # Emotion analysis
            if self.emotion_analyzer and speaker_text:
                try:
                    emotion_results = self.emotion_analyzer(speaker_text[:512])
                    if emotion_results:
                        profile.dominant_emotions = [
                            result["label"] 
                            for result in sorted(emotion_results, key=lambda x: x["score"], reverse=True)[:3]
                        ]
                except Exception as e:
                    self.logger.warning(f"Emotion analysis failed for {speaker_id}: {e}")
            
            result.speaker_profiles[speaker_id] = profile
        
        self.logger.info(f"✅ Analyzed {len(result.speaker_profiles)} speaker profiles")
    
    async def _extract_conversation_insights(self, segments: List[Dict], result: ContextAnalysisResult):
        """Extract key insights from the conversation"""
        self.logger.info("💡 Extracting conversation insights...")
        
        # Combine all text
        full_text = " ".join(seg.get("text", "") for seg in segments)
        
        insights = ConversationInsight()
        
        if self.nlp and full_text:
            doc = self.nlp(full_text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    insights.mentioned_people.append(ent.text)
                elif ent.label_ == "ORG":
                    insights.organizations.append(ent.text)
                elif ent.label_ == "DATE":
                    insights.important_dates.append(ent.text)
            
            # Remove duplicates and limit
            insights.mentioned_people = list(set(insights.mentioned_people))[:10]
            insights.organizations = list(set(insights.organizations))[:10]
            insights.important_dates = list(set(insights.important_dates))[:10]
            
            # Extract technical terms (nouns that appear frequently)
            nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN" and len(token.text) > 3]
            noun_counts = Counter(nouns)
            insights.technical_terms = [term for term, count in noun_counts.most_common(15)]
        
        # Extract action items and decisions (pattern-based)
        action_patterns = [
            r"(?:we need to|must|should|will|going to|action item|next step)\s+([^.!?]+)",
            r"(?:decision|decide|agreed|concluded)\s+([^.!?]+)",
            r"(?:follow up|follow-up)\s+([^.!?]+)"
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            insights.action_items.extend(matches[:5])  # Limit to 5 per pattern
        
        # Topic modeling with TF-IDF if available
        if SKLEARN_AVAILABLE and len(segments) > 5:
            try:
                segment_texts = [seg.get("text", "") for seg in segments if seg.get("text")]
                
                if len(segment_texts) >= 3:
                    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(segment_texts)
                    
                    # Get top terms
                    feature_names = vectorizer.get_feature_names_out()
                    scores = tfidf_matrix.sum(axis=0).A1
                    top_terms = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
                    
                    insights.main_topics = [term for term, score in top_terms[:10]]
                    
            except Exception as e:
                self.logger.warning(f"Topic modeling failed: {e}")
        
        # Sentiment timeline
        for i, segment in enumerate(segments):
            if self.sentiment_analyzer and segment.get("text"):
                try:
                    sentiment = self.sentiment_analyzer(segment["text"][:512])
                    if sentiment:
                        insights.sentiment_timeline.append({
                            "segment_index": i,
                            "timestamp": segment.get("start", 0),
                            "speaker": segment.get("speaker", "Unknown"),
                            "sentiment": sentiment[0]["label"],
                            "confidence": sentiment[0]["score"]
                        })
                except Exception as e:
                    continue
        
        result.conversation_insights = insights
        self.logger.info("✅ Conversation insights extracted")
    
    async def _analyze_content_structure(self, segments: List[Dict], result: ContextAnalysisResult):
        """Analyze the structural patterns of the conversation"""
        self.logger.info("📊 Analyzing content structure...")
        
        structure = {
            "total_duration": 0,
            "speaking_distribution": {},
            "conversation_flow": [],
            "topic_transitions": [],
            "silence_patterns": [],
            "interruption_count": 0
        }
        
        if not segments:
            result.content_structure = structure
            return
        
        # Calculate total duration
        if segments:
            structure["total_duration"] = segments[-1].get("end", 0) - segments[0].get("start", 0)
        
        # Speaking time distribution
        speaker_times = defaultdict(float)
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            duration = segment.get("end", 0) - segment.get("start", 0)
            speaker_times[speaker] += duration
        
        total_speaking_time = sum(speaker_times.values())
        if total_speaking_time > 0:
            structure["speaking_distribution"] = {
                speaker: (time / total_speaking_time) * 100
                for speaker, time in speaker_times.items()
            }
        
        # Conversation flow analysis
        current_speaker = None
        turn_start = 0
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            start_time = segment.get("start", 0)
            
            if speaker != current_speaker:
                if current_speaker:
                    structure["conversation_flow"].append({
                        "speaker": current_speaker,
                        "start": turn_start,
                        "end": start_time,
                        "duration": start_time - turn_start
                    })
                current_speaker = speaker
                turn_start = start_time
        
        # Add final segment
        if current_speaker and segments:
            structure["conversation_flow"].append({
                "speaker": current_speaker,
                "start": turn_start,
                "end": segments[-1].get("end", 0),
                "duration": segments[-1].get("end", 0) - turn_start
            })
        
        # Detect silence patterns (gaps between segments)
        for i in range(len(segments) - 1):
            current_end = segments[i].get("end", 0)
            next_start = segments[i + 1].get("start", 0)
            gap = next_start - current_end
            
            if gap > 2.0:  # Silence longer than 2 seconds
                structure["silence_patterns"].append({
                    "start": current_end,
                    "end": next_start,
                    "duration": gap
                })
        
        result.content_structure = structure
        self.logger.info("✅ Content structure analyzed")
    
    async def _analyze_interactions(self, segments: List[Dict], result: ContextAnalysisResult):
        """Analyze speaker interactions and relationships"""
        self.logger.info("🔗 Analyzing speaker interactions...")
        
        if not NETWORKX_AVAILABLE:
            self.logger.warning("NetworkX not available, skipping interaction analysis")
            return
        
        # Create interaction graph
        G = nx.DiGraph()
        
        # Add speakers as nodes
        speakers = set(seg.get("speaker", "Unknown") for seg in segments)
        G.add_nodes_from(speakers)
        
        # Analyze conversation turns to build edges
        for i in range(len(segments) - 1):
            current_speaker = segments[i].get("speaker", "Unknown")
            next_speaker = segments[i + 1].get("speaker", "Unknown")
            
            if current_speaker != next_speaker:
                if G.has_edge(current_speaker, next_speaker):
                    G[current_speaker][next_speaker]["weight"] += 1
                else:
                    G.add_edge(current_speaker, next_speaker, weight=1)
        
        # Calculate interaction metrics
        interaction_data = {
            "total_interactions": G.number_of_edges(),
            "speaker_centrality": {},
            "interaction_matrix": {},
            "dominant_speakers": [],
            "conversation_leaders": []
        }
        
        if G.number_of_nodes() > 0:
            # Calculate centrality measures
            try:
                centrality = nx.degree_centrality(G)
                interaction_data["speaker_centrality"] = centrality
                
                # Identify dominant speakers
                sorted_speakers = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                interaction_data["dominant_speakers"] = [speaker for speaker, score in sorted_speakers[:3]]
                
            except Exception as e:
                self.logger.warning(f"Centrality calculation failed: {e}")
        
        # Build interaction matrix
        for speaker1 in speakers:
            interaction_data["interaction_matrix"][speaker1] = {}
            for speaker2 in speakers:
                if G.has_edge(speaker1, speaker2):
                    interaction_data["interaction_matrix"][speaker1][speaker2] = G[speaker1][speaker2]["weight"]
                else:
                    interaction_data["interaction_matrix"][speaker1][speaker2] = 0
        
        result.interaction_network = interaction_data
        self.logger.info("✅ Speaker interactions analyzed")
    
    async def _generate_executive_summary(self, segments: List[Dict], result: ContextAnalysisResult):
        """Generate an executive summary of the conversation"""
        self.logger.info("📝 Generating executive summary...")
        
        # Combine all text
        full_text = " ".join(seg.get("text", "") for seg in segments)
        
        if not full_text.strip():
            result.executive_summary = "No content available for summary generation."
            return
        
        # Use transformer summarizer if available
        if self.summarizer:
            try:
                # Split text into chunks if too long
                max_chunk_length = 1024
                chunks = [full_text[i:i+max_chunk_length] for i in range(0, len(full_text), max_chunk_length)]
                
                summaries = []
                for chunk in chunks[:3]:  # Limit to first 3 chunks
                    if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                        summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                        if summary:
                            summaries.append(summary[0]["summary_text"])
                
                if summaries:
                    combined_summary = " ".join(summaries)
                    
                    # Add context information
                    speaker_count = len(result.speaker_profiles)
                    duration_minutes = result.content_structure.get("total_duration", 0) / 60
                    
                    executive_summary = f"""
EXECUTIVE SUMMARY

Duration: {duration_minutes:.1f} minutes
Participants: {speaker_count} speakers
Main Content: {combined_summary}

KEY PARTICIPANTS:
"""
                    
                    # Add speaker information
                    for speaker_id, profile in result.speaker_profiles.items():
                        speaking_time = profile.total_duration / 60
                        executive_summary += f"• {speaker_id}: {speaking_time:.1f} min speaking time, {profile.word_count} words\n"
                    
                    # Add key insights
                    if result.conversation_insights.main_topics:
                        executive_summary += f"\nMAIN TOPICS: {', '.join(result.conversation_insights.main_topics[:5])}\n"
                    
                    if result.conversation_insights.organizations:
                        executive_summary += f"\nORGANIZATIONS MENTIONED: {', '.join(result.conversation_insights.organizations[:5])}\n"
                    
                    if result.conversation_insights.action_items:
                        executive_summary += f"\nACTION ITEMS: {len(result.conversation_insights.action_items)} identified\n"
                    
                    result.executive_summary = executive_summary.strip()
                else:
                    result.executive_summary = self._generate_fallback_summary(segments, result)
                    
            except Exception as e:
                self.logger.warning(f"AI summarization failed: {e}")
                result.executive_summary = self._generate_fallback_summary(segments, result)
        else:
            result.executive_summary = self._generate_fallback_summary(segments, result)
        
        self.logger.info("✅ Executive summary generated")
    
    def _generate_fallback_summary(self, segments: List[Dict], result: ContextAnalysisResult) -> str:
        """Generate a basic summary when AI summarization is not available"""
        speaker_count = len(result.speaker_profiles)
        total_words = sum(profile.word_count for profile in result.speaker_profiles.values())
        duration_minutes = result.content_structure.get("total_duration", 0) / 60
        
        summary = f"""
EXECUTIVE SUMMARY (Basic Analysis)

• Duration: {duration_minutes:.1f} minutes
• Participants: {speaker_count} speakers
• Total words: {total_words:,}
• Content segments: {len(segments)}

SPEAKER BREAKDOWN:
"""
        
        for speaker_id, profile in result.speaker_profiles.items():
            speaking_time = profile.total_duration / 60
            percentage = (profile.total_duration / result.content_structure.get("total_duration", 1)) * 100
            summary += f"• {speaker_id}: {speaking_time:.1f} min ({percentage:.1f}% of conversation)\n"
        
        if result.conversation_insights.main_topics:
            summary += f"\nKEY TOPICS: {', '.join(result.conversation_insights.main_topics[:5])}"
        
        return summary.strip()
    
    async def _calculate_quality_metrics(self, segments: List[Dict], result: ContextAnalysisResult):
        """Calculate quality and confidence metrics for the analysis"""
        self.logger.info("📊 Calculating quality metrics...")
        
        metrics = {
            "transcription_confidence": 0.0,
            "speaker_identification_confidence": 0.0,
            "content_completeness": 0.0,
            "analysis_depth": 0.0
        }
        
        # Transcription confidence (based on segment confidence if available)
        confidences = [seg.get("confidence", 1.0) for seg in segments if "confidence" in seg]
        if confidences:
            metrics["transcription_confidence"] = sum(confidences) / len(confidences)
        else:
            metrics["transcription_confidence"] = 0.85  # Default estimate
        
        # Speaker identification confidence
        identified_speakers = len([s for s in result.speaker_profiles.keys() if s != "Unknown"])
        total_speakers = len(result.speaker_profiles)
        if total_speakers > 0:
            metrics["speaker_identification_confidence"] = identified_speakers / total_speakers
        
        # Content completeness (based on presence of various analysis components)
        completeness_factors = [
            len(result.conversation_insights.main_topics) > 0,
            len(result.speaker_profiles) > 0,
            len(result.executive_summary) > 100,
            len(result.conversation_insights.mentioned_people) > 0,
            result.content_structure.get("total_duration", 0) > 0
        ]
        metrics["content_completeness"] = sum(completeness_factors) / len(completeness_factors)
        
        # Analysis depth (based on advanced features used)
        depth_factors = [
            self.sentiment_analyzer is not None,
            self.emotion_analyzer is not None,
            self.nlp is not None,
            SKLEARN_AVAILABLE,
            NETWORKX_AVAILABLE
        ]
        metrics["analysis_depth"] = sum(depth_factors) / len(depth_factors)
        
        result.quality_metrics = metrics
        self.logger.info("✅ Quality metrics calculated")
    
    async def _save_analysis_results(self, result: ContextAnalysisResult, output_path: Path):
        """Save comprehensive analysis results to files"""
        self.logger.info(f"💾 Saving analysis results to {output_path}")
        
        # Save executive summary
        summary_file = output_path / "executive_summary.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("# Executive Summary\n\n")
            f.write(result.executive_summary)
            f.write("\n\n---\n*Generated by Advanced Context Analyzer*")
        
        # Save speaker profiles
        speakers_file = output_path / "speaker_profiles.json"
        speaker_data = {}
        for speaker_id, profile in result.speaker_profiles.items():
            speaker_data[speaker_id] = {
                "total_duration": profile.total_duration,
                "word_count": profile.word_count,
                "segment_count": profile.segment_count,
                "speaking_rate": profile.speaking_rate,
                "key_phrases": profile.key_phrases,
                "sentiment_scores": profile.sentiment_scores,
                "dominant_emotions": profile.dominant_emotions,
                "role_indicators": profile.role_indicators,
                "expertise_areas": profile.expertise_areas
            }
        
        with open(speakers_file, "w", encoding="utf-8") as f:
            json.dump(speaker_data, f, indent=2, ensure_ascii=False)
        
        # Save conversation insights
        insights_file = output_path / "conversation_insights.json"
        insights_data = {
            "main_topics": result.conversation_insights.main_topics,
            "key_decisions": result.conversation_insights.key_decisions,
            "action_items": result.conversation_insights.action_items,
            "important_dates": result.conversation_insights.important_dates,
            "mentioned_people": result.conversation_insights.mentioned_people,
            "organizations": result.conversation_insights.organizations,
            "technical_terms": result.conversation_insights.technical_terms,
            "sentiment_timeline": result.conversation_insights.sentiment_timeline
        }
        
        with open(insights_file, "w", encoding="utf-8") as f:
            json.dump(insights_data, f, indent=2, ensure_ascii=False)
        
        # Save complete analysis report
        complete_report = output_path / "complete_analysis_report.json"
        complete_data = {
            "executive_summary": result.executive_summary,
            "speaker_profiles": speaker_data,
            "conversation_insights": insights_data,
            "content_structure": result.content_structure,
            "interaction_network": result.interaction_network,
            "quality_metrics": result.quality_metrics,
            "processing_metadata": result.processing_metadata
        }
        
        with open(complete_report, "w", encoding="utf-8") as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)
        
        # Create a detailed markdown report
        detailed_report = output_path / "detailed_analysis_report.md"
        await self._create_detailed_markdown_report(result, detailed_report)
        
        self.logger.info("✅ Analysis results saved successfully")
    
    async def _create_detailed_markdown_report(self, result: ContextAnalysisResult, file_path: Path):
        """Create a comprehensive markdown report"""
        
        report_content = f"""# Comprehensive Audio Transcription Analysis Report

*Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}*

## Executive Summary

{result.executive_summary}

## Speaker Analysis

"""
        
        for speaker_id, profile in result.speaker_profiles.items():
            report_content += f"""### {speaker_id}

- **Speaking Time**: {profile.total_duration/60:.1f} minutes
- **Word Count**: {profile.word_count:,} words
- **Speaking Rate**: {profile.speaking_rate:.1f} words/minute
- **Segments**: {profile.segment_count}

**Key Phrases**: {', '.join(profile.key_phrases[:5])}

**Expertise Areas**: {', '.join(profile.expertise_areas[:3])}

**Dominant Emotions**: {', '.join(profile.dominant_emotions)}

"""
        
        report_content += f"""## Conversation Insights

### Main Topics
{', '.join(result.conversation_insights.main_topics[:10])}

### Organizations Mentioned
{', '.join(result.conversation_insights.organizations)}

### People Mentioned
{', '.join(result.conversation_insights.mentioned_people)}

### Important Dates
{', '.join(result.conversation_insights.important_dates)}

### Action Items
"""
        
        for i, item in enumerate(result.conversation_insights.action_items[:5], 1):
            report_content += f"{i}. {item}\n"
        
        report_content += f"""
## Content Structure

- **Total Duration**: {result.content_structure.get('total_duration', 0)/60:.1f} minutes
- **Speaking Distribution**:
"""
        
        for speaker, percentage in result.content_structure.get('speaking_distribution', {}).items():
            report_content += f"  - {speaker}: {percentage:.1f}%\n"
        
        report_content += f"""
## Quality Metrics

- **Transcription Confidence**: {result.quality_metrics.get('transcription_confidence', 0):.2%}
- **Speaker ID Confidence**: {result.quality_metrics.get('speaker_identification_confidence', 0):.2%}
- **Content Completeness**: {result.quality_metrics.get('content_completeness', 0):.2%}
- **Analysis Depth**: {result.quality_metrics.get('analysis_depth', 0):.2%}

---

*Report generated by Advanced Context Analyzer*
"""
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    
    def _get_models_status(self) -> Dict[str, bool]:
        """Get status of loaded models"""
        return {
            "spacy": self.nlp is not None,
            "sentiment_analyzer": self.sentiment_analyzer is not None,
            "emotion_analyzer": self.emotion_analyzer is not None,
            "summarizer": self.summarizer is not None,
            "sklearn": SKLEARN_AVAILABLE,
            "networkx": NETWORKX_AVAILABLE
        } 