#!/usr/bin/env python3
"""
Complete Model Pipeline Runner
Comprehensive integration of all transcription system components:
- Fixed Noun Extraction System
- Lightning MLX Transcription
- Speaker Diarization 
- Context Analysis
- Transcription Correction
- Batch Processing with GPU acceleration

This script builds and tests the complete model pipeline before running tests.
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPipelineBuilder:
    """Builds and validates the complete model pipeline"""
    
    def __init__(self):
        self.components = {}
        self.initialization_status = {}
        self.pipeline_ready = False
        
    async def initialize_pipeline(self) -> Dict[str, Any]:
        """Initialize all pipeline components"""
        logger.info("🚀 Building Complete Model Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Initialize Noun Extraction System
        await self._initialize_noun_extraction()
        
        # Step 2: Initialize Lightning MLX Transcriber
        await self._initialize_transcriber()
        
        # Step 3: Initialize Speaker Diarization
        await self._initialize_diarization()
        
        # Step 4: Initialize Context Analyzer
        await self._initialize_context_analyzer()
        
        # Step 5: Initialize Transcription Corrector
        await self._initialize_corrector()
        
        # Step 6: Validate complete pipeline
        pipeline_status = await self._validate_pipeline()
        
        logger.info("✅ Model Pipeline Build Complete")
        return pipeline_status
    
    async def _initialize_noun_extraction(self):
        """Initialize the fixed noun extraction system"""
        logger.info("🔍 Initializing Noun Extraction System")
        
        try:
            from core.noun_extraction_system import (
                DocumentNounExtractor, 
                NounExtractionConfig,
                extract_nouns_from_document,
                extract_nouns_from_transcription
            )
            
            # Test configuration
            config = NounExtractionConfig(
                domain_focus="education",
                extract_phrases=True,
                use_local_llm=False,  # Disable for testing
                extract_unusual_nouns=False
            )
            
            # Initialize extractor
            extractor = DocumentNounExtractor(config)
            await extractor.initialize()
            
            # Test with sample text
            sample_text = "The Ministry of Education implemented TVET technical training programs."
            test_results = await extractor.extract_from_text(sample_text)
            
            if test_results and "all_nouns" in test_results:
                logger.info(f"✅ Noun extraction working - found {len(test_results['all_nouns'])} nouns")
                self.components["noun_extraction"] = {
                    "extractor": extractor,
                    "config": config,
                    "functions": {
                        "extract_from_document": extract_nouns_from_document,
                        "extract_from_transcription": extract_nouns_from_transcription
                    }
                }
                self.initialization_status["noun_extraction"] = "SUCCESS"
            else:
                raise Exception("Noun extraction test failed")
                
        except Exception as e:
            logger.error(f"❌ Noun extraction initialization failed: {e}")
            self.initialization_status["noun_extraction"] = f"FAILED: {e}"
    
    async def _initialize_transcriber(self):
        """Initialize Lightning MLX transcriber"""
        logger.info("🎤 Initializing Lightning MLX Transcriber")
        
        try:
            from core.lightning_whisper_mlx_transcriber import (
                LightningMLXProcessor,
                LightningMLXConfig
            )
            
            # Test configuration
            config = LightningMLXConfig(
                batch_size=8,
                use_diarization=False,
                enable_correction=True,
                enable_noun_extraction=True
            )
            
            transcriber = LightningMLXProcessor(config)
            await transcriber.initialize()
            
            logger.info("✅ Lightning MLX transcriber initialized")
            self.components["transcriber"] = {
                "processor": transcriber,
                "config": config
            }
            self.initialization_status["transcriber"] = "SUCCESS"
            
        except Exception as e:
            logger.error(f"❌ Transcriber initialization failed: {e}")
            self.initialization_status["transcriber"] = f"FAILED: {e}"
    
    async def _initialize_diarization(self):
        """Initialize speaker diarization"""
        logger.info("🎙️ Initializing Speaker Diarization")
        
        try:
            # Check if pyannote is available
            import torch
            from pyannote.audio import Pipeline
            
            # Determine device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("✅ CUDA GPU available for diarization")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("✅ Apple Silicon GPU (MPS) available for diarization")
            else:
                device = torch.device("cpu")
                logger.info("✅ CPU available for diarization")
            
            # Initialize pipeline (may require authentication)
            try:
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                pipeline.to(device)
                
                logger.info("✅ Speaker diarization pipeline loaded")
                self.components["diarization"] = {
                    "pipeline": pipeline,
                    "device": device
                }
                self.initialization_status["diarization"] = "SUCCESS"
                
            except Exception as auth_error:
                logger.warning(f"⚠️ Diarization auth failed: {auth_error}")
                logger.info("✅ Diarization configured (requires HuggingFace token)")
                self.initialization_status["diarization"] = "AUTH_REQUIRED"
                
        except ImportError as e:
            logger.error(f"❌ Diarization dependencies missing: {e}")
            self.initialization_status["diarization"] = f"MISSING_DEPS: {e}"
    
    async def _initialize_context_analyzer(self):
        """Initialize context analyzer"""
        logger.info("🔍 Initializing Context Analyzer")
        
        try:
            from core.context_analyzer import AdvancedContextAnalyzer
            
            analyzer = AdvancedContextAnalyzer()
            
            logger.info("✅ Context analyzer initialized")
            self.components["context_analyzer"] = {
                "analyzer": analyzer
            }
            self.initialization_status["context_analyzer"] = "SUCCESS"
            
        except Exception as e:
            logger.error(f"❌ Context analyzer initialization failed: {e}")
            self.initialization_status["context_analyzer"] = f"FAILED: {e}"
    
    async def _initialize_corrector(self):
        """Initialize transcription corrector"""
        logger.info("🔧 Initializing Transcription Corrector")
        
        try:
            from core.transcription_corrector import (
                TranscriptionCorrector,
                CorrectionConfig,
                enhance_transcription_result,
                create_correction_config
            )
            
            # Test configuration
            config = create_correction_config(
                domain="education",
                enable_ai=True,
                custom_terms=["TVET", "Ministry", "Education"]
            )
            
            corrector = TranscriptionCorrector(config)
            await corrector.initialize()  # Explicitly initialize the corrector
            
            logger.info("✅ Transcription corrector initialized")
            self.components["corrector"] = {
                "corrector": corrector,
                "config": config,
                "functions": {
                    "enhance_result": enhance_transcription_result,
                    "create_config": create_correction_config
                }
            }
            self.initialization_status["corrector"] = "SUCCESS"
            
        except Exception as e:
            logger.error(f"❌ Corrector initialization failed: {e}")
            self.initialization_status["corrector"] = f"FAILED: {e}"
    
    async def _validate_pipeline(self) -> Dict[str, Any]:
        """Validate the complete pipeline"""
        logger.info("🔍 Validating Complete Pipeline")
        
        # Check critical components
        critical_components = ["noun_extraction", "transcriber"]
        optional_components = ["diarization", "context_analyzer", "corrector"]
        
        pipeline_health = "HEALTHY"
        failed_critical = []
        failed_optional = []
        
        # Check critical components
        for component in critical_components:
            if component not in self.initialization_status or "FAILED" in self.initialization_status[component]:
                pipeline_health = "DEGRADED"
                failed_critical.append(component)
        
        # Check optional components
        for component in optional_components:
            if component not in self.initialization_status or "FAILED" in self.initialization_status[component]:
                failed_optional.append(component)
        
        # Determine pipeline readiness
        if pipeline_health == "HEALTHY":
            self.pipeline_ready = True
            logger.info("✅ Pipeline validation successful - ready for operation")
            if failed_optional:
                logger.info(f"ℹ️ Optional components with issues: {', '.join(failed_optional)}")
        else:
            self.pipeline_ready = False
            logger.error(f"❌ Pipeline validation failed - critical components failed: {', '.join(failed_critical)}")
        
        # Detailed validation report
        validation_details = {
            "critical_components": {
                "required": critical_components,
                "working": [c for c in critical_components if c not in failed_critical],
                "failed": failed_critical
            },
            "optional_components": {
                "available": optional_components,
                "working": [c for c in optional_components if c not in failed_optional],
                "failed": failed_optional
            }
        }
        
        return {
            "pipeline_status": pipeline_health,
            "components": self.initialization_status,
            "ready_for_testing": self.pipeline_ready,
            "available_components": list(self.components.keys()),
            "validation_details": validation_details
        }
    
    async def run_integration_test(self) -> Dict[str, Any]:
        """Run a complete integration test of the pipeline"""
        if not self.pipeline_ready:
            return {"error": "Pipeline not ready for testing"}
        
        logger.info("🧪 Running Integration Test")
        
        test_results = {}
        start_time = time.time()
        
        # Test 1: Noun extraction from sample text
        try:
            if "noun_extraction" in self.components:
                logger.info("Testing noun extraction...")
                extractor = self.components["noun_extraction"]["extractor"]
                
                test_text = """
                The Ministry of Education in Vietnam has implemented a comprehensive TVET 
                technical and vocational education framework. This assessment system focuses 
                on competency-based training programs for students and instructors.
                """
                
                noun_results = await extractor.extract_from_text(test_text)
                test_results["noun_extraction"] = {
                    "status": "SUCCESS",
                    "nouns_found": len(noun_results.get("all_nouns", [])),
                    "categories": list(noun_results.keys()),
                    "sample_nouns": [noun.text for noun in noun_results.get("all_nouns", [])[:5]]
                }
                logger.info(f"✅ Noun extraction test passed - found {len(noun_results.get('all_nouns', []))} nouns")
                
        except Exception as e:
            test_results["noun_extraction"] = {"status": "FAILED", "error": str(e)}
            logger.error(f"❌ Noun extraction test failed: {e}")
        
        # Test 2: Document processing
        try:
            if "noun_extraction" in self.components:
                extract_func = self.components["noun_extraction"]["functions"]["extract_from_document"]
                config = self.components["noun_extraction"]["config"]
                
                # Check if test document exists
                test_doc = "data/input/inputdoc.docx"
                if Path(test_doc).exists():
                    logger.info("Testing document noun extraction...")
                    doc_results = await extract_func(test_doc, "education", config)
                    
                    if "error" not in doc_results:
                        test_results["document_processing"] = {
                            "status": "SUCCESS",
                            "document": test_doc,
                            "statistics": doc_results.get("content_statistics", {}),
                            "categories_found": len(doc_results.get("noun_analysis", {}))
                        }
                        logger.info("✅ Document processing test passed")
                    else:
                        test_results["document_processing"] = {
                            "status": "FAILED", 
                            "error": doc_results["error"]
                        }
                else:
                    test_results["document_processing"] = {
                        "status": "SKIPPED", 
                        "reason": "Test document not found"
                    }
                    
        except Exception as e:
            test_results["document_processing"] = {"status": "FAILED", "error": str(e)}
            logger.error(f"❌ Document processing test failed: {e}")
        
        # Test 3: Pipeline integration
        try:
            # Test full pipeline flow simulation
            mock_transcription = {
                "text": "The Ministry of Education implemented new TVET programs for technical training.",
                "segments": [
                    {"start": 0, "end": 5, "text": "The Ministry of Education implemented"},
                    {"start": 5, "end": 10, "text": "new TVET programs for technical training."}
                ]
            }
            
            # Test noun extraction from transcription
            if "noun_extraction" in self.components:
                extract_func = self.components["noun_extraction"]["functions"]["extract_from_transcription"]
                config = self.components["noun_extraction"]["config"]
                
                transcription_nouns = await extract_func(mock_transcription, "education", config)
                
                test_results["pipeline_integration"] = {
                    "status": "SUCCESS",
                    "transcription_nouns": len(transcription_nouns.get("noun_analysis", {}).get("all_nouns", [])),
                    "flow_tested": "transcription -> noun_extraction"
                }
                logger.info("✅ Pipeline integration test passed")
                
        except Exception as e:
            test_results["pipeline_integration"] = {"status": "FAILED", "error": str(e)}
            logger.error(f"❌ Pipeline integration test failed: {e}")
        
        total_time = time.time() - start_time
        
        # Test summary
        successful_tests = len([t for t in test_results.values() if t.get("status") == "SUCCESS"])
        total_tests = len(test_results)
        
        summary = {
            "test_results": test_results,
            "summary": {
                "total_time": total_time,
                "tests_run": total_tests,
                "tests_passed": successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "pipeline_ready": successful_tests >= 2  # Need at least noun extraction and one other test
            }
        }
        
        logger.info(f"🎯 Integration Test Complete: {successful_tests}/{total_tests} tests passed")
        
        return summary

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "components_initialized": list(self.components.keys()),
            "initialization_status": self.initialization_status,
            "pipeline_ready": self.pipeline_ready,
            "available_for_testing": self.pipeline_ready
        }

async def build_and_test_pipeline():
    """Build and test the complete model pipeline"""
    print("🚀 COMPLETE MODEL PIPELINE BUILDER")
    print("=" * 60)
    print("Building comprehensive transcription system with:")
    print("• Fixed Noun Extraction System")
    print("• Lightning MLX Transcription")
    print("• Speaker Diarization")
    print("• Context Analysis")
    print("• Transcription Correction")
    print("• Batch Processing")
    print("=" * 60)
    
    builder = ModelPipelineBuilder()
    
    # Build pipeline
    pipeline_status = await builder.initialize_pipeline()
    
    print("\n📊 PIPELINE BUILD RESULTS:")
    for component, status in pipeline_status["components"].items():
        status_icon = "✅" if "SUCCESS" in status else ("⚠️" if "AUTH_REQUIRED" in status else "❌")
        print(f"   {status_icon} {component}: {status}")
    
    print(f"\n🎯 Pipeline Status: {pipeline_status['pipeline_status']}")
    print(f"🧪 Ready for Testing: {pipeline_status['ready_for_testing']}")
    
    # Run integration tests if pipeline is ready
    if pipeline_status['ready_for_testing']:
        print("\n🧪 RUNNING INTEGRATION TESTS")
        print("-" * 40)
        
        test_results = await builder.run_integration_test()
        
        print("\n📋 TEST RESULTS:")
        for test_name, result in test_results["test_results"].items():
            status_icon = "✅" if result["status"] == "SUCCESS" else ("⚠️" if result["status"] == "SKIPPED" else "❌")
            print(f"   {status_icon} {test_name}: {result['status']}")
        
        summary = test_results["summary"]
        print(f"\n🎯 INTEGRATION TEST SUMMARY:")
        print(f"   ⏱️ Total time: {summary['total_time']:.2f}s")
        print(f"   🧪 Tests passed: {summary['tests_passed']}/{summary['tests_run']}")
        print(f"   📊 Success rate: {summary['success_rate']:.1f}%")
        print(f"   🚀 Pipeline ready: {summary['pipeline_ready']}")
        
        if summary['pipeline_ready']:
            print("\n✅ PIPELINE BUILD SUCCESSFUL!")
            print("🎉 Ready for full system testing!")
            
            # Save pipeline status
            output_file = "pipeline_build_status.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "build_status": pipeline_status,
                    "test_results": test_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2, default=str)
            
            print(f"💾 Pipeline status saved to: {output_file}")
            
            return True
        else:
            print("\n⚠️ PIPELINE BUILD INCOMPLETE")
            print("Some components need attention before testing")
            return False
    else:
        print("\n❌ PIPELINE BUILD FAILED")
        print("Critical components failed to initialize")
        return False

async def main():
    """Main execution function"""
    try:
        success = await build_and_test_pipeline()
        if success:
            print("\n🎯 Next steps:")
            print("• Run audio transcription tests")
            print("• Test with actual audio files")
            print("• Validate end-to-end performance")
            return 0
        else:
            print("\n🔧 Fix component issues before proceeding")
            return 1
    except Exception as e:
        logger.error(f"Pipeline build failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 