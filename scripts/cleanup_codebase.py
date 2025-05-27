#!/usr/bin/env python3
"""
Codebase Cleanup Script
Organizes files and prepares the system for new tests while preserving working components
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

def cleanup_output_directories():
    """Clean up output directories while preserving final results"""
    print("🧹 Cleaning up output directories...")
    
    # Archive successful test results
    archive_dir = Path("results/successful_tests")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Archive the latest successful integrated test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_archive = archive_dir / f"integrated_test_success_{timestamp}"
    success_archive.mkdir(exist_ok=True)
    
    # Preserve key successful results
    key_files = [
        "output_complete_test/complete_integrated_transcript.txt",
        "output_complete_test/complete_integrated_data.json",
        "output_complete_test/comprehensive_enhanced_transcript.txt",
        "output_complete_test/comprehensive_enhanced_data.json"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            shutil.copy2(file_path, success_archive / Path(file_path).name)
            print(f"  ✅ Archived: {Path(file_path).name}")
    
    # Clean up temporary files
    temp_patterns = [
        "output_complete_test/*_optimized_*.json",
        "output_complete_test/batch_summary_*.json",
        "output_complete_test/system_diagnostics.json",
        "output_complete_test/timing_diagnostics.json",
        "output_complete_test/complete_analysis_data.json"
    ]
    
    import glob
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            Path(file_path).unlink()
            print(f"  🗑️  Removed temp file: {Path(file_path).name}")

def cleanup_test_scripts():
    """Organize test scripts"""
    print("📝 Organizing test scripts...")
    
    # Move old test scripts to archive
    test_archive = Path("tests/archive")
    test_archive.mkdir(parents=True, exist_ok=True)
    
    old_test_files = [
        "enhanced_complete_transcription.py",
        "comprehensive_transcript_enhancer.py", 
        "test_system_diagnostics.py",
        "test_complete_system.py"
    ]
    
    for test_file in old_test_files:
        if Path(test_file).exists():
            shutil.move(test_file, test_archive / test_file)
            print(f"  📁 Archived: {test_file}")

def update_gitignore():
    """Update .gitignore for clean repository"""
    print("🔧 Updating .gitignore...")
    
    gitignore_additions = [
        "\n# Test Results Archive",
        "results/successful_tests/",
        "tests/archive/",
        "\n# Temporary test outputs", 
        "*_optimized_*.json",
        "batch_summary_*.json",
        "system_diagnostics.json",
        "timing_diagnostics.json"
    ]
    
    with open('.gitignore', 'r') as f:
        content = f.read()
    
    # Add new rules if not already present
    for rule in gitignore_additions:
        if rule.strip() and rule.strip() not in content:
            content += rule + "\n"
    
    with open('.gitignore', 'w') as f:
        f.write(content)
    
    print("  ✅ Updated .gitignore")

def create_test_ready_structure():
    """Ensure clean structure for new tests"""
    print("🏗️  Preparing clean test environment...")
    
    # Ensure core directories exist
    core_dirs = [
        "tests/integration",
        "tests/unit", 
        "results/test_outputs",
        "scripts/utilities",
        "docs/test_results"
    ]
    
    for dir_path in core_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  📁 Ensured directory: {dir_path}")
    
    # Move the working integrated test to tests/integration
    if Path("test_complete_integrated_system.py").exists():
        shutil.copy2("test_complete_integrated_system.py", "tests/integration/")
        print("  ✅ Copied working integrated test to tests/integration/")

def create_cleanup_summary():
    """Create summary of cleanup actions"""
    print("📊 Creating cleanup summary...")
    
    summary = {
        "cleanup_timestamp": datetime.now().isoformat(),
        "actions_performed": [
            "Archived successful test results",
            "Removed temporary files", 
            "Organized test scripts",
            "Updated .gitignore",
            "Prepared clean test environment"
        ],
        "preserved_files": [
            "src/ - Core system modules",
            "tests/integration/test_complete_integrated_system.py - Working integrated test",
            "results/successful_tests/ - Archived successful results",
            "data/ - Input data and models",
            "requirements/ - Dependency specifications"
        ],
        "ready_for": "New comprehensive tests with clean environment"
    }
    
    with open("CLEANUP_STATUS.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("  ✅ Created CLEANUP_STATUS.json")

def main():
    """Main cleanup function"""
    print("🚀" * 30)
    print("  CODEBASE CLEANUP UTILITY")
    print("  Preparing for New Tests")
    print("🚀" * 30)
    
    cleanup_output_directories()
    cleanup_test_scripts() 
    update_gitignore()
    create_test_ready_structure()
    create_cleanup_summary()
    
    print("\n✅ CLEANUP COMPLETED SUCCESSFULLY!")
    print("📁 Codebase is now organized and ready for new tests")
    print("🔧 Working components preserved in tests/integration/")
    print("📊 Successful results archived in results/successful_tests/")

if __name__ == "__main__":
    main() 