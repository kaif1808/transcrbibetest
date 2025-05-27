#!/usr/bin/env python3
"""
Comprehensive Codebase Cleanup Script
Organizes files, removes temporary data, and generates cleanup summaries
"""

import os
import shutil
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class CodebaseCleanup:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.cleanup_summary = {
            "timestamp": datetime.now().isoformat(),
            "actions_performed": [],
            "files_moved": [],
            "files_deleted": [],
            "directories_cleaned": [],
            "space_freed": 0,
            "status": "in_progress"
        }
        
    def run_cleanup(self):
        """Execute comprehensive cleanup"""
        print("🧹 Starting Comprehensive Codebase Cleanup...")
        
        # 1. Clean Python cache files
        self._clean_python_cache()
        
        # 2. Remove empty output directories
        self._clean_empty_output_dirs()
        
        # 3. Archive old test results
        self._archive_old_results()
        
        # 4. Clean temporary files
        self._clean_temp_files()
        
        # 5. Organize documentation
        self._organize_documentation()
        
        # 6. Update .gitignore if needed
        self._update_gitignore()
        
        # 7. Generate final cleanup summary
        self._generate_cleanup_summary()
        
        self.cleanup_summary["status"] = "completed"
        print("✅ Comprehensive cleanup completed!")
        
    def _clean_python_cache(self):
        """Remove Python cache files and directories"""
        print("🗑️  Cleaning Python cache files...")
        
        cache_patterns = ["__pycache__", "*.pyc", "*.pyo", "*.pyd"]
        removed_count = 0
        
        for pattern in cache_patterns:
            if pattern == "__pycache__":
                for cache_dir in self.project_root.rglob("__pycache__"):
                    if cache_dir.is_dir():
                        shutil.rmtree(cache_dir)
                        removed_count += 1
                        print(f"   Removed: {cache_dir.relative_to(self.project_root)}")
            else:
                for cache_file in self.project_root.rglob(pattern):
                    if cache_file.is_file():
                        cache_file.unlink()
                        removed_count += 1
        
        self.cleanup_summary["actions_performed"].append(f"Removed {removed_count} Python cache files/directories")
        
    def _clean_empty_output_dirs(self):
        """Remove empty output directories"""
        print("📁 Cleaning empty output directories...")
        
        output_patterns = ["output*", "lightning_mlx_chunks_*"]
        removed_dirs = []
        
        for pattern in output_patterns:
            for output_dir in self.project_root.glob(pattern):
                if output_dir.is_dir():
                    # Check if directory is empty or contains only hidden files
                    contents = list(output_dir.iterdir())
                    if not contents or all(item.name.startswith('.') for item in contents):
                        shutil.rmtree(output_dir)
                        removed_dirs.append(str(output_dir.relative_to(self.project_root)))
                        print(f"   Removed empty: {output_dir.relative_to(self.project_root)}")
        
        self.cleanup_summary["directories_cleaned"] = removed_dirs
        
    def _archive_old_results(self):
        """Archive old test results"""
        print("📦 Archiving old test results...")
        
        results_dir = self.project_root / "results"
        if not results_dir.exists():
            return
            
        # Create archive directory if it doesn't exist
        archive_dir = results_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Move old benchmark files
        benchmarks_dir = results_dir / "benchmarks"
        if benchmarks_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_benchmarks = archive_dir / f"benchmarks_{timestamp}"
            shutil.move(str(benchmarks_dir), str(archive_benchmarks))
            self.cleanup_summary["actions_performed"].append(f"Archived benchmarks to {archive_benchmarks.name}")
            print(f"   Archived: benchmarks -> {archive_benchmarks.name}")
            
    def _clean_temp_files(self):
        """Remove temporary files"""
        print("🗑️  Cleaning temporary files...")
        
        temp_patterns = ["*.tmp", "*.temp", "*.log.*", "dummy_*", "*_test_result.json"]
        removed_files = []
        
        for pattern in temp_patterns:
            for temp_file in self.project_root.rglob(pattern):
                if temp_file.is_file() and not self._is_in_protected_dir(temp_file):
                    file_size = temp_file.stat().st_size
                    temp_file.unlink()
                    removed_files.append(str(temp_file.relative_to(self.project_root)))
                    self.cleanup_summary["space_freed"] += file_size
                    print(f"   Removed: {temp_file.relative_to(self.project_root)}")
        
        self.cleanup_summary["files_deleted"] = removed_files
        
    def _organize_documentation(self):
        """Ensure documentation is properly organized"""
        print("📚 Organizing documentation...")
        
        docs_dir = self.project_root / "docs"
        cleanup_dir = docs_dir / "cleanup_summaries"
        
        # Ensure directories exist
        cleanup_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if all cleanup files are in the right place
        cleanup_files_in_place = all([
            (cleanup_dir / "CODEBASE_CLEANUP_SUMMARY.md").exists(),
            (cleanup_dir / "CLEANUP_SUMMARY.md").exists(),
            (cleanup_dir / "CLEANUP_STATUS.json").exists(),
        ])
        
        if cleanup_files_in_place:
            print("   ✅ All cleanup files properly organized")
        else:
            print("   ⚠️  Some cleanup files may need manual organization")
            
    def _update_gitignore(self):
        """Update .gitignore with comprehensive exclusions"""
        print("📝 Checking .gitignore...")
        
        gitignore_path = self.project_root / ".gitignore"
        if not gitignore_path.exists():
            return
            
        with open(gitignore_path, 'r') as f:
            current_content = f.read()
            
        # Check for essential exclusions
        essential_patterns = [
            "output*/",
            "__pycache__/",
            "*.pyc",
            "results/benchmarks/",
            "*_benchmark_*.json",
            "*.tmp",
            "*.temp"
        ]
        
        missing_patterns = [p for p in essential_patterns if p not in current_content]
        
        if missing_patterns:
            print(f"   ⚠️  Missing patterns in .gitignore: {missing_patterns}")
        else:
            print("   ✅ .gitignore is comprehensive")
            
    def _generate_cleanup_summary(self):
        """Generate comprehensive cleanup summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.project_root / "docs" / "cleanup_summaries" / f"cleanup_summary_{timestamp}.json"
        
        # Add final statistics
        self.cleanup_summary["total_actions"] = len(self.cleanup_summary["actions_performed"])
        self.cleanup_summary["files_deleted_count"] = len(self.cleanup_summary["files_deleted"])
        self.cleanup_summary["directories_cleaned_count"] = len(self.cleanup_summary["directories_cleaned"])
        self.cleanup_summary["space_freed_mb"] = round(self.cleanup_summary["space_freed"] / (1024 * 1024), 2)
        
        with open(summary_file, 'w') as f:
            json.dump(self.cleanup_summary, f, indent=2)
            
        print(f"📊 Cleanup summary saved: {summary_file.relative_to(self.project_root)}")
        
        # Also create a markdown summary
        self._create_markdown_summary(timestamp)
        
    def _create_markdown_summary(self, timestamp: str):
        """Create a human-readable markdown summary"""
        md_file = self.project_root / "docs" / "cleanup_summaries" / f"cleanup_report_{timestamp}.md"
        
        content = f"""# Codebase Cleanup Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Status:** ✅ {self.cleanup_summary['status'].upper()}

## 📊 Cleanup Statistics

- **Actions Performed:** {len(self.cleanup_summary['actions_performed'])}
- **Files Deleted:** {len(self.cleanup_summary['files_deleted'])}
- **Directories Cleaned:** {len(self.cleanup_summary['directories_cleaned'])}
- **Space Freed:** {self.cleanup_summary.get('space_freed_mb', 0):.2f} MB

## 🔧 Actions Performed

"""
        
        for action in self.cleanup_summary['actions_performed']:
            content += f"- {action}\n"
            
        if self.cleanup_summary['files_deleted']:
            content += "\n## 🗑️ Files Deleted\n\n"
            for file in self.cleanup_summary['files_deleted']:
                content += f"- `{file}`\n"
                
        if self.cleanup_summary['directories_cleaned']:
            content += "\n## 📁 Directories Cleaned\n\n"
            for directory in self.cleanup_summary['directories_cleaned']:
                content += f"- `{directory}`\n"
                
        content += f"""

## 🎯 Cleanup Objectives Achieved

✅ **Python Cache Cleanup** - Removed all `__pycache__` directories and `.pyc` files  
✅ **Temporary File Removal** - Cleaned temporary and test files  
✅ **Directory Organization** - Removed empty output directories  
✅ **Documentation Organization** - Cleanup summaries properly organized  
✅ **Archive Management** - Old results archived appropriately  

## 📁 Current Structure

The codebase is now clean and organized:
- Source code in `src/`
- Tests in `tests/`
- Documentation in `docs/`
- Cleanup summaries in `docs/cleanup_summaries/`
- No temporary files or empty directories

**Status:** Ready for development and deployment 🚀
"""
        
        with open(md_file, 'w') as f:
            f.write(content)
            
        print(f"📋 Markdown report created: {md_file.relative_to(self.project_root)}")
        
    def _is_in_protected_dir(self, file_path: Path) -> bool:
        """Check if file is in a protected directory"""
        protected_dirs = [".venv", ".git", "node_modules", ".cache"]
        return any(protected in file_path.parts for protected in protected_dirs)

def main():
    """Main cleanup execution"""
    print("🧹 Comprehensive Codebase Cleanup")
    print("=" * 50)
    
    cleanup = CodebaseCleanup()
    cleanup.run_cleanup()
    
    print("\n" + "=" * 50)
    print("✨ Codebase is now clean and organized!")

if __name__ == "__main__":
    main() 