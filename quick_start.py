#!/usr/bin/env python3
"""
Quick Start Script for Ultra-Fast Transcription & Diarization
Run this script to test the system with minimal configuration
"""

import sys
import time
from pathlib import Path

# Import configuration
try:
    from config_ultrafast import *
    print("✅ Configuration loaded successfully")
except ImportError:
    print("❌ Could not import config_ultrafast.py")
    print("Make sure config_ultrafast.py is in the same directory")
    sys.exit(1)

def check_requirements():
    """Check if all required packages are installed"""
    print("\n🔍 Checking requirements...")
    
    required_packages = [
        'torch', 'torchaudio', 'transformers', 'pyannote.audio', 
        'librosa', 'soundfile', 'numpy', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('.', '/'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_ultrafast.txt")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def check_audio_file():
    """Check if the audio file exists"""
    print(f"\n📁 Checking audio file: {AUDIO_FILE}")
    
    if not Path(AUDIO_FILE).exists():
        print(f"❌ Audio file not found: {AUDIO_FILE}")
        print("\n💡 Solutions:")
        print("1. Place your audio file in this directory")
        print("2. Update AUDIO_FILE in config_ultrafast.py")
        print("3. Use absolute path to your audio file")
        return False
    
    print("✅ Audio file found!")
    return True

def check_huggingface_auth():
    """Check Hugging Face authentication"""
    print("\n🤗 Checking Hugging Face authentication...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to access a model to verify authentication
        try:
            api.model_info("pyannote/speaker-diarization-3.1")
            print("✅ Hugging Face authentication working!")
            return True
        except Exception as e:
            if "not found" in str(e).lower():
                print("❌ Access denied to pyannote models")
                print("\n💡 Required steps:")
                print("1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("2. Accept the license agreement")
                print("3. Run: huggingface-cli login")
                return False
            else:
                print(f"⚠️  Authentication check inconclusive: {e}")
                return True
                
    except ImportError:
        print("⚠️  Could not check HF authentication (missing huggingface_hub)")
        return True

def run_ultra_fast_script():
    """Run the ultra-fast transcription script"""
    print("\n🚀 Starting Ultra-Fast Transcription...")
    print("=" * 60)
    
    # Show current configuration
    print_config_summary()
    
    try:
        # Import and run the ultra-fast script
        from ultra_fast_diarization import main
        import asyncio
        
        start_time = time.time()
        asyncio.run(main())
        total_time = time.time() - start_time
        
        print(f"\n🎉 Quick start completed in {total_time:.2f}s!")
        print(f"📁 Check output in: {OUTPUT_DIR}")
        
    except ImportError:
        print("❌ Could not import ultra_fast_diarization.py")
        print("Make sure ultra_fast_diarization.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main quick start function"""
    print("🚀 ULTRA-FAST TRANSCRIPTION QUICK START")
    print("=" * 50)
    
    # Step 1: Check requirements
    if not check_requirements():
        return
    
    # Step 2: Check audio file
    if not check_audio_file():
        return
    
    # Step 3: Check HF authentication  
    if not check_huggingface_auth():
        print("\n⚠️  Continuing anyway (might fail later)...")
    
    # Step 4: Run transcription
    print(f"\n🎯 Target: Sub-0.2x processing ratio")
    print("Processing will start in 3 seconds...")
    time.sleep(3)
    
    success = run_ultra_fast_script()
    
    if success:
        print("\n✅ Quick start completed successfully!")
        print("\n📋 Next steps:")
        print("1. Check the output files")
        print("2. Adjust config_ultrafast.py for your needs")
        print("3. Try extreme_speed_transcription.py for maximum speed")
    else:
        print("\n❌ Quick start failed. Check the error messages above.")

if __name__ == "__main__":
    main() 