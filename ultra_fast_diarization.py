import asyncio
import sys

if __name__ == "__main__":
    try:
        print("🚀 Starting Ultra-Fast Diarization & Transcription System...")
        asyncio.run(main())
        print("🏁 Processing complete!")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 