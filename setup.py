#!/usr/bin/env python3
"""
Audio Transcription System Setup
Lightning MLX-powered transcription with speaker diarization and contextual enhancement
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="audio-transcription-system",
    version="1.0.0",
    author="Audio Transcription Team",
    author_email="contact@example.com",
    description="Complete AI-powered transcription system with Lightning MLX and contextual enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/audio-transcription-system",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    python_requires=">=3.11",
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
        ],
        "full": [
            "gensim>=4.3.0",
            "pyLDAvis>=3.4.0",
            "wordcloud>=1.9.0",
            "gradio>=4.15.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "transcribe-audio=src.core.lightning_whisper_mlx_transcriber:main",
            "build-pipeline=complete_model_pipeline:main",
        ],
    },
    
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    
    keywords=[
        "audio", "transcription", "speech-to-text", "speaker-diarization",
        "lightning-mlx", "apple-silicon", "meeting-intelligence", "nlp"
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/example/audio-transcription-system/issues",
        "Source": "https://github.com/example/audio-transcription-system",
        "Documentation": "https://github.com/example/audio-transcription-system/wiki",
    },
) 