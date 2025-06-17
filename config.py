"""
Configuration file for Medical Conversation System
"""

import torch

# Device configuration
DEVICE_MAP = "auto"  # Options: "auto", "cuda", "cpu", "cuda:0", etc.
MODEL_DTYPE = torch.float16  # Options: torch.float16, torch.float32

# Model names (can be changed if needed)
ATLAS_MODEL_NAME = "MBZUAI-Paris/Atlas-Chat-2B"
QWEN_MODEL_NAME = "Qwen/Qwen3-4B"
MARIAN_MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-ar"

# Audio processing settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_DURATION = 0.8
SPEECH_THRESHOLD = 0.6
STARTED_TALKING_THRESHOLD = 0.3

# Text processing settings
MAX_ARABIC_CHUNK_LENGTH = 315
MAX_NEW_TOKENS = 256
QWEN_MAX_NEW_TOKENS = 100

# Diacritization settings
ADD_HARAKAT_DEFAULT = True

# Memory management
AUTO_CLEANUP = True
CLEANUP_AFTER_PROCESSING = True
