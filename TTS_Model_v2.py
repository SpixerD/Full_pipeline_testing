import asyncio
import re
import os
import tempfile
import wave
import numpy as np
from typing import AsyncGenerator, Generator, Optional, Tuple, Protocol
from functools import lru_cache
from dataclasses import dataclass
import torch


def print_memory(label=""):
    if torch.cuda.is_available():
        allocated = round(torch.cuda.memory_allocated() / 1024**2, 1)
        cached = round(torch.cuda.memory_reserved() / 1024**2, 1)
        print(f"[{label}] GPU memory - Allocated: {allocated}MB | Cached: {cached}MB")

# Define class protocols and options
class TTSOptions:
    pass

class TTSModel(Protocol):
    def tts(self, text: str, options: Optional[TTSOptions] = None) -> Tuple[int, np.ndarray]: ...
    
    async def stream_tts(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]: ...
    
    def stream_tts_sync(
        self, text: str, options: Optional[TTSOptions] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]: ...

@dataclass
class ArabicTTSOptions(TTSOptions):
    speaker: int = 1
    pace: float = 1.0
    denoise: float = 0.005
    volume: float = 0.9
    pitch_mul: float = 1.0
    pitch_add: float = 0.0
    vowelizer: None = None
    model_id: str = 'fastpitch'
    vocoder_id: str = 'vocos44'
    cuda: int = 0
    bits_per_sample: int = 32
    lang: str = "ar"

class ArabicTTSModel(TTSModel):
    def __init__(self):
        try:
            from tts_arabic import tts as arabic_tts
            self.arabic_tts = arabic_tts
            self._arabic_tts_available = True
            print("✅ Arabic TTS module loaded successfully")
        except ImportError:
            print("❌ ERROR: tts_arabic not found. Install with pip install git+https://github.com/nipponjo/tts_arabic.git sounddevice")
            self._arabic_tts_available = False
    
    def tts(self, text: str, options: Optional[ArabicTTSOptions] = None) -> Tuple[int, np.ndarray]:
        """Generate full audio for text"""
        if not self._arabic_tts_available:
            raise ImportError("tts_arabic is not installed")
            
        options = options or ArabicTTSOptions()
        
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            print(f"🔊 Generating audio for: '{text}'")
            # Call the arabic_tts function with the provided options
            print_memory("generating audio in tts_model class")
            self.arabic_tts(
                text=text,
                speaker=options.speaker,
                pace=options.pace,
                denoise=options.denoise,
                volume=options.volume,
                play=False,
                pitch_mul=options.pitch_mul,
                pitch_add=options.pitch_add,
                vowelizer=options.vowelizer,
                model_id=options.model_id,
                vocoder_id=options.vocoder_id,
                cuda=options.cuda,
                save_to=temp_path,
                bits_per_sample=options.bits_per_sample,
            )

            print_memory("after in tts_model class")
            
            # Read the audio file and convert to numpy array
            with wave.open(temp_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
                
                # Convert to numpy array based on bits_per_sample
                if options.bits_per_sample == 8:
                    audio = np.frombuffer(audio_bytes, dtype=np.int8).astype(np.float32) / 128.0
                elif options.bits_per_sample == 16:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:  # 32 bits
                    audio = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            
            print(f"🔊 Generated audio with sample rate {sample_rate} Hz")
            
            # Return sample_rate and audio data
            return sample_rate, audio
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file: {e}")

    async def stream_tts(
        self, text: str, options: Optional[ArabicTTSOptions] = None
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """Stream audio chunks asynchronously - improved implementation based on Kokoro method"""
        options = options or ArabicTTSOptions()
        
        # Split text into sentences using Arabic and English punctuation
        sentences = re.split(r'(?<=[.!?؟.])\s+', text.strip())
        
        for s_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            try:
                # Generate audio for this sentence
                sample_rate, audio = self.tts(sentence, options)
                
                # For better streaming, break into reasonable chunks (200ms)
                chunk_size = int(sample_rate * 0.2)  # 200ms chunks
                
                # Add a small silence between sentences (except for the first sentence)
                if s_idx > 0:
                    # Insert a short pause between sentences (similar to Kokoro model)
                    yield sample_rate, np.zeros(sample_rate // 7, dtype=np.float32)
                
                # Stream audio chunks
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    yield sample_rate, chunk
                    # Small delay for natural streaming feeling
                    await asyncio.sleep(0.05)
                    
            except Exception as e:
                print(f"Error generating audio for sentence '{sentence}': {e}")
                continue
    
    def stream_tts_sync(
        self, text: str, options: Optional[ArabicTTSOptions] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Synchronous wrapper for the async streaming function"""
        loop = asyncio.new_event_loop()
        
        # Use the loop to run the async generator
        iterator = self.stream_tts(text, options).__aiter__()
        
        try:
            while True:
                try:
                    yield loop.run_until_complete(iterator.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

@lru_cache
def get_tts_model() -> TTSModel:
    """Get the TTS model (with caching to avoid multiple initializations)"""
    model = ArabicTTSModel()
    # Optional: Warm up the model with a simple phrase
    try:
        model.tts("مرحبا")
    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")
    return model