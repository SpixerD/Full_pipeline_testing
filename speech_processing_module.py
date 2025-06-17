import torch
import gc
from TTS_Model_v2 import get_tts_model
from STT_Model import get_stt_model
from typing import Optional, Generator

class SpeechProcessingModule:
    """
    Handles Speech-to-Text and Text-to-Speech processing
    """
    
    def __init__(self):
        self.stt_model = None
        self.tts_model = None
    
    def load_stt_model(self):
        """Load Speech-to-Text model"""
        if self.stt_model is None:
            print("⏳ Loading STT model...")
            self.stt_model = get_stt_model()
            print("✅ STT model loaded successfully")
        return self.stt_model
    
    def load_tts_model(self):
        """Load Text-to-Speech model"""
        if self.tts_model is None:
            print("⏳ Loading TTS model...")
            self.tts_model = get_tts_model()
            print("✅ TTS model loaded successfully")
        return self.tts_model
    
    def speech_to_text(self, audio) -> str:
        """
        Convert speech audio to text
        
        Args:
            audio: Audio input for speech recognition
            
        Returns:
            Transcribed text
        """
        stt = self.load_stt_model()
        print("🎤 Processing speech input...")
        
        try:
            text = stt.stt(audio)
            print(f"🎤 Speech recognition complete: '{text}'")
            return text
        except Exception as e:
            print(f"❌ Error in speech recognition: {e}")
            raise e
    
    def text_to_speech(self, text: str) -> Generator:
        """
        Convert text to speech audio chunks
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Audio chunks
        """
        tts = self.load_tts_model()
        print(f"🔊 Converting text to speech: '{text}'")
        
        try:
            print("🔊 Converting response to speech...")
            audio_chunks = list(tts.stream_tts_sync(text))
            
            # Yield the audio chunks
            for chunk in audio_chunks:
                yield chunk
                
            print("✅ Text-to-speech conversion complete")
            
        except Exception as e:
            print(f"❌ Error in text-to-speech conversion: {e}")
            raise e
    
    def unload_stt_model(self):
        """Unload Speech-to-Text model to free memory"""
        if self.stt_model is not None:
            print("⏳ Unloading STT model...")
            del self.stt_model
            self.stt_model = None
            torch.cuda.empty_cache()
            gc.collect()
            print("✅ STT model unloaded successfully")
    
    def unload_tts_model(self):
        """Unload Text-to-Speech model to free memory"""
        if self.tts_model is not None:
            print("⏳ Unloading TTS model...")
            del self.tts_model
            self.tts_model = None
            torch.cuda.empty_cache()
            gc.collect()
            print("✅ TTS model unloaded successfully")
    
    def unload_all_models(self):
        """Unload all speech processing models"""
        self.unload_stt_model()
        self.unload_tts_model()
