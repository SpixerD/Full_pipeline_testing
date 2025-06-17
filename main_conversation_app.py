import os
import torch
import gc
import warnings
import threading
from fastrtc import (ReplyOnPause, Stream, AlgoOptions)

# Import our custom modules
from speech_processing_module import SpeechProcessingModule
from medical_conversation_pipeline import MedicalConversationPipeline

# Filter specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*`do_sample` is set to.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Sliding Window Attention is enabled.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*You are calling .generate.*")
warnings.filterwarnings("ignore", message=".*Invalid model-index.*")

# Disable HuggingFace warnings and logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"  # Suppress tqdm progress bars

# Configure logging
import logging
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("fastrtc.webrtc_connection_mixin").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub.repocard_data").setLevel(logging.ERROR)

class MedicalConversationApp:
    """
    Main application class that handles the complete medical conversation workflow
    """
    
    def __init__(self, device_map: str = "auto", model_dtype=torch.float16):
        # Initialize processing modules
        self.speech_processor = SpeechProcessingModule()
        self.conversation_pipeline = MedicalConversationPipeline(device_map, model_dtype)
        
        # Synchronization mechanism to prevent concurrent execution
        self.processing_lock = threading.Lock()
        
        print("🚀 Medical Conversation App initialized")
    
    def process_audio_input(self, audio):
        """
        Main processing function that handles the complete audio-to-audio pipeline
        
        Args:
            audio: Input audio from microphone
            
        Yields:
            Audio chunks for TTS output
        """
        # Use thread locking to prevent concurrent execution
        if self.processing_lock.locked():
            print("⚠️ Processing is already in progress, skipping this input")
            return
        
        # Acquire the lock to ensure only one processing flow at a time
        with self.processing_lock:
            # Clear memory before starting
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # Step 1: Speech to text
                print("🎤 Step 1: Converting speech to text...")
                self.speech_processor.unload_tts_model()  # Ensure TTS is unloaded
                
                # Get the transcribed text
                darija_text = self.speech_processor.speech_to_text(audio)
                
                # Free STT model memory immediately after use
                self.speech_processor.unload_stt_model()
                
                # Step 2: Process through the medical conversation pipeline
                print("💭 Step 2: Processing through medical pipeline...")
                arabic_response = self.conversation_pipeline.process_darija_input(
                    darija_text, 
                    add_diacritization=False  # Set to True if you want diacritization
                )
                
                # Step 3: Convert response to speech
                print("🔊 Step 3: Converting response to speech...")
                
                # Generate TTS audio chunks
                audio_chunks = list(self.speech_processor.text_to_speech(arabic_response))
                
                # Yield the audio chunks
                for chunk in audio_chunks:
                    yield chunk
                    
                print("✅ Processing complete")

                # Release TTS model resources after yielding audio
                self.speech_processor.unload_tts_model()

            except Exception as e:
                print(f"❌ Error in processing: {str(e)}")
                # Clean up any models that might be loaded to avoid memory issues
                self.speech_processor.unload_all_models()
                self.conversation_pipeline.cleanup_memory()
    
    def cleanup_all_resources(self):
        """Clean up all resources and unload all models"""
        print("🧹 Cleaning up all resources...")
        self.speech_processor.unload_all_models()
        self.conversation_pipeline.unload_all_models()
        print("✅ All resources cleaned up")
    
    def get_system_status(self) -> dict:
        """Get the status of all system components"""
        return {
            "speech_models": {
                "stt_loaded": self.speech_processor.stt_model is not None,
                "tts_loaded": self.speech_processor.tts_model is not None,
            },
            "conversation_models": self.conversation_pipeline.get_model_status(),
            "processing_locked": self.processing_lock.locked()
        }

def create_stream_with_app():
    """
    Create and configure the FastRTC stream with our medical conversation app
    """
    # Initialize the medical conversation app
    app = MedicalConversationApp()
    
    # Stream configuration with adjusted parameters
    options = AlgoOptions(
        audio_chunk_duration=0.8,  # Slightly longer chunks for more stable processing
        started_talking_threshold=0.3,
        speech_threshold=0.6,
    )

    # Initialize the stream with our app's processing function
    stream = Stream(
        ReplyOnPause(
            app.process_audio_input,  # Use our app's processing method
            input_sample_rate=16000,
            algo_options=options,
        ),
        modality="audio",
        mode="send-receive"
    )
    
    return stream, app

def main():
    """
    Main entry point for the medical conversation application
    """
    print("🚀 Starting Medical Conversational Agent...")
    
    # Clear GPU memory at startup
    torch.cuda.empty_cache()
    gc.collect()
    print("💾 GPU memory cleared and ready")
    
    try:
        # Create the stream and app
        stream, app = create_stream_with_app()
        
        # Launch the UI
        print("🌐 Launching user interface...")
        stream.ui.launch(share=True)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
    finally:
        # Cleanup resources on exit
        try:
            app.cleanup_all_resources()
        except:
            pass
        print("✅ Application shutdown complete")

if __name__ == "__main__":
    main()