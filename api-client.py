import requests
import time
import os
import torch
import numpy as np
import gc
import warnings
import re
import threading
from fastrtc import (ReplyOnPause, Stream, AlgoOptions)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  
from TTS_Model_v2 import get_tts_model
from STT_Model import get_stt_model

# Configuration: Replace with your performant machine's IP address and port
PERFORMANT_MACHINE_IP = "127.0.0.1"  # e.g., "192.168.1.100"
API_BASE_URL = f"http://{PERFORMANT_MACHINE_IP}:8000"


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

# Global variables for model references
stt_model = None
tts_model = None

# Synchronization mechanism to prevent concurrent execution
processing_lock = threading.Lock()

# Memory management settings
DEVICE_MAP = "auto"  # Let the model decide where to allocate tensors
MODEL_DTYPE = torch.float16  # Use float16 for all models to reduce memory usage

def load_tts_model():
    global tts_model
    if tts_model is None:
        print("⏳ Loading TTS model...")
        tts_model = get_tts_model()
        print("✅ TTS model loaded successfully")
    return tts_model

def unload_tts_model():
    global tts_model
    if tts_model is not None:
        print("⏳ Unloading TTS model...")
        del tts_model
        tts_model = None
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ TTS model unloaded successfully")

def load_stt_model():
    global stt_model
    if stt_model is None:
        print("⏳ Loading STT model...")
        stt_model = get_stt_model()
        print("✅ STT model loaded successfully")
    return stt_model

def unload_stt_model():
    global stt_model
    if stt_model is not None:
        print("⏳ Unloading STT model...")
        del stt_model
        stt_model = None
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ STT model unloaded successfully")

def call_api(endpoint: str, input_text: str) -> str | None:
    """Helper function to call an API endpoint."""
    url = f"{API_BASE_URL}{endpoint}"
    payload = {"text": input_text}
    try:
        print(f"➡️  Client: Calling {url} with text: '{input_text}'")
        response = requests.post(url, json=payload, timeout=180) # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        response_data = response.json()
        processed_text = response_data.get("processed_text")
        print(f"⬅️  Client: Received from {url}: '{processed_text}'")
        return processed_text
    except requests.exceptions.RequestException as e:
        print(f"❗️ Client: API call to {url} failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"❗️ Client: Server error detail: {e.response.json()}")
            except requests.exceptions.JSONDecodeError:
                print(f"❗️ Client: Server error content: {e.response.text}")
        return None
    except Exception as e:
        print(f"❗️ Client: An unexpected error occurred: {e}")
        return None


def run_full_pipeline_on_client(darija_input: str):
    print(f"\n🚀 Client: Starting full processing pipeline for: '{darija_input}'")
    start_time = time.time()

    # Step 1: Translate Darija to English via API
    english_text = call_api("/translate/darija-to-english", darija_input)
    if not english_text:
        print("❗️ Client: Failed to translate Darija to English. Aborting.")
        return

    # Step 2: Generate response in English via API
    english_response = call_api("/generate/doctor-response", english_text)
    if not english_response:
        print("❗️ Client: Failed to generate doctor's response. Aborting.")
        return

    # Step 3: Translate English response back to Arabic via API
    arabic_response = call_api("/translate/english-to-arabic", english_response)
    if not arabic_response:
        print("❗️ Client: Failed to translate response to Arabic. Aborting.")
        return

    end_time = time.time()
    print(f"✅ Client: Processing complete in {end_time - start_time:.2f} seconds.")
    return arabic_response





def echo(audio):
    # Use thread locking to prevent concurrent execution
    if processing_lock.locked():
        print("⚠️ Processing is already in progress, skipping this input")
        return
    
    # Acquire the lock to ensure only one processing flow at a time
    with processing_lock:
        # Clear memory before starting
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Step 1: Speech to text

            unload_tts_model()
            stt = load_stt_model()
            
            print("🎤 Processing speech input...")
            prompt = stt.stt(audio)
            
            # Free STT model memory immediately
            unload_stt_model()
            
            response_text_arabic = run_full_pipeline_on_client(prompt)

            tts = load_tts_model()
            
            print("🔊 Converting response to speech...")
            audio_chunks = list(tts.stream_tts_sync(response_text_arabic))
            

            
            # Now yield the audio chunks
            for chunk in audio_chunks:
                yield chunk
                
            print("✅ Processing complete")

            # Release TTS model resources before yielding audio
            unload_tts_model()

        except Exception as e:
            print(f"❌ Error in processing: {str(e)}")
            # Clean up any models that might be loaded to avoid memory issues
            unload_stt_model()
            unload_tts_model()
            torch.cuda.empty_cache()
            gc.collect()

# Stream configuration with adjusted parameters
options = AlgoOptions(
    audio_chunk_duration=0.8,  # Slightly longer chunks for more stable processing
    started_talking_threshold=0.3,
    speech_threshold=0.6,
)

# Initialize the stream
stream = Stream(
    ReplyOnPause(
        echo,  # Use our synchronous function directly
        input_sample_rate=16000,
        algo_options=options,
    ),
    modality="audio",
    mode="send-receive"
)

#   h the UI
if __name__ == "__main__":
    print("🚀 Starting conversational agent...")
    torch.cuda.empty_cache()
    gc.collect()
    print("💾 GPU memory cleared and ready")
    stream.ui.launch()