import requests
import time
import os
import torch
import numpy as np
import gc
import warnings
import re
import threading
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  
from TTS_Model_v2 import get_tts_model
# Removed STT_Model import and fastrtc imports since we're not using audio input

# Configuration: Replace with your performant machine's IP address and port
PERFORMANT_MACHINE_IP = "127.0.0.1"  # e.g., "192.168.1.100"
API_BASE_URL = f"http://{PERFORMANT_MACHINE_IP}:8000"

# Default Darija sentence - Can be changed via the UI
DEFAULT_DARIJA_TEXT = "كيفاش نقدر نحسن من صحتي؟"  # Example: "How can I improve my health?"

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
logging.getLogger("huggingface_hub.repocard_data").setLevel(logging.ERROR)

# Global variables for model references
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
        return None, None, None, None

    # Step 2: Generate response in English via API
    english_response = call_api("/generate/doctor-response", english_text)
    if not english_response:
        print("❗️ Client: Failed to generate doctor's response. Aborting.")
        return english_text, None, None, None

    # Step 3: Translate English response back to Arabic via API
    arabic_response = call_api("/translate/english-to-arabic", english_response)
    if not arabic_response:
        print("❗️ Client: Failed to translate response to Arabic. Aborting.")
        return english_text, english_response, None, None

    end_time = time.time()
    print(f"✅ Client: Processing complete in {end_time - start_time:.2f} seconds.")
    return english_text, english_response, arabic_response, end_time - start_time


def process_darija_text(darija_input: str):
    """Process Darija text and return results for Gradio interface"""
    
    if not darija_input.strip():
        return "❗️ Please enter some Darija text", "", "", None, 0
    
    # Use thread locking to prevent concurrent execution
    if processing_lock.locked():
        return "⚠️ Processing is already in progress, please wait...", "", "", None, 0
    
    # Acquire the lock to ensure only one processing flow at a time
    with processing_lock:
        # Clear memory before starting
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            print(f"📝 Processing Darija text: '{darija_input}'")
            
            # Process the Darija text through the pipeline
            english_text, english_response, arabic_response, processing_time = run_full_pipeline_on_client(darija_input)
            
            if arabic_response:
                # Load TTS model and convert response to speech
                tts = load_tts_model()
                
                print("🔊 Converting response to speech...")
                
                # Generate audio file
                try:
                    # Use TTS to generate audio and save to a temporary file
                    audio_chunks = list(tts.stream_tts_sync(arabic_response))
                    
                    # Combine audio chunks into a single audio array
                    if audio_chunks:
                        print(f"📊 Processing {len(audio_chunks)} audio chunks")
                        
                        # Process and normalize audio chunks
                        normalized_chunks = []
                        target_sample_rate = 44100  # Use the actual sample rate from your TTS
                        
                        for i, chunk in enumerate(audio_chunks):
                            try:
                                print(f"🔍 Chunk {i}: type={type(chunk)}")
                                
                                # Handle different chunk formats
                                if isinstance(chunk, (list, tuple)):
                                    # If chunk is a list/tuple, it might contain multiple audio segments
                                    print(f"   Chunk {i} is a {type(chunk).__name__} with {len(chunk)} elements")
                                    
                                    # Process each element in the chunk
                                    for j, element in enumerate(chunk):
                                        try:
                                            if isinstance(element, (list, tuple, np.ndarray)):
                                                # Convert to numpy array
                                                audio_data = np.array(element, dtype=np.float32)
                                                
                                                # Handle different shapes
                                                if len(audio_data.shape) > 1:
                                                    # Flatten multi-dimensional arrays
                                                    audio_data = audio_data.flatten()
                                                
                                                print(f"     Element {j}: shape={audio_data.shape}, dtype={audio_data.dtype}")
                                                
                                                # Normalize if needed
                                                if audio_data.dtype != np.float32:
                                                    audio_data = audio_data.astype(np.float32)
                                                
                                                # Ensure values are in [-1, 1] range
                                                if np.max(np.abs(audio_data)) > 1.0:
                                                    # Normalize if values are outside [-1, 1]
                                                    audio_data = audio_data / np.max(np.abs(audio_data))
                                                
                                                normalized_chunks.append(audio_data)
                                                
                                        except Exception as element_error:
                                            print(f"     ❗️ Error processing element {j}: {element_error}")
                                            continue
                                            
                                elif isinstance(chunk, np.ndarray):
                                    # Direct numpy array
                                    print(f"   Chunk {i}: numpy array shape={chunk.shape}, dtype={chunk.dtype}")
                                    
                                    # Handle multi-dimensional arrays
                                    if len(chunk.shape) > 1:
                                        chunk = chunk.flatten()
                                    
                                    # Normalize data type
                                    if chunk.dtype != np.float32:
                                        chunk = chunk.astype(np.float32)
                                    
                                    # Normalize values
                                    if np.max(np.abs(chunk)) > 1.0:
                                        chunk = chunk / np.max(np.abs(chunk))
                                    
                                    normalized_chunks.append(chunk)
                                    
                                else:
                                    # Try to convert whatever it is to a numpy array
                                    print(f"   Chunk {i}: attempting conversion from {type(chunk)}")
                                    audio_data = np.array(chunk, dtype=np.float32)
                                    
                                    if len(audio_data.shape) > 1:
                                        audio_data = audio_data.flatten()
                                    
                                    if np.max(np.abs(audio_data)) > 1.0:
                                        audio_data = audio_data / np.max(np.abs(audio_data))
                                    
                                    normalized_chunks.append(audio_data)
                                
                            except Exception as chunk_error:
                                print(f"❗️ Error processing chunk {i}: {chunk_error}")
                                print(f"   Chunk type: {type(chunk)}")
                                if hasattr(chunk, 'shape'):
                                    print(f"   Chunk shape: {chunk.shape}")
                                continue
                        
                        if normalized_chunks:
                            # Concatenate all normalized chunks
                            combined_audio = np.concatenate(normalized_chunks)
                            print(f"📊 Combined audio shape: {combined_audio.shape}, dtype: {combined_audio.dtype}")
                            
                            # Save to temporary file
                            import tempfile
                            import scipy.io.wavfile as wavfile
                            
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                            
                            # Convert to int16 for WAV file (standard format)
                            audio_int16 = (combined_audio * 32767).astype(np.int16)
                            
                            wavfile.write(temp_file.name, target_sample_rate, audio_int16)
                            audio_file_path = temp_file.name
                            temp_file.close()
                            
                            print(f"✅ Audio saved to: {audio_file_path}")
                        else:
                            print("❗️ No valid audio chunks to process")
                            audio_file_path = None
                    else:
                        print("❗️ No audio chunks generated")
                        audio_file_path = None
                        
                except Exception as audio_error:
                    print(f"❗️ Audio generation failed: {audio_error}")
                    audio_file_path = None
                
                print("✅ Processing complete")
                
                # Release TTS model resources
                unload_tts_model()
                
                # Format results
                status = f"✅ Processing completed successfully in {processing_time:.2f} seconds"
                english_translation = english_text or "Failed to translate"
                doctor_response = english_response or "Failed to generate response"
                arabic_final = arabic_response
                
                return status, english_translation, doctor_response, arabic_final, audio_file_path
            else:
                return "❗️ Failed to process the text through the pipeline", "", "", "", None

        except Exception as e:
            print(f"❌ Error in processing: {str(e)}")
            # Clean up any models that might be loaded to avoid memory issues
            unload_tts_model()
            torch.cuda.empty_cache()
            gc.collect()
            return f"❌ Error: {str(e)}", "", "", "", None


def create_gradio_interface():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(title="Darija Medical Assistant") as interface:
        gr.Markdown("# 🏥 Darija Medical Assistant")
        gr.Markdown("Enter your medical question in Darija and get a response from an AI doctor")
        
        with gr.Row():
            with gr.Column():
                # Input text box
                input_text = gr.Textbox(
                    label="Enter your question in Darija",
                    placeholder=DEFAULT_DARIJA_TEXT,
                    value=DEFAULT_DARIJA_TEXT,
                    lines=3
                )
                
                # Process button
                process_btn = gr.Button("🚀 Process Question", variant="primary")
                
                # Status output
                status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column():
                # Output fields
                english_output = gr.Textbox(label="English Translation", interactive=False, lines=2)
                doctor_output = gr.Textbox(label="Doctor's Response (English)", interactive=False, lines=4)
                arabic_output = gr.Textbox(label="Final Response (Arabic)", interactive=False, lines=4)
                audio_output = gr.Audio(label="Audio Response", interactive=False)
        
        # Clear button
        clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        # Event handlers
        process_btn.click(
            fn=process_darija_text,
            inputs=[input_text],
            outputs=[status_output, english_output, doctor_output, arabic_output, audio_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", "", "", None),
            outputs=[input_text, status_output, english_output, doctor_output, arabic_output, audio_output]
        )
        
        # Example questions
        gr.Markdown("### Example Questions:")
        example_questions = [
            "كيفاش نقدر نحسن من صحتي؟",
            "واش عندي مشكل في القلب؟",
            "شنو الأكل اللي خاصني ناكل؟"
        ]
        
        for i, example in enumerate(example_questions):
            gr.Button(f"Example {i+1}: {example}").click(
                fn=lambda x=example: x,
                outputs=[input_text]
            )
    
    return interface


# Function to run processing without UI (for testing)
def run_standalone():
    """Run the processing pipeline standalone without the UI"""
    print("🚀 Running standalone processing...")
    torch.cuda.empty_cache()
    gc.collect()
    print("💾 GPU memory cleared and ready")
    
    test_text = DEFAULT_DARIJA_TEXT
    result = process_darija_text(test_text)
    print(f"🎯 Results: {result}")


# Launch the UI
if __name__ == "__main__":
    print("🚀 Starting Darija Medical Assistant...")
    torch.cuda.empty_cache()
    gc.collect()
    print("💾 GPU memory cleared and ready")
    
    # Uncomment the next line to run standalone without UI
    # run_standalone()
    
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",  # Makes it accessible from other devices on your network
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        debug=True
    )