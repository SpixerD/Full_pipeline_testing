import requests
import time
import os
import torch
import numpy as np
import gc
import warnings
import threading
import tempfile
import scipy.io.wavfile as wavfile
import gradio as gr
from TTS_Model_v2 import get_tts_model

# Configuration: Replace with your API server's IP address and port
API_BASE_URL = "http://127.0.0.1:8000"  # Change this to your server's address

# Default Darija sentence
DEFAULT_DARIJA_TEXT = "كيفاش نقدر نحسن من صحتي؟"  # "How can I improve my health?"

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

# Global variables
tts_model = None
processing_lock = threading.Lock()

def load_tts_model():
    """Load TTS model for audio generation"""
    global tts_model
    if tts_model is None:
        print("⏳ Loading TTS model...")
        try:
            tts_model = get_tts_model()
            print("✅ TTS model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading TTS model: {e}")
            tts_model = None
    return tts_model

def unload_tts_model():
    """Unload TTS model to free memory"""
    global tts_model
    if tts_model is not None:
        print("⏳ Unloading TTS model...")
        del tts_model
        tts_model = None
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ TTS model unloaded successfully")

def call_api(endpoint: str, input_text: str, timeout: int = 180) -> str | None:
    """Helper function to call API endpoints"""
    url = f"{API_BASE_URL}{endpoint}"
    payload = {"text": input_text}
    
    try:
        print(f"➡️  Calling {endpoint} with: '{input_text[:50]}...'")
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        response_data = response.json()
        processed_text = response_data.get("processed_text")
        print(f"⬅️  Received from {endpoint}: '{processed_text[:50] if processed_text else None}...'")
        return processed_text
        
    except requests.exceptions.RequestException as e:
        print(f"❗️ API call to {endpoint} failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"❗️ Server error: {e.response.json()}")
            except:
                print(f"❗️ Server error: {e.response.text}")
        return None
    except Exception as e:
        print(f"❗️ Unexpected error: {e}")
        return None

def check_api_health():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def generate_audio_from_text(text: str) -> str | None:
    """Generate audio file from Arabic text using TTS"""
    if not text or not text.strip():
        return None
        
    try:
        tts = load_tts_model()
        if not tts:
            return None
            
        print("🔊 Converting text to speech...")
        
        # Generate audio chunks
        audio_chunks = list(tts.stream_tts_sync(text))
        
        if not audio_chunks:
            print("❗️ No audio chunks generated")
            return None
            
        # Process and normalize audio chunks
        normalized_chunks = []
        target_sample_rate = 44100
        
        for i, chunk in enumerate(audio_chunks):
            try:
                if isinstance(chunk, (list, tuple)):
                    for element in chunk:
                        if isinstance(element, (list, tuple, np.ndarray)):
                            audio_data = np.array(element, dtype=np.float32)
                            if len(audio_data.shape) > 1:
                                audio_data = audio_data.flatten()
                            if np.max(np.abs(audio_data)) > 1.0:
                                audio_data = audio_data / np.max(np.abs(audio_data))
                            normalized_chunks.append(audio_data)
                            
                elif isinstance(chunk, np.ndarray):
                    if len(chunk.shape) > 1:
                        chunk = chunk.flatten()
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32)
                    if np.max(np.abs(chunk)) > 1.0:
                        chunk = chunk / np.max(np.abs(chunk))
                    normalized_chunks.append(chunk)
                    
                else:
                    audio_data = np.array(chunk, dtype=np.float32)
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.flatten()
                    if np.max(np.abs(audio_data)) > 1.0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    normalized_chunks.append(audio_data)
                    
            except Exception as e:
                print(f"❗️ Error processing chunk {i}: {e}")
                continue
        
        if normalized_chunks:
            # Combine all audio chunks
            combined_audio = np.concatenate(normalized_chunks)
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio_int16 = (combined_audio * 32767).astype(np.int16)
            wavfile.write(temp_file.name, target_sample_rate, audio_int16)
            
            temp_file.close()
            print(f"✅ Audio generated: {temp_file.name}")
            return temp_file.name
        else:
            print("❗️ No valid audio chunks processed")
            return None
            
    except Exception as e:
        print(f"❗️ Audio generation failed: {e}")
        return None
    finally:
        # Clean up TTS model
        unload_tts_model()

def process_medical_question(darija_input: str, progress=gr.Progress()):
    """Main processing function for the medical assistant"""
    
    if not darija_input or not darija_input.strip():
        return "❗️ Please enter some text in Darija", "", "", "", None
    
    # Check if processing is already in progress
    if processing_lock.locked():
        return "⚠️ Processing already in progress, please wait...", "", "", "", None
    
    with processing_lock:
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            start_time = time.time()
            
            # Check API health first
            progress(0.1, desc="Checking API connection...")
            if not check_api_health():
                return "❗️ API server is not responding. Please check if the server is running.", "", "", "", None
            
            # Step 1: Translate Darija to English
            progress(0.2, desc="Translating Darija to English...")
            english_text = call_api("/translate/darija-to-english", darija_input)
            if not english_text:
                return "❗️ Failed to translate Darija to English", "", "", "", None
            
            # Step 2: Generate doctor's response
            progress(0.4, desc="Generating medical response...")
            english_response = call_api("/generate/doctor-response", english_text)
            if not english_response:
                return english_text, "❗️ Failed to generate doctor's response", "", "", None
            
            # Step 3: Translate response to Arabic
            progress(0.6, desc="Translating response to Arabic...")
            arabic_response = call_api("/translate/english-to-arabic", english_response)
            if not arabic_response:
                return english_text, english_response, "❗️ Failed to translate response to Arabic", "", None
            
            # Step 4: Generate audio
            progress(0.8, desc="Generating audio...")
            audio_file = generate_audio_from_text(arabic_response)
            
            progress(1.0, desc="Complete!")
            
            processing_time = time.time() - start_time
            status = f"✅ Processing completed successfully in {processing_time:.2f} seconds"
            
            return status, english_text, english_response, arabic_response, audio_file
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            unload_tts_model()
            torch.cuda.empty_cache()
            gc.collect()
            return f"❌ Error: {str(e)}", "", "", "", None

def translate_darija_only(darija_text: str):
    """Translate Darija to English only"""
    if not darija_text or not darija_text.strip():
        return "Please enter some Darija text"
    
    english_text = call_api("/translate/darija-to-english", darija_text)
    return english_text if english_text else "Translation failed"

def translate_english_to_arabic(english_text: str):
    """Translate English to Arabic with diacritization"""
    if not english_text or not english_text.strip():
        return "Please enter some English text"
    
    arabic_text = call_api("/translate/english-to-arabic", english_text)
    return arabic_text if arabic_text else "Translation failed"

def generate_doctor_response(english_concern: str):
    """Generate doctor's response to medical concern"""
    if not english_concern or not english_concern.strip():
        return "Please enter a medical concern"
    
    response = call_api("/generate/doctor-response", english_concern)
    return response if response else "Response generation failed"

def diacritize_arabic_text(arabic_text: str):
    """Add diacritics to Arabic text"""
    if not arabic_text or not arabic_text.strip():
        return "Please enter some Arabic text"
    
    # Use the diacritization endpoint
    url = f"{API_BASE_URL}/diacritize/arabic"
    payload = {"text": arabic_text, "add_harakat": True}
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("processed_text", "Diacritization failed")
    except Exception as e:
        print(f"Diacritization error: {e}")
        return "Diacritization failed"

def create_gradio_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(
        title="Darija Medical Assistant",
        theme=gr.themes.Soft(),
        css="""
        .main-header {text-align: center; margin-bottom: 20px;}
        .status-box {background-color: #f0f8ff; padding: 10px; border-radius: 5px;}
        .example-btn {margin: 5px;}
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🏥 Darija Medical Assistant
            ### Ask your medical questions in Darija and get responses from an AI doctor
            """,
            elem_classes=["main-header"]
        )
        
        # API Status indicator
        with gr.Row():
            api_status = gr.HTML(
                f"""
                <div style="text-align: center; padding: 10px;">
                    <span style="color: {'green' if check_api_health() else 'red'};">
                        API Status: {'🟢 Online' if check_api_health() else '🔴 Offline'}
                    </span>
                </div>
                """
            )
        
        # Main processing tab
        with gr.Tab("🚀 Full Medical Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    darija_input = gr.Textbox(
                        label="Enter your medical question in Darija",
                        placeholder=DEFAULT_DARIJA_TEXT,
                        value=DEFAULT_DARIJA_TEXT,
                        lines=3,
                        max_lines=5
                    )
                    
                    process_btn = gr.Button(
                        "🚀 Process Medical Question",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Example questions
                    gr.Markdown("### 📝 Example Questions:")
                    examples = [
                        "كيفاش نقدر نحسن من صحتي؟",
                        "واش عندي مشكل في القلب؟",
                        "شنو الأكل اللي خاصني ناكل؟",
                        "كيفاش نقدر نخسر الوزن؟",
                        "شنو هي أعراض السكري؟"
                    ]
                    
                    for i, example in enumerate(examples):
                        gr.Button(
                            f"Example {i+1}: {example}",
                            elem_classes=["example-btn"]
                        ).click(
                            fn=lambda x=example: x,
                            outputs=[darija_input]
                        )
                
                with gr.Column(scale=1):
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    
                    english_translation = gr.Textbox(
                        label="English Translation",
                        interactive=False,
                        lines=2
                    )
                    
                    doctor_response = gr.Textbox(
                        label="Doctor's Response (English)",
                        interactive=False,
                        lines=4
                    )
                    
                    arabic_response = gr.Textbox(
                        label="Final Response (Arabic with Diacritics)",
                        interactive=False,
                        lines=4
                    )
                    
                    audio_output = gr.Audio(
                        label="🔊 Audio Response",
                        interactive=False
                    )
            
            # Clear button
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        # Individual tools tabs
        with gr.Tab("🔄 Translation Tools"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Darija → English")
                    darija_input_simple = gr.Textbox(
                        label="Darija Text",
                        placeholder="Enter Darija text here..."
                    )
                    translate_btn1 = gr.Button("Translate to English")
                    english_output_simple = gr.Textbox(
                        label="English Translation",
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### English → Arabic")
                    english_input_simple = gr.Textbox(
                        label="English Text",
                        placeholder="Enter English text here..."
                    )
                    translate_btn2 = gr.Button("Translate to Arabic")
                    arabic_output_simple = gr.Textbox(
                        label="Arabic Translation",
                        interactive=False
                    )
        
        with gr.Tab("👨‍⚕️ Doctor Response"):
            gr.Markdown("### Generate Medical Response")
            medical_concern = gr.Textbox(
                label="Medical Concern (English)",
                placeholder="Describe your medical concern in English...",
                lines=3
            )
            generate_btn = gr.Button("Generate Doctor's Response")
            medical_response = gr.Textbox(
                label="Doctor's Response",
                interactive=False,
                lines=4
            )
        
        with gr.Tab("✨ Arabic Diacritization"):
            gr.Markdown("### Add Diacritics to Arabic Text")
            arabic_input_diac = gr.Textbox(
                label="Arabic Text (without diacritics)",
                placeholder="Enter Arabic text here...",
                lines=3
            )
            diacritize_btn = gr.Button("Add Diacritics")
            arabic_output_diac = gr.Textbox(
                label="Diacritized Arabic Text",
                interactive=False,
                lines=3
            )
        
        # Event handlers
        process_btn.click(
            fn=process_medical_question,
            inputs=[darija_input],
            outputs=[status_output, english_translation, doctor_response, arabic_response, audio_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", "", "", None),
            outputs=[darija_input, status_output, english_translation, doctor_response, arabic_response, audio_output]
        )
        
        translate_btn1.click(
            fn=translate_darija_only,
            inputs=[darija_input_simple],
            outputs=[english_output_simple]
        )
        
        translate_btn2.click(
            fn=translate_english_to_arabic,
            inputs=[english_input_simple],
            outputs=[arabic_output_simple]
        )
        
        generate_btn.click(
            fn=generate_doctor_response,
            inputs=[medical_concern],
            outputs=[medical_response]
        )
        
        diacritize_btn.click(
            fn=diacritize_arabic_text,
            inputs=[arabic_input_diac],
            outputs=[arabic_output_diac]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            💡 **Tips:**
            - Make sure your API server is running on the configured address
            - For best results, ask clear and specific medical questions
            - The audio generation may take a moment to process
            """
        )
    
    return interface

def main():
    """Main function to launch the interface"""
    print("🚀 Starting Darija Medical Assistant Interface...")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    print("💾 GPU memory cleared")
    
    # Check API connection
    if check_api_health():
        print("✅ API server is responsive")
    else:
        print("⚠️ Warning: API server not responding. Please start your FastAPI server.")
    
    # Create and launch interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",  # Accessible from network
        server_port=7860,       # Default Gradio port
        share=False,            # Set True for public link
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()