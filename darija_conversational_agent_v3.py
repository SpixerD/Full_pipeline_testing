import os
import time
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
qwen_model = None
qwen_tokenizer = None
atlas_pipeline = None
# Memory management settings
DEVICE_MAP = "auto"  # Let the model decide where to allocate tensors
MODEL_DTYPE = torch.float16  # Use float16 for all models to reduce memory usage
# Synchronization mechanism to prevent concurrent execution
processing_lock = threading.Lock()

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

def load_translator_model():


    unload_translator_model()
    unload_response_model()
    unload_stt_model()
    unload_tts_model()

    

    global atlas_pipeline
    if atlas_pipeline is None:
        print("⏳ Loading translator model...")
        # Force CPU usage for translator to avoid GPU memory issues
        device = "cpu"
        print(f"   Using device: {device} for translator model")
        
        # Load the model with device_map="auto" to let it initialize properly
        model = AutoModelForCausalLM.from_pretrained(
            "MBZUAI-Paris/Atlas-Chat-2B",
            torch_dtype=MODEL_DTYPE,
            low_cpu_mem_usage=True,
            device_map=device  # Use CPU to avoid CUDA memory issues
        )
        
        tokenizer = AutoTokenizer.from_pretrained("MBZUAI-Paris/Atlas-Chat-2B")
        
        atlas_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            temperature=0.0
        )
        print("✅ Translator model loaded successfully")
    return atlas_pipeline

def unload_translator_model():
    global atlas_pipeline
    if atlas_pipeline is not None:
        print("⏳ Unloading translator model...")
        del atlas_pipeline
        atlas_pipeline = None
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ Translator model unloaded successfully")

def load_response_model():
    global qwen_model, qwen_tokenizer
    if qwen_model is None or qwen_tokenizer is None:
        print("⏳ Loading response generation model...")
        model_name = "Qwen/Qwen3-1.7B"
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check available VRAM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device} for response model")
        
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Response model loaded successfully")
    return qwen_model, qwen_tokenizer

def unload_response_model():
    global qwen_model, qwen_tokenizer
    if qwen_model is not None:
        print("⏳ Unloading response generation model...")
        del qwen_model
        del qwen_tokenizer
        qwen_model = None
        qwen_tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ Response model unloaded successfully")

def translate_to_english(prompt):
    # Load the translator model and unload others
    unload_response_model()
    unload_tts_model()
    pipe = load_translator_model()
    
    print(f"🗣️ Translating to English: '{prompt}'")
    # Improved prompt for more accurate translation
    messages = [
        {"role": "user", "content": f"You are a translator from Moroccan Darija to English, Translate this text to English: {prompt}"}
    ]
    
    with torch.inference_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            outputs = pipe(messages, max_new_tokens=256, temperature=0.0, do_sample=False)
    
    # Extract just the translation part
    full_response = outputs[0]["generated_text"]
    
    # Process the response to extract only the translated part
    if isinstance(full_response, list) and len(full_response) > 0:
        # Handle Atlas-Chat outputs that return a list of message dicts
        translated_prompt = full_response[-1]["content"].strip() if isinstance(full_response[-1], dict) and "content" in full_response[-1] else str(full_response[-1])
    else:
        # Handle string outputs
        # Try to extract the translation using various patterns
        translation_patterns = [
            r'"([^"]*)"',  # Text inside quotes
            r'English: (.*)',  # Text after "English:"
            r'Translation: (.*)',  # Text after "Translation:"
            r'Translated text: (.*)'  # Text after "Translated text:"
        ]
        
        translated_prompt = None
        for pattern in translation_patterns:
            match = re.search(pattern, str(full_response), re.IGNORECASE)
            if match:
                translated_prompt = match.group(1).strip()
                break
        
        if not translated_prompt:
            # If no patterns match, just take everything after the last colon or the full text
            if ':' in str(full_response):
                translated_prompt = str(full_response).split(':')[-1].strip()
            else:
                translated_prompt = str(full_response).strip()
    
    print(f"🔄 Translation to English complete: '{translated_prompt}'")
    torch.cuda.empty_cache()
    
    return translated_prompt

def translate_to_arabic(prompt):
    # Load the translator model and unload others
    unload_response_model()
    unload_tts_model()
    pipe = load_translator_model()
    
    print(f"🗣️ Translating to Arabic: '{prompt}'")
    # Improved prompt for more accurate translation
    messages = [
        {"role": "user", "content": f"You are a translator from English to Arabic with diacritics (Tashkeel), Translate this text to Arabic with proper diacritics: {prompt}"}
    ]
    
    with torch.inference_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            outputs = pipe(messages, max_new_tokens=256, temperature=0.0, do_sample=False)
    
    # Extract just the translation part
    full_response = outputs[0]["generated_text"]
    
    # Process the response to extract only the translated part
    if isinstance(full_response, list) and len(full_response) > 0:
        # Handle Atlas-Chat outputs that return a list of message dicts
        translated_prompt = full_response[-1]["content"].strip() if isinstance(full_response[-1], dict) and "content" in full_response[-1] else str(full_response[-1])
    else:
        # Handle string outputs
        # Try to extract the translation using various patterns
        translation_patterns = [
            r'"([^"]*)"',  # Text inside quotes
            r'Arabic: (.*)',  # Text after "Arabic:"
            r'Translation: (.*)',  # Text after "Translation:"
            r'Translated text: (.*)'  # Text after "Translated text:"
        ]
        
        translated_prompt = None
        for pattern in translation_patterns:
            match = re.search(pattern, str(full_response), re.IGNORECASE)
            if match:
                translated_prompt = match.group(1).strip()
                break
        
        if not translated_prompt:
            # If no patterns match, just take everything after the last colon or the full text
            if ':' in str(full_response):
                translated_prompt = str(full_response).split(':')[-1].strip()
            else:
                translated_prompt = str(full_response).strip()
    
    print(f"🔄 Translation to Arabic complete: '{translated_prompt}'")
    torch.cuda.empty_cache()
    
    return translated_prompt

def generate_response(translated_prompt):
    # Load response model and unload others
    unload_translator_model()
    unload_tts_model()
    model, tokenizer = load_response_model()
    
    print(f"💭 Generating response for: '{translated_prompt}'")
    # Improved prompt with stricter instructions to limit length

    chat_prompt = "Generate a brief, 1-2 sentence doctor's response to this concern: "
    
    # Format as a proper chat for Qwen
    messages = [
        {"role": "user", "content": chat_prompt}
    ]
    
    # Convert to the format expected by the model
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Get the model's device
    device = next(model.parameters()).device
    
    # Tokenize the input
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=100
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
        
        # Decode the generated tokens
    response_text = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("response text:")
    print(response_text)
    
    # Extract just the assistant's response - remove the prompt
    assistant_response = response_text.split(chat_prompt)[-1].strip()

    print("assistant_response:")
    print(assistant_response)
    
    # Clean up any system text or leftover tags
    for phrase in ["<human>", "<assistant>", "<system>", "system:", "user:", "assistant:"]:
        if phrase in assistant_response.lower():
            assistant_response = assistant_response.split(phrase)[0].strip()

    print("assistant_response v2:")
    print(assistant_response) 
    
    # Limit to 2 sentences maximum
    sentences = re.split(r'(?<=[.!?])\s+', assistant_response)
    if len(sentences) > 2:
        assistant_response = ' '.join(sentences[:2])
        
    print(f"✍️ Response generated: '{assistant_response}'")
    torch.cuda.empty_cache()
    
    return assistant_response

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
            unload_translator_model()
            unload_response_model()
            unload_tts_model()
            stt = load_stt_model()
            
            print("🎤 Processing speech input...")
            prompt = stt.stt(audio)
            
            # Free STT model memory immediately
            unload_stt_model()
            
            if not (prompt and prompt.strip()):
                print("⚠️ Empty speech recognized, nothing to process")
                return
                
            print(f"👂 Recognized speech: '{prompt}'")
            
            # Step 2: Translate to English
            translated_text = translate_to_english(prompt)
            
            # Step 3: Generate response
            response_text_english = generate_response(translated_text)

            
            # Step 4: Translate back to Arabic
            response_text_arabic = translate_to_arabic(response_text_english)
            
            # Step 5: Text to speech
            unload_response_model()
            unload_translator_model()
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
            unload_translator_model()
            unload_response_model()
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