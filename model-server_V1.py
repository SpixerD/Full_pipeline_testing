import os
import time
import torch
import numpy as np
import gc
import warnings
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import uvicorn # For running the FastAPI app

# --- Model Configuration & Globals ---
# Memory management settings
DEVICE_MAP = "auto"
MODEL_DTYPE = torch.float16

qwen_model = None
qwen_tokenizer = None
atlas_pipeline = None

# --- Pydantic Models for Request/Response ---
class TextIn(BaseModel):
    text: str

class TextOut(BaseModel):
    processed_text: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM Processing API",
    description="API to access translation and response generation models.",
    version="1.0.0"
)

# --- Model Loading Functions (Modified for API context) ---
def load_translator_model_once():
    global atlas_pipeline
    if atlas_pipeline is None:
        print("⏳ Loading translator model (Atlas-Chat-2B)...")
        # Force CPU usage for translator as in original, or change if VRAM allows
        device = "cpu"
        # device = "cuda" if torch.cuda.is_available() else "cpu" # Alternative: use CUDA if available
        print(f"   Using device: {device} for translator model")

        model = AutoModelForCausalLM.from_pretrained(
            "MBZUAI-Paris/Atlas-Chat-2B",
            torch_dtype=MODEL_DTYPE,
            low_cpu_mem_usage=True, # Good for CPU loading
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained("MBZUAI-Paris/Atlas-Chat-2B")
        atlas_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=False, # Keep deterministic
            temperature=0.0
        )
        print("✅ Translator model loaded successfully")
    return atlas_pipeline

def load_response_model_once():
    global qwen_model, qwen_tokenizer
    if qwen_model is None or qwen_tokenizer is None:
        print("⏳ Loading response generation model (Qwen3-1.7B)...")
        model_name = "Qwen/Qwen3-1.7B" # Make sure this matches your needs
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device} for response model")

        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=MODEL_DTYPE,
            device_map=DEVICE_MAP # "auto" should work well here
        )
        print("✅ Response model loaded successfully")
    return qwen_model, qwen_tokenizer

# --- Core Logic Functions (Adapted from your original code) ---
def _translate_text(text_to_translate: str, target_language: str, source_language: str = "source"):
    pipe = load_translator_model_once()
    print(f"🗣️ API: Translating from {source_language} to {target_language}: '{text_to_translate}'")

    if target_language.lower() == "english":
        prompt_template = "You are a translator from Moroccan Darija to English, Translate this text to English: {text}"
        patterns = [r'"([^"]*)"', r'English: (.*)', r'Translation: (.*)', r'Translated text: (.*)']
    elif target_language.lower() == "arabic":
        prompt_template = "You are a translator from English to Arabic with diacritics (Tashkeel), Translate this text to Arabic with proper diacritics: {text}"
        patterns = [r'"([^"]*)"', r'Arabic: (.*)', r'Translation: (.*)', r'Translated text: (.*)']
    else:
        raise ValueError("Unsupported target language for translation")

    messages = [{"role": "user", "content": prompt_template.format(text=text_to_translate)}]

    with torch.inference_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            outputs = pipe(messages, max_new_tokens=256, temperature=0.0, do_sample=False)

    full_response = outputs[0]["generated_text"]
    translated_text = None

    if isinstance(full_response, list) and len(full_response) > 0:
        last_message = full_response[-1]
        if isinstance(last_message, dict) and "content" in last_message:
            translated_text = last_message["content"].strip()
        else:
            translated_text = str(last_message)
    else:
        for pattern in patterns:
            match = re.search(pattern, str(full_response), re.IGNORECASE | re.DOTALL) # Added re.DOTALL
            if match:
                translated_text = match.group(1).strip()
                break
        if not translated_text:
            if ':' in str(full_response):
                translated_text = str(full_response).split(':', 1)[-1].strip() # Split only on the first colon
            else:
                translated_text = str(full_response).strip()
                # If it's a response to a chat, it might contain the original prompt.
                # This is a common issue with chat models for translation.
                # Attempt to remove the original prompt if it's present at the start.
                if messages[0]['content'].lower() in translated_text.lower():
                     translated_text = translated_text.lower().replace(messages[0]['content'].lower(), "").strip()
                # Remove any trailing instructions or role markers if the model adds them
                if "assistant\n" in translated_text: # Specific to some models
                    translated_text = translated_text.split("assistant\n")[-1].strip()


    print(f"🔄 API: Translation to {target_language} complete: '{translated_text}'")
    torch.cuda.empty_cache() # Still good to clear cache after a call
    gc.collect()
    return translated_text

def _generate_qwen_response(english_prompt: str):
    model, tokenizer = load_response_model_once()
    print(f"💭 API: Generating Qwen response for: '{english_prompt}'")

    chat_prompt_template = "Generate a brief, 1-2 sentence doctor's response to this concern: {concern}"
    full_user_prompt = chat_prompt_template.format(concern=english_prompt)

    messages = [{"role": "user", "content": full_user_prompt}]

    formatted_prompt_for_model = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Qwen3 specific - set to False if not using its thinking feature
    )

    model_inputs = tokenizer([formatted_prompt_for_model], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id # Important for some models
            )
            # The generated_ids for Qwen include input_ids. We need to slice them off.
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # Qwen3's "thinking" feature might add special tokens like 151668 (</think>).
            # This parsing might need adjustment based on exact Qwen3 output.
            try:
                # rindex finding 151668 (</think>) or other markers.
                # If enable_thinking=False, this might not be necessary.
                unwanted_tokens = [151668, tokenizer.eos_token_id] # Add other special tokens if needed
                first_unwanted_idx = len(output_ids)
                for token_id in unwanted_tokens:
                    try:
                        idx = output_ids.index(token_id)
                        if idx < first_unwanted_idx:
                            first_unwanted_idx = idx
                    except ValueError:
                        pass # Token not found
                final_output_ids = output_ids[:first_unwanted_idx]

            except ValueError: # If </think> or other special tokens aren't found
                final_output_ids = output_ids

    response_text = tokenizer.decode(final_output_ids, skip_special_tokens=True).strip()
    print(f"Raw Qwen response_text: '{response_text}'") # For debugging

    # Clean up (this part might need model-specific adjustments)
    # Qwen response might not need splitting like original if add_generation_prompt=True is handled well
    assistant_response = response_text

    # Limit to 2 sentences
    sentences = re.split(r'(?<=[.!?])\s+', assistant_response)
    if len(sentences) > 2:
        assistant_response = ' '.join(sentences[:2]) + ('.' if not sentences[1].endswith(('.', '!', '?')) else '')


    print(f"✍️ API: Response generated: '{assistant_response}'")
    torch.cuda.empty_cache()
    gc.collect()
    return assistant_response

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Load models on server startup to make first API call faster."""
    print("Server starting up. Pre-loading models...")
    # You can choose to load all models or only frequently used ones here
    load_translator_model_once()
    load_response_model_once()
    print("Models pre-loading attempt complete.")


@app.post("/translate/darija-to-english", response_model=TextOut)
async def translate_darija_to_english_api(data: TextIn):
    try:
        translated_text = _translate_text(data.text, "english", "Moroccan Darija")
        return TextOut(processed_text=translated_text)
    except Exception as e:
        print(f"Error in Darija to English translation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/english-to-arabic", response_model=TextOut)
async def translate_english_to_arabic_api(data: TextIn):
    try:
        translated_text = _translate_text(data.text, "arabic", "English")
        return TextOut(processed_text=translated_text)
    except Exception as e:
        print(f"Error in English to Arabic translation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/doctor-response", response_model=TextOut)
async def generate_doctor_response_api(data: TextIn):
    try:
        response_text = _generate_qwen_response(data.text)
        return TextOut(processed_text=response_text)
    except Exception as e:
        print(f"Error in doctor response generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Main entry point for Uvicorn ---
if __name__ == "__main__":
    # Make sure to run on 0.0.0.0 to be accessible from other machines on your network
    # Choose a port that is not in use.
    uvicorn.run(app, host="0.0.0.0", port=8000)
