import torch
import numpy as np
import gc
import warnings
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, MarianMTModel, MarianTokenizer
from typing import Optional

class TranslationModule:
    """
    Handles translation between Darija-English and English-Arabic
    """
    
    def __init__(self, device_map: str = "auto", model_dtype=torch.float16):
        self.device_map = device_map
        self.model_dtype = model_dtype
        
        # Atlas model for Darija to English
        self.atlas_pipeline = None
        
        # MarianMT model for English to Arabic
        self.marian_model = None
        self.marian_tokenizer = None
    
    def load_atlas_model(self):
        """Load Atlas model for Darija to English translation"""
        if self.atlas_pipeline is None:
            print("⏳ Loading translator model (Atlas-Chat-2B)...")
            device = self.device_map
            print(f"   Using device: {device} for translator model")

            model = AutoModelForCausalLM.from_pretrained(
                "MBZUAI-Paris/Atlas-Chat-2B",
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=True,
                device_map=device
            )
            tokenizer = AutoTokenizer.from_pretrained("MBZUAI-Paris/Atlas-Chat-2B")
            self.atlas_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                temperature=0.0
            )
            print("✅ Atlas translator model loaded successfully")
        return self.atlas_pipeline
    
    def load_marian_model(self):
        """Load MarianMT model for English to Arabic translation"""
        if self.marian_model is None or self.marian_tokenizer is None:
            print("⏳ Loading MarianMT model for English to Arabic translation...")
            model_name = "Helsinki-NLP/opus-mt-tc-big-en-ar"
            try:
                self.marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.marian_model = MarianMTModel.from_pretrained(model_name)
                print("✅ MarianMT model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading MarianMT model: {e}")
                raise e
        return self.marian_model, self.marian_tokenizer
    
    def translate_darija_to_english(self, text: str) -> str:
        """
        Translate Darija text to English using Atlas model
        
        Args:
            text: Darija text to translate
            
        Returns:
            English translation
        """
        pipe = self.load_atlas_model()
        print(f"🗣️ Translating from Darija to English: '{text}'")

        prompt_template = "You are a translator from Moroccan Darija to English, Translate this text to English: {text}"
        patterns = [r'"([^"])"', r'English: (.)', r'Translation: (.)', r'Translated text: (.)']

        messages = [{"role": "user", "content": prompt_template.format(text=text)}]

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
                match = re.search(pattern, str(full_response), re.IGNORECASE | re.DOTALL)
                if match:
                    translated_text = match.group(1).strip()
                    break
            if not translated_text:
                if ':' in str(full_response):
                    translated_text = str(full_response).split(':', 1)[-1].strip()
                else:
                    translated_text = str(full_response).strip()
                    if messages[0]['content'].lower() in translated_text.lower():
                        translated_text = translated_text.lower().replace(messages[0]['content'].lower(), "").strip()
                    if "assistant\n" in translated_text:
                        translated_text = translated_text.split("assistant\n")[-1].strip()

        print(f"🔄 Translation to English complete: '{translated_text}'")
        torch.cuda.empty_cache()
        gc.collect()
        return translated_text
    
    def translate_english_to_arabic(self, text: str) -> str:
        """
        Translate English text to Arabic using MarianMT model

        Args:
            text: English text to translate

        Returns:
            Arabic translation
        """
        model, tokenizer = self.load_marian_model()
        print(f"🗣️ Translating English to Arabic with MarianMT: '{text}'")

        try:
            # Format input text with language code as required by the model
            src_text = [f">>ara<< {text}"]

            # Tokenize and generate translation
            with torch.inference_mode():
                translated_tokens = model.generate(
                    **tokenizer(src_text, return_tensors="pt", padding=True)
                )

            # Decode the translation
            translated_text = ""
            for t in translated_tokens:
                translated_text = tokenizer.decode(t, skip_special_tokens=True)
                break  # We only have one text to translate

            print(f"🔄 MarianMT translation complete: '{translated_text}'")

            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()

            return translated_text

        except Exception as e:
            print(f"❌ Error in MarianMT translation: {e}")
            raise e
    
    def unload_models(self):
        """Unload all translation models to free memory"""
        if self.atlas_pipeline:
            del self.atlas_pipeline
            self.atlas_pipeline = None
            print("✅ Atlas model unloaded")
        
        if self.marian_model:
            del self.marian_model
            self.marian_model = None
            print("✅ MarianMT model unloaded")
        
        if self.marian_tokenizer:
            del self.marian_tokenizer
            self.marian_tokenizer = None
            print("✅ MarianMT tokenizer unloaded")
        
        torch.cuda.empty_cache()
        gc.collect()
