import time
import torch
import gc
from typing import Optional

from translation_module import TranslationModule
from response_generation_module import ResponseGenerationModule
from diacritization_module import DiacritizationModule

class MedicalConversationPipeline:
    """
    Main pipeline class that orchestrates the complete medical conversation workflow
    """
    
    def __init__(self, device_map: str = "auto", model_dtype=torch.float16):
        self.device_map = device_map
        self.model_dtype = model_dtype
        
        # Initialize modules
        self.translator = TranslationModule(device_map, model_dtype)
        #self.response_generator = ResponseGenerationModule(device_map, model_dtype)
        self.diacritizer = DiacritizationModule()
    
    def process_darija_input(self, darija_input: str, add_diacritization: bool = False) -> str:
        """
        Complete pipeline: Darija -> English -> Response -> Arabic (-> Diacritization)
        
        Args:
            darija_input: Input text in Moroccan Darija
            add_diacritization: Whether to add diacritization to the final Arabic response
            
        Returns:
            Final Arabic response (optionally diacritized)
        """
        print(f"\n🚀 Starting full processing pipeline for: '{darija_input}'")
        start_time = time.time()

        try:
            # Step 1: Translate Darija to English
            print("📝 Step 1: Translating Darija to English...")
            english_text = self.translator.translate_darija_to_english(darija_input)
            if not english_text:
                raise Exception("Failed to translate Darija to English")

            # Step 2: Generate response in English
            print("💭 Step 2: Generating medical response...")
            english_response = "I suggest that you talk to a doctor about that."
            #english_response = self.response_generator.generate_response(english_text)
            if not english_response:
                raise Exception("Failed to generate medical response")

            # Step 3: Translate English response back to Arabic
            print("🔄 Step 3: Translating response to Arabic...")
            arabic_response = self.translator.translate_english_to_arabic(english_response)
            if not arabic_response:
                raise Exception("Failed to translate response to Arabic")

            # Step 4: Optional diacritization
            final_response = arabic_response
            if add_diacritization:
                print("🎯 Step 4: Adding diacritization...")
                final_response = self.diacritizer.diacritize_text(arabic_response, add_harakat=True)

            end_time = time.time()
            print(f"✅ Processing complete in {end_time - start_time:.2f} seconds.")
            print(f"🎯 Final response: '{final_response}'")
            
            return final_response

        except Exception as e:
            print(f"❌ Error in pipeline processing: {e}")
            raise e
    
    def translate_and_diacritize(self, english_text: str) -> str:
        """
        Translate English to Arabic and add diacritization
        
        Args:
            english_text: English text to translate
            
        Returns:
            Diacritized Arabic text
        """
        print(f"🔄 Translating and diacritizing: '{english_text}'")
        
        try:
            # Translate to Arabic
            arabic_text = self.translator.translate_english_to_arabic(english_text)
            
            # Add diacritization
            diacritized_text = self.diacritizer.diacritize_text(arabic_text, add_harakat=True)
            
            return diacritized_text
            
        except Exception as e:
            print(f"❌ Error in translation and diacritization: {e}")
            raise e
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        torch.cuda.empty_cache()
        gc.collect()
        print("💾 GPU memory cleaned")
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        print("🧹 Unloading all models...")
        self.translator.unload_models()
        #self.response_generator.unload_model()
        self.diacritizer.unload_model()
        self.cleanup_memory()
        print("✅ All models unloaded successfully")
    
    def get_model_status(self) -> dict:
        """Get the loading status of all models"""
        return {
            "atlas_pipeline": self.translator.atlas_pipeline is not None,
            "marian_model": self.translator.marian_model is not None,
            #"qwen_model": self.response_generator.qwen_model is not None,
            "shakkala_instance": self.diacritizer.shakkala_instance is not None,
        }
