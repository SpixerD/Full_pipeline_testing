import torch
import gc
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class ResponseGenerationModule:
    """
    Handles response generation using Qwen model for medical receptionist tasks
    """
    
    def __init__(self, device_map: str = "auto", model_dtype=torch.float16):
        self.device_map = device_map
        self.model_dtype = model_dtype
        self.qwen_model = None
        self.qwen_tokenizer = None
    
    def load_model(self):
        """Load Qwen model for response generation"""
        if self.qwen_model is None or self.qwen_tokenizer is None:
            print("⏳ Loading response generation model (Qwen/Qwen3-4B)...")
            model_name = "Qwen/Qwen3-4B"
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Using device: {device} for response model")

            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.model_dtype,
                device_map=self.device_map
            )
            print("✅ Response model loaded successfully")
        return self.qwen_model, self.qwen_tokenizer
    
    def generate_response(self, english_prompt: str) -> str:
        """
        Generate a medical receptionist response based on English input using Qwen3-4B.
        
        Args:
            english_prompt: English text describing patient's concern
            
        Returns:
            Generated response from medical receptionist
        """
        model, tokenizer = self.load_model()
        print(f"💭 Generating Qwen response for: '{english_prompt}'")

        # 1. **CORRECTED**: Structure messages with distinct system and user roles.
        messages = [
            {"role": "system", "content": "You are a virtual medical receptionist. Answer briefly the patient's concern and ask simple medical questions that a receptionist at a doctor's office would ask."},
            {"role": "user", "content": english_prompt}
        ]

        # 2. **CUSTOMIZED FOR QWEN3**: Apply the chat template.
        #    We set `enable_thinking=False` for direct, faster responses suitable for a receptionist.
        formatted_prompt_for_model = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  
        )

        model_inputs = tokenizer([formatted_prompt_for_model], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            # Note: The Qwen3 documentation suggests specific generation parameters for non-thinking mode.
            # You might want to experiment with these for optimal performance.
            # e.g., temperature=0.7, top_p=0.8
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=100,  # 100 is plenty for a short receptionist response
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7, 
                top_p=0.8
            )
            
            # 3. **SIMPLIFIED**: The output parsing is now much simpler.
            #    Since thinking is disabled, we don't need to look for </think> tokens.
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        print(f"✍️ Response generated: '{response_text}'")
        
        torch.cuda.empty_cache()
        gc.collect()
        return response_text
    def unload_model(self):
        """Unload the response generation model to free memory"""
        if self.qwen_model:
            del self.qwen_model
            self.qwen_model = None
            print("✅ Qwen model unloaded")
        
        if self.qwen_tokenizer:
            del self.qwen_tokenizer
            self.qwen_tokenizer = None
            print("✅ Qwen tokenizer unloaded")
        
        torch.cuda.empty_cache()
        gc.collect()
