import re
from shakkala import Shakkala
from typing import List, Optional

class DiacritizationModule:
    """
    Handles Arabic text diacritization using Shakkala
    """
    
    def __init__(self):
        self.shakkala_instance = None
    
    def load_model(self):
        """Load Shakkala model for Arabic diacritization"""
        if self.shakkala_instance is None:
            print("⏳ Loading Shakkala model for Arabic diacritization...")
            try:
                self.shakkala_instance = Shakkala()
                print("✅ Shakkala model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading Shakkala model: {e}")
                raise e
        return self.shakkala_instance
    
    def split_arabic_text(self, text: str, max_length: int = 315) -> List[str]:
        """
        Split Arabic text into chunks while preserving sentence boundaries
        
        Args:
            text: Arabic text to split
            max_length: Maximum length of each chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        # Arabic sentence endings
        sentence_endings = ['।', '؟', '!', '.', '؛']
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = re.split(r'([।؟!.؛])', text)
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            ending = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + ending
            
            # If adding this sentence would exceed limit, save current chunk
            if len(current_chunk) + len(full_sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = full_sentence
            else:
                current_chunk += full_sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If any chunk is still too long, split by words
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Split long chunks by words
                words = chunk.split()
                current_word_chunk = ""
                
                for word in words:
                    if len(current_word_chunk) + len(word) + 1 <= max_length:
                        current_word_chunk += (" " if current_word_chunk else "") + word
                    else:
                        if current_word_chunk:
                            final_chunks.append(current_word_chunk)
                        current_word_chunk = word
                        
                        # Handle extremely long single words
                        if len(word) > max_length:
                            # Split the word itself
                            while len(current_word_chunk) > max_length:
                                final_chunks.append(current_word_chunk[:max_length])
                                current_word_chunk = current_word_chunk[max_length:]
                
                if current_word_chunk:
                    final_chunks.append(current_word_chunk)
        
        return final_chunks
    
    def diacritize_text(self, text: str, add_harakat: bool = True) -> str:
        """
        Add diacritization (harakat) to Arabic text using Shakkala
        
        Args:
            text: Arabic text to diacritize
            add_harakat: Whether to add harakat or not
            
        Returns:
            Diacritized Arabic text
        """
        if not add_harakat:
            return text
            
        sh = self.load_model()
        print(f"🎯 Diacritizing Arabic text: '{text}'")
        
        try:
            # Split text into chunks if it's too long
            text_chunks = self.split_arabic_text(text, max_length=315)
            
            # Process each chunk
            diacritized_chunks = []
            for chunk in text_chunks:
                if chunk.strip():  # Skip empty chunks
                    # Prepare input for the model
                    arabic_response_prepared = sh.prepare_input(chunk)
                    
                    # Get the model
                    sh_model = sh.get_model()
                    
                    # Determine if get_model() returns a tuple (model, graph) or just the model
                    if isinstance(sh_model, tuple) and len(sh_model) > 0:
                        model = sh_model[0]
                    else:
                        model = sh_model
                    
                    # Predict harakat
                    logits = model.predict(arabic_response_prepared)[0]
                    predicted_harakat = sh.logits_to_text(logits)
                    
                    # Get final diacritized text
                    diacritized_chunk = sh.get_final_text(chunk, predicted_harakat)
                    diacritized_chunks.append(diacritized_chunk)
            
            # Join the processed chunks back together
            diacritized_text = " ".join(diacritized_chunks)
            
            print(f"🎯 Diacritization complete: '{diacritized_text}'")
            
            return diacritized_text
            
        except Exception as e:
            print(f"❌ Error in Arabic diacritization: {e}")
            # Return original text if diacritization fails
            return text
    
    def unload_model(self):
        """Unload the diacritization model to free memory"""
        if self.shakkala_instance:
            del self.shakkala_instance
            self.shakkala_instance = None
            print("✅ Shakkala model unloaded")
