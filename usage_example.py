"""
Example usage of the modular medical conversation system
"""

import torch
import time
from medical_conversation_pipeline import MedicalConversationPipeline
from translation_module import TranslationModule
from response_generation_module import ResponseGenerationModule
from diacritization_module import DiacritizationModule

def example_text_only_processing():
    """
    Example of using the pipeline for text-only processing
    """
    print("=== Text-Only Processing Example ===\n")
    
    # Initialize the pipeline
    pipeline = MedicalConversationPipeline()
    
    # Example Darija input
    darija_input = "عندي صداع وحرارة"  # "I have a headache and fever"
    
    try:
        # Process the input through the complete pipeline
        result = pipeline.process_darija_input(
            darija_input, 
            add_diacritization=True  # Add diacritization to the output
        )
        
        print(f"Original Darija: {darija_input}")
        print(f"Final Arabic Response: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        pipeline.unload_all_models()

def example_individual_modules():
    """
    Example of using individual modules separately
    """
    print("=== Individual Modules Example ===\n")
    
    # Initialize individual modules
    translator = TranslationModule()
    response_gen = ResponseGenerationModule()
    diacritizer = DiacritizationModule()
    
    try:
        # Example 1: Just translation
        darija_text = "كيف الحال؟"  # "How are you?"
        english_text = translator.translate_darija_to_english(darija_text)
        print(f"Darija: {darija_text}")
        print(f"English: {english_text}\n")
        
        # Example 2: Generate response
        patient_concern = "I have been feeling dizzy lately"
        response = response_gen.generate_response(patient_concern)
        print(f"Patient: {patient_concern}")
        print(f"Receptionist: {response}\n")
        
        # Example 3: Translate to Arabic and add diacritization
        english_response = "What medications are you currently taking?"
        arabic_response = translator.translate_english_to_arabic(english_response)
        diacritized_response = diacritizer.diacritize_text(arabic_response)
        
        print(f"English: {english_response}")
        print(f"Arabic: {arabic_response}")
        print(f"Diacritized: {diacritized_response}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        translator.unload_models()
        response_gen.unload_model()
        diacritizer.unload_model()

def example_batch_processing():
    """
    Example of processing multiple inputs efficiently
    """
    print("=== Batch Processing Example ===\n")
    
    pipeline = MedicalConversationPipeline()
    
    # Multiple patient inputs
    patient_inputs = [
        "عندي وجع في البطن",      # "I have stomach pain"
        "ما قدرتش نرقد البارح",   # "I couldn't sleep last night"
        "عندي حساسية من الغبرة",   # "I have dust allergies"
    ]
    
    try:
        results = []
        for i, input_text in enumerate(patient_inputs, 1):
            print(f"Processing input {i}/{len(patient_inputs)}: {input_text}")
            result = pipeline.process_darija_input(input_text)
            results.append(result)
            print(f"Response: {result}\n")
            
            # Small delay between processing to manage memory
            time.sleep(1)
        
        print("=== All Results ===")
        for i, (input_text, result) in enumerate(zip(patient_inputs, results), 1):
            print(f"{i}. Input: {input_text}")
            print(f"   Output: {result}\n")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.unload_all_models()

def main():
    """
    Run all examples
    """
    print("🚀 Medical Conversation System - Usage Examples\n")
    
    # Clear GPU memory at start
    torch.cuda.empty_cache()
    
    # Run examples
    example_text_only_processing()
    print("\n" + "="*50 + "\n")
    
    example_individual_modules()
    print("\n" + "="*50 + "\n")
    
    example_batch_processing()
    
    print("✅ All examples completed!")

if __name__ == "__main__":
    main()
