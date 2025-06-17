# STT_Model_arabic.py
import librosa
import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import numpy as np

class boumehdi_wav2vec2:
    def __init__(self):
        # Add sampling_rate to the feature extractor to avoid silent errors
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "boumehdi/wav2vec2-large-xlsr-moroccan-darija",
            sampling_rate=16000
        )
        
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "boumehdi/wav2vec2-large-xlsr-moroccan-darija",
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )
        
        # Create processor with the properly configured feature extractor
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=self.tokenizer
        )
        
        self.model = Wav2Vec2ForCTC.from_pretrained('boumehdi/wav2vec2-large-xlsr-moroccan-darija')
    
    def stt(self, audio: tuple[int, np.ndarray]) -> str:
        _, input_audio = audio
        
        # Convert to float32 to match model's expected input type
        if input_audio.dtype != np.float32:
            input_audio = input_audio.astype(np.float32)
        
        # Process input audio
        input_values = self.processor(
            input_audio, 
            return_tensors="pt", 
            sampling_rate=16000, 
            padding=True
        ).input_values
        
        # Retrieve logits
        logits = self.model(input_values).logits
        tokens = torch.argmax(logits, axis=-1)
        
        # Decode using n-gram
        transcription = self.tokenizer.batch_decode(tokens)[0]  # Get first element of the batch
        return transcription

    def stt_from_file(self, path) -> str:
        input_audio, _ = librosa.load(path, sr=16000)

        # Convert to float32 to match model's expected input type
        if input_audio.dtype != np.float32:
            input_audio = input_audio.astype(np.float32)
        
        # Process input audio
        input_values = self.processor(
            input_audio, 
            return_tensors="pt", 
            sampling_rate=16000, 
            padding=True
        ).input_values
        
        # Retrieve logits
        logits = self.model(input_values).logits
        tokens = torch.argmax(logits, axis=-1)
        
        # Decode using n-gram
        transcription = self.tokenizer.batch_decode(tokens)[0]  # Get first element of the batch
        return transcription

def get_stt_model():
    return boumehdi_wav2vec2()