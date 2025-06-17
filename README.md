# Medical Conversation System

A modular medical conversation system that processes Moroccan Darija speech input and provides Arabic responses through a medical receptionist interface.

## Features

- 🎤 Speech-to-Text processing
- 🔄 Darija to English translation
- 💭 Medical receptionist response generation
- 🌐 English to Arabic translation
- 🎯 Arabic text diacritization
- 🔊 Text-to-Speech output
- 🧠 Modular architecture for easy maintenance

## Quick Start

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Copy your existing TTS and STT models:**
   - Copy your `TTS_Model_v2.py` content into the created file
   - Copy your `STT_Model.py` content into the created file

4. **Add the module code:**
   - Copy the provided code into each corresponding Python file

5. **Run the application:**
   ```bash
   python main_conversation_app.py
   ```

## Project Structure

```
medical_conversation_system/
├── venv/                             # Virtual environment
├── main_conversation_app.py          # Main application entry point
├── medical_conversation_pipeline.py  # Main pipeline orchestrator
├── translation_module.py             # Translation functionality
├── response_generation_module.py     # Response generation
├── diacritization_module.py         # Arabic diacritization
├── speech_processing_module.py      # STT/TTS processing
├── usage_example.py                 # Usage examples
├── TTS_Model_v2.py                  # TTS model (add your code)
├── STT_Model.py                     # STT model (add your code)
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Configuration

Edit `config.py` to adjust:
- Device settings (GPU/CPU)
- Model parameters
- Audio processing settings
- Memory management options

## Usage Examples

See `usage_example.py` for comprehensive examples of how to use the system.

## Contributing

1. Make sure to activate the virtual environment before development
2. Follow the modular architecture
3. Add proper error handling
4. Update documentation as needed

## License

[Add your license information here]
