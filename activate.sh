#!/bin/bash
# Convenience script to activate the virtual environment

echo "🐍 Activating Medical Conversation System virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated!"
echo "📍 Current directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Pip version: $(pip --version)"
echo ""
echo "Ready to work! Run 'python main_conversation_app.py' to start the application."
