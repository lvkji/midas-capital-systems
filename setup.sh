#!/bin/bash

# Midas Capital Systems - Easy Setup Script
# This will create a virtual environment and install all dependencies

echo "🚀 Setting up Midas Capital Systems..."
echo ""

# Check if python3-venv is installed
if ! dpkg -l | grep -q python3-venv; then
    echo "📦 Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3-venv python3-full
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv midas_env

# Activate virtual environment
echo "✅ Activating virtual environment..."
source midas_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✨ Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment:"
echo "   source midas_env/bin/activate"
echo ""
echo "2. Run the app:"
echo "   streamlit run midas_enhanced.py"
echo ""
echo "3. When you're done, deactivate with:"
echo "   deactivate"
echo ""
