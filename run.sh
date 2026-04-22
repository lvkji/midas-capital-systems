#!/bin/bash

# Quick run script for Midas Capital Systems
# Automatically activates virtual environment and runs the app

echo "🚀 Starting Midas Capital Systems..."

# Check if virtual environment exists
if [ ! -d "midas_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run './setup.sh' first"
    exit 1
fi

# Activate virtual environment
source midas_env/bin/activate

# Run the application
streamlit run midas_enhanced.py

# Deactivate when streamlit closes
deactivate
