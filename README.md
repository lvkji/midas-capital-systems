# Midas Capital Systems - Quick Start Guide

### Option 1: Use the setup script

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup (this will create virtual environment and install everything)
./setup.sh

# Run the app
source midas_env/bin/activate
streamlit run midas_enhanced.py
```

### Option 2: Use the run script (after setup)

```bash
# Make it executable
chmod +x run.sh

# Run the app (automatically activates venv)
./run.sh
```

---

## Manual Setup

### Step 1: Create Virtual Environment
```bash
python3 -m venv midas_env
```

### Step 2: Activate Virtual Environment
```bash
source midas_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run midas_enhanced.py
```

### Step 5: When Done, Deactivate
```bash
deactivate
```

---

## Troubleshooting

### Issue: "python3-venv not found"
```bash
sudo apt update
sudo apt install python3-venv python3-full
```

### Issue: "Permission denied" when running scripts
```bash
chmod +x setup.sh run.sh
```

### Issue: Streamlit won't start
```bash
# Make sure you're in the virtual environment
source midas_env/bin/activate

# Check if streamlit is installed
pip list | grep streamlit

# If not, reinstall
pip install streamlit
```

---

## `What's Included

### Files:
- `midas_enhanced.py` - Main application
- `requirements.txt` - Python dependencies
- `setup.sh` - Automated setup script
- `run.sh` - Quick run script
- `PROJECT_GUIDE.md` - Feature ideas and implementation guide

### Features:
- Real-time price updates (auto-refresh)
- Machine learning predictions (Random Forest)
- Technical analysis (15 indicators)
- Beautiful UI with animations
- Interactive Plotly charts
- Portfolio analytics (Sharpe ratio, returns)
- Sector allocation
- Trade history

---

## Quick Demo Flow

1. **Start the app** (it opens in browser automatically)
2. **Dashboard Tab** - See your account overview
3. **AI Insights Tab** - Pick a stock and click "Generate Prediction"
   - Shows 5-day forecast
   - Displays confidence scores
   - Gives BUY/SELL signal
4. **Trade Tab** - Buy some stocks
5. **Portfolio Tab** - See your positions and sector allocation
6. **Performance Tab** - View your equity curve

---

---

## Common Commands

```bash
# Activate virtual environment
source midas_env/bin/activate

# Run the app
streamlit run midas_enhanced.py

# Check what's installed
pip list

# Update a package
pip install --upgrade streamlit

# Deactivate virtual environment
deactivate
```

---

---

## Project Info

**Developer:** Andrew Ignatius  
**Year:** 2026  
**Project:** Senior Project - AI-Powered Paper Trading Platform

---

**Always remember to activate before running:**
```bash
source midas_env/bin/activate
```

**You'll know it's activated when you see:**
```bash
(midas_env) adignati@adignati-HP-All-in-One-24-df1xxx:~$
```

### Streamlit Issues
If the browser doesn't open automatically:
1. Look for the URL in the terminal (usually `http://localhost:8501`)
2. Copy and paste it into your browser

---

**Happy Trading! 🚀📈**
