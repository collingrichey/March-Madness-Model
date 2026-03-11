#!/bin/bash

# March Madness Prediction Model - Install Required Libraries
# Run this script with: bash install_requirements.sh

echo "============================================"
echo "Installing Python Libraries"
echo "============================================"
echo ""

# Core data manipulation and analysis
echo "Installing pandas, numpy, scipy..."
pip install pandas numpy scipy

# Machine learning
echo "Installing scikit-learn..."
pip install scikit-learn

# Optional: Advanced gradient boosting (uncomment if you want to use)
# echo "Installing xgboost and lightgbm..."
# pip install xgboost lightgbm

# Web scraping
echo "Installing requests, beautifulsoup4, lxml..."
pip install requests beautifulsoup4 lxml html5lib

# For handling SSL certificates
echo "Installing urllib3..."
pip install urllib3

# Visualization
echo "Installing matplotlib and seaborn..."
pip install matplotlib seaborn

# File handling
echo "Installing openpyxl..."
pip install openpyxl

# Optional but useful
echo "Installing jupyter, tqdm, joblib..."
pip install jupyter ipython tqdm joblib

echo ""
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "All required libraries have been installed."
echo "You can now run:"
echo "  python data_scraper.py    - to scrape data"
echo "  python prediction_model.py - to train the model"
echo "  python bracket_sim.py     - to simulate brackets"
echo ""
