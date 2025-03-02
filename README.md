# Crop Yield Prediction using XGBoost

## Overview

This project predicts crop yield based on environmental factors like rainfall, pesticide usage, and temperature. It utilizes **XGBoost** for training and **Streamlit** for deployment.

## Dataset Details

**Source**: [Kaggle](https://www.kaggle.com)\
Two datasets are used:

1. **yield.csv**
   - `Domain Code`, `Domain`, `Area Code`, `Area`, `Element Code`, `Element`, `Item Code`, `Item`, `Year Code`, `Year`, `Unit`, `Value`
2. **yield\_df.csv**
   - `Area`, `Item`, `Year`, `hg/ha_yield`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`

## Tech Stack

- **Google Colab**: Data Preprocessing, Feature Engineering, EDA, Model Training (XGBoost)
- **VS Code**: Model Deployment using Streamlit & Joblib
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, XGBoost, Joblib, Streamlit

## Installation

### Clone the Repository

```bash
git clone https://github.com/vivekszan/Crop-Yield-Prediction-Using-XGBoost.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

## Features

- **Single Input Prediction**: Enter values manually
- **Batch Prediction**: Upload a CSV file

## Architecture Diagram

![Image](https://github.com/user-attachments/assets/b0538377-5629-4758-a72b-e2b8b8cdb09a)

## Project Structure

```
├── Source Code.ipynb  # Jupyter Notebook for training
├── yield.csv          # Raw dataset
├── yield_df.csv       # Processed dataset
├── sample_crop_yield_input.csv  # Demo input dataset
├── app.py             # Streamlit app
├── crop_yield_model.pkl  # Trained XGBoost model
└── README.md          # Documentation
```

## Sample Prediction

Using `sample_crop_yield_input.csv`:

```
| Year | Avg Rainfall | Pesticides | Avg Temp | Predicted Yield |
|------|-------------|------------|----------|-----------------|
| 2025 | 1200 mm     | 50 tonnes  | 25°C     | 3000 hg/ha      |
```

## Authors

Developed by John Nathanael J, Vivek S & Abdul Hashim S.

