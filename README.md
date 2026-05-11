# CMP7005 PRAC1 — Beijing Air Quality Analysis

## Project Overview
This project analyses hourly air quality data from four monitoring 
stations in Beijing (March 2013 – February 2017) to investigate 
spatial and temporal pollution patterns and build a PM2.5 prediction model.

## Stations Selected
- **Urban:** Dongsi, Tiantan
- **Suburban:** Dingling, Huairou

## Project Structure
- `CMP7005_PRAC1.ipynb` — Main analysis notebook
- `app.py` — Streamlit interactive application
- `requirements.txt` — Python dependencies
- `PRSA_Data_*.csv` — Raw station datasets
- merged_air_quality.csv - Merged datasets
- `preprocessed_air_quality.csv` — Cleaned and processed dataset
- `best_model.pkl` — Trained Random Forest model
- `scaler.pkl` — Fitted StandardScaler

## Streamlit Application
Live app: https://cmp7005-prac1-beijing-airquality-analysis.streamlit.app

## Key Findings
- Urban stations show 22.7% higher PM2.5 than suburban stations
- SO2 declined 53.2% from 2013-2016 (strongest policy impact)
- PM2.5 is the primary AQI driver in 53.8% of readings
- Optimised Random Forest achieves R² = 0.89 for PM2.5 prediction
- CO is the dominant predictive feature (importance = 0.69)

## Technologies Used
Python, pandas, NumPy, matplotlib, seaborn, Plotly, scikit-learn, Streamlit

## Author
Student ID: ST20349610  
Module: CMP7005 Programming for Data Analysis  
Cardiff Metropolitan University
