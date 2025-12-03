# Subseasonal-to-Seasonal-Global-Crop-Yield-Forecasting
This repository contains the main code and source data used in the analysis and modeling for our paper, " Explainable Deep Learning Improves Subseasonal-to-Seasonal Global Crop Yield Forecasting ", submitted to Nature Communications. 

This repository provides a complete workflow for regional crop yield forecasting using multi-source dataset.

The framework integrates daily to 8-day scale environmental variables, vegetation indices, and soil properties, and applies LSTM models with hyperparameter optimization to predict and evaluate yield forecasting capability.

Files Included
Main Code
The Main code folder contains the GEE (Google Earth Engine) and Jupyter notebook scripts for different steps of the analysis pipeline£¨grouped into data acquisition, preprocessing, modeling, and evaluation phases£©. Below is a description of each file:

1.01_GEE_Download_Daily_Meteorology_VI.txt
Run on Google Earth Engine (GEE) to download daily meteorological and vegetation index data. Includes ERA5-Land variables (temperature_2m, precipitation, solar_radiation, wind, dewpoint_temperature), computes VPD and wind speed, and extracts NDVI, EVI, and KNDVI from MCD43A4. Also merges weighted soil and spatial information (AWC, CEC_SOIL, CLAY, ORG_CARBON, PH_WATER, SAND, SILT, TOTAL_N, Lon, Lat, Elevation).

2.02_GEE_Download_8Day_GPP_LAI_FPAR.txt
Run on GEE to download 8-day GPP, LAI, and FPAR datasets for the target region.
3.03_Aggregate_Daily_to_8Day_Region.ipynb
Aggregate daily data to 8-day scale. Compute derived variables: CDD, HDD, GDD, and SPEI. Merge vegetation indices and link with regional yield data.

4.04_Select_GrowthStage_VI.ipynb
Select the optimum vegetation indicator and extract three key growth-stage periods.

5.05_Select_Model_Features.ipynb
Choose the optimal combination of predictor variable  for regional model construction.

6.06_LSTM_Hyperparameter_Optimization.ipynb
Use Optuna to search for the best LSTM hyperparameter combination.

7.07_LSTM_Model_Training_and_Evaluation.ipynb
Train the LSTM model using the optimal parameters. Evaluate performance on test years against a null model and estimate factor importance via permutation analysis.

8.08_Download_S2S_ECMWF_Reforecast.ipynb
Automatically download ECMWF Subseasonal-to-Seasonal (S2S) reforecast data for the forecast years.


9.09_GEE_Download_Historical_weather.txt
Run on GEE to download 30 years of daily historical weather data.

10.10_Process_Historical_Weather_to_8Day.ipynb
Convert historical 30-year weather data to 8-day format. Use actual weather data before the forecast period and historical 30-year weather data after it.For VI, use actual values before the forecast and the most similar historical year after it.

11.11_Process_S2S_Data_to_8Day.ipynb
Process S2S weather forecast data into 8-day scale inputs.Use actual weather data before the forecast period and S2S forecasts after it.For VI, use actual values before the forecast and the most similar historical year after it.

12.12_Apply_Trained_LSTM_to_Dynamic_Yield_Forecast.ipynb
Integrate S2S and historical weather data into trained LSTM models to assess 8-day dynamic forecast accuracy for lead weeks 1-12.

13.13_LSTM_Shapley_Factor_Importance.ipynb
Import the trained model and modeling data to compute SHAP-based feature importance over time and by variable.
