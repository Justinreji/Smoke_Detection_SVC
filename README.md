# Fire Alarm Detection using IoT Sensor Data

This project applies machine learning techniques to classify whether a fire alarm should be triggered based on various environmental sensor readings. The goal is to use Support Vector Machines (SVM) to predict fire risk from real-world IoT data accurately.

## Dataset

Source: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset

Records: 62,629 observations

## Features:

| Feature             | Description                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| **Temperature\[C]** | Ambient temperature measured in degrees Celsius.                         |
| **Humidity\[%]**    | Relative humidity of the air in percentage.                              |
| **TVOC\[ppb]**      | Total Volatile Organic Compounds concentration in parts per billion.     |
| **eCO2\[ppm]**      | Equivalent carbon dioxide level inferred from VOCs in parts per million. |
| **Raw H2**          | Raw sensor signal representing hydrogen gas concentration.               |
| **Raw Ethanol**     | Raw sensor signal representing ethanol vapor concentration.              |
| **Pressure\[hPa]**  | Atmospheric pressure measured in hectopascals.                           |
| **PM1.0**           | Concentration of particulate matter ≤ 1.0 microns in diameter.           |
| **PM2.5**           | Concentration of particulate matter ≤ 2.5 microns in diameter.           |
| **NC0.5**           | Number concentration of particles > 0.5 microns per unit air volume.     |
| **NC1.0**           | Number concentration of particles > 1.0 microns per unit air volume.     |
| **NC2.5**           | Number concentration of particles > 2.5 microns per unit air volume.     |
| **CNT**             | Sensor reading index or sample counter.                                  |
| **Fire Alarm**      | Target variable: 1 indicates fire detected, 0 means no fire detected.    |

## Data Preprocessing

- Removed unnecessary columns (Unnamed: 0, UTC)
- Handled missing values
- Sampled 10,000 rows for efficient training
- Normalized feature values using standard scaling

## Exploratory Data Analysis

Bar plot showing class imbalance between fire alarms triggered vs not triggered

![fire_alarm_class_dist](https://github.com/user-attachments/assets/d2a86d35-fdbf-4acf-862a-86fab6ed6bac)

Correlation heatmap to identify feature relationships

![feature_correlation](https://github.com/user-attachments/assets/fc62e705-043c-401d-9709-2a69b0f9b960)


