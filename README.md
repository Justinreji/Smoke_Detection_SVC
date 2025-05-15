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

