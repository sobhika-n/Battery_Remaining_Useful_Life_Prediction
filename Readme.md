# Battery Remaining Useful Life Prediction

## Project Overview
The "Battery Remaining Useful Life Prediction" project leverages machine learning techniques to predict and classify the remaining useful life (RUL) of batteries based on various performance metrics. This project has significant implications in areas such as maintenance planning, resource allocation, and prolonging the life of batteries, ultimately contributing to cost savings and efficiency improvements.

## Python Packages Used
This project utilizes the following Python packages:
- Data Manipulation: pandas, numpy for efficient data handling.
- Data Visualization: seaborn, matplotlib for creating informative visualizations.
- Machine Learning: scikit-learn (sklearn), xgboost, TensorFlow (tensorflow) with Keras (keras)

## Project Files
Battery_RUL_EDA.ipynb: This notebook contains the exploratory data analysis (EDA) steps.
Battery_RUL_Modeling.ipynb: This notebook focuses on building machine learning models and evaluating their performance for RUL prediction.

## Dataset Information
The dataset used in this project contains the following columns:
- Cycle Index: A numerical value representing the number of cycles undergone by the battery.
- Discharge Time (s): The duration of the battery discharge in seconds during a cycle.
- Time at 4.15V (s): The time, in seconds, when the voltage reaches 4.15V during discharge.
- Time Constant Current (s): The duration, in seconds, of the constant current phase during discharge.
- Decrement 3.6-3.4V (s): The time, in seconds, taken for the voltage to decrease from 3.6V to 3.4V.
- Max. Voltage Discharge (V): The highest voltage reached during the discharge phase.
- Min. Voltage Charge (V): The lowest voltage recorded during the charging phase.
- Charging Time (s): The duration, in seconds, of the battery charging process.
- Total time (s): The total time, in seconds, taken for the battery cycle.
- RUL (Remaining Useful Life): The target variable representing the remaining useful life of the battery.

## Exploratory Data Analysis (EDA)
In this phase, we explore the dataset and prepare it for modeling. EDA involves the following steps:
- Data Exploration: Visualizing data distributions using histograms, bar plots, and box plots.
- Visualization: Univariate and multivariate analysis to gain insights into feature relationships.

## Feature Engineering
In this project, we remove certain features due to multicollinearity, simplifying the dataset for modeling.

## Modeling
The modeling phase involves building and evaluating machine learning models for RUL prediction. The main steps include:
- Data Preparation: Preparing the data for modeling.
- Model Selection: Evaluating various models, including Random Forest, XGBoost, Ridge Regression, and a Feedforward Neural Network.
- Model Evaluation: Assessing model performance using metrics like RMSE, MAE, R-squared, and Explained Variance.
- Model Comparison: Based on model evaluation metrics, we compare different models to identify the best-performing one for RUL prediction. Random Forest and XGBoost are the top contenders, while Ridge Regression and the Neural Network also contribute to the analysis.

## Hyperparameter Tuning
Hyperparameter tuning is performed for the two best-performing models, Random Forest and XGBoost, to optimize their performance.
- Random Forest Hyperparameter Tuning: GridSearchCV is used to find the best hyperparameters.
- XGBoost Hyperparameter Tuning: GridSearchCV is used to fine-tune the XGBoost model.

## Model Metrics Comparison
Here is a table of model metrics for different models sorted by various evaluation metrics:

|      Model            |    RMSE    |     MAE    | R-squared  | Explained Variance |
|-----------------------|------------|------------|------------|--------------------|
| Random Forest         | 3.7293     | 2.0978     | 0.9999     | 0.9999             |
| XGBoost               | 4.7873     | 3.1916     | 0.9998     | 0.9998             |
| Ridge Regression      | 7.4167     | 4.6425     | 0.9995     | 0.9995             |
| Neural Network        | 9.0735     | 6.2622     | 0.9992     | 0.9992             |
| Tuned Random Forest   | 3.6848     | 2.0675     | 0.9999     | 0.9999             |
| Tuned XGBoost         | 3.4024     | 1.7565     | 0.9999     | 0.9999             |

## Best Model: Random Forest or Tuned XGBoost
Random Forest and Tuned XGBoost are the top choices for predicting RUL due to their outstanding performance across all evaluation metrics.

## Conclusion
This project highlights the effectiveness of machine learning, specifically Random Forest and Tuned XGBoost, in RUL prediction, making them the recommended models for battery management across various industries and applications. In conclusion, when dealing with a not-so-large dataset for battery Remaining Useful Life (RUL) prediction, it's important to make informed choices to achieve accurate predictions and efficient battery lifecycle management. 
