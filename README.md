# **Calories Burnt Prediction**

This repository contains a Jupyter Notebook that demonstrates how to predict the number of calories burnt during exercise using machine learning techniques. The dataset used in this project combines exercise-related data with calorie information.

---

## **Overview**

Understanding calorie expenditure during exercise is crucial for fitness tracking and health management. This project uses an **XGBoost Regressor**, a powerful gradient boosting algorithm, to predict the number of calories burnt based on features such as age, gender, heart rate, and body temperature.

The dataset includes demographic and exercise-related features, with the target variable (`Calories`) representing the calories burnt.

---

## **Dataset**

- **Source**: The dataset appears to be related to publicly available exercise and calorie datasets.
- **Features**:
  - `User_ID`: Unique identifier for each individual.
  - `Gender`: Gender of the individual (male/female).
  - `Age`: Age of the individual.
  - `Height`: Height of the individual in centimeters.
  - `Weight`: Weight of the individual in kilograms.
  - `Duration`: Duration of exercise in minutes.
  - `Heart_Rate`: Heart rate during exercise (beats per minute).
  - `Body_Temp`: Body temperature during exercise (°C).
- **Target Variable**:
  - `Calories`: Calories burnt during exercise.

---

## **Project Workflow**

1. **Data Loading**:
   - Two datasets (`exercise.csv` and `calories.csv`) are loaded into Pandas DataFrames and merged on `User_ID`.
2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics and visualizations are generated using Seaborn and Matplotlib to explore relationships between features and calories burnt.
3. **Data Preprocessing**:
   - Categorical variables such as `Gender` are encoded into numerical values using Label Encoding.
   - Features are scaled where necessary for better model performance.
4. **Model Training**:
   - An XGBoost Regressor is trained to predict calorie expenditure.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Model Evaluation**:
   - Performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are calculated to evaluate model accuracy.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost

You can install these dependencies using pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/CaloriesBurntPrediction.git
   cd CaloriesBurntPrediction
   ```

2. Ensure that the dataset files (`exercise.csv` and `calories.csv`) are in the same directory as the notebook.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Calories-Burnt-Pred.ipynb
   ```

4. Run all cells in the notebook to execute the code.

---

## **Results**

The XGBoost Regressor provides predictions for calorie expenditure based on input features like age, gender, heart rate, and body temperature. Evaluation metrics such as MAE and RMSE indicate how well the model performs in predicting calories burned. Experimenting with hyperparameter tuning or feature engineering can further improve the model's performance.

---

## **Acknowledgments**

- The dataset was sourced from publicly available exercise and calorie datasets or repositories.
- Special thanks to XGBoost developers for providing a robust machine learning library.

---
