# Air-Quality-Prediction-ML
Air Quality Index Prediction using five Machine Learning regression models
# Air Quality Prediction Using Machine Learning 🌍

## Project Overview

This project focuses on analysing air pollution data across Indian cities and predicting the Air Quality Index (AQI) using Machine Learning regression algorithms.

The objective is to understand the relationship between different air pollutants and AQI, perform exploratory data analysis, and compare the performance of multiple regression models.

## Dataset

**Dataset:** Air Quality Data in India (2015–2020)

The dataset contains air pollution measurements collected across multiple Indian cities.

The features include:
- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- Benzene
- Toluene
- Xylene

**Target variable:** AQI (Air Quality Index)

## Project Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Handling missing values and duplicate records
4. Exploratory Data Analysis (EDA)
5. Feature preparation
6. Training and evaluating machine learning models
7. Comparing model performance
8. Analysing feature importance

## Exploratory Data Analysis

The project includes visualisations such as:

- AQI distribution
- Average AQI across cities
- Correlation heatmap
- AQI by calendar month
- PM2.5 vs AQI relationship
- Actual vs Predicted AQI
- Residual analysis
- Prediction error distribution
- Feature importance

## Machine Learning Models

Five regression models were implemented:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Gradient Boosting Regression
5. Extra Trees Regression

## Model Evaluation

The models were evaluated using:

- MAE – Mean Absolute Error
- MSE – Mean Squared Error
- RMSE – Root Mean Squared Error
- R² Score – Coefficient of Determination

### Model Comparison Results

| Model | RMSE | R² Score |
|---|---:|---:|
| Random Forest | 40.78 | 0.9092 |
| Extra Trees | 41.38 | 0.9065 |
| Gradient Boosting | 43.40 | 0.8972 |
| Decision Tree | 52.16 | 0.8514 |
| Linear Regression | 59.45 | 0.8070 |

**Best-performing model:** Random Forest Regression

It achieved the lowest RMSE and highest R² score among the five models in this evaluation.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run the Project

1. Clone or download this repository.
2. Make sure Python is installed.
3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

4. Keep `air_quality_readable.py` and `city_day.csv` in the same folder.
5. Run the Python file:

```bash
python air_quality_readable.py
```

The script generates model evaluation results, visualisations, prediction outputs and a saved trained model.

## Project Files

- `air_quality_readable.py` – Main Python implementation
- `city_day.csv` – Dataset
- `Model_Comparison.csv` – Model evaluation results
- `Model_Comparison.png` – Model comparison visualisation
- `Feature_Importance.png` – Feature importance graph
- Other PNG files – EDA and prediction visualisations

## Key Learning Outcomes

Through this project, I gained practical experience in:

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Machine Learning regression
- Model evaluation and comparison
- Data visualisation
- Interpreting feature importance

## Limitations

The model estimates AQI using pollutant measurements. Its evaluation uses a random train-test split, so further testing on later dates and unseen cities is needed to assess how well it generalises.

## Author

**Ameena Saja**

B.Tech – Computer Science and Engineering  
Lovely Professional University

---

*This project was developed as part of my academic learning in Machine Learning and Data Analytics.*
