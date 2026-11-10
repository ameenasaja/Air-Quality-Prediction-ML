# Air Quality Prediction Using Machine Learning
# This project explores air quality data and compares five regression models.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. Load and explore the dataset
# --------------------------------------------------

df = pd.read_csv("city_day.csv")

print("First five records:")
print(df.head())

print("\nDataset size (rows, columns):")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

print("\nCities in the dataset:")
print(df["City"].unique())


# --------------------------------------------------
# 2. Prepare the data for exploratory analysis
# --------------------------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nNumber of duplicate records:")
print(df.duplicated().sum())

df["Date"] = pd.to_datetime(
    df["Date"], dayfirst=False, errors="coerce"
)

# AQI is the value we want to predict, so rows without it
# cannot be used in the analysis or model training.
df = df.dropna(subset=["AQI"])

df = df.drop_duplicates()

# Fill missing numerical values with each column's median.
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Keep missing AQI categories clearly labelled.
df["AQI_Bucket"] = df["AQI_Bucket"].fillna("Unknown")

print("\nDataset size after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nData preparation for EDA is complete.")


# --------------------------------------------------
# 3. Explore the air quality data
# --------------------------------------------------

# 3.1 Distribution of AQI values
plt.figure(figsize=(10, 5))
sns.histplot(df["AQI"], bins=40, kde=True)
plt.title("Distribution of Air Quality Index (AQI)")
plt.xlabel("AQI")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.savefig("AQI_Distribution.png", dpi=300)
plt.show()


# 3.2 Ten cities with the highest average AQI
city_average_aqi = df.groupby("City")["AQI"].mean()
top_cities = city_average_aqi.sort_values(ascending=False).head(10)

plt.figure(figsize=(11, 6))
sns.barplot(x=top_cities.values, y=top_cities.index)
plt.title("Top 10 Cities by Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("Top_Polluted_Cities.png", dpi=300)
plt.show()


# 3.3 Compare the spread of AQI values across cities
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="City", y="AQI")
plt.title("AQI Distribution Across Cities")
plt.xlabel("City")
plt.ylabel("AQI")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("City_AQI_Comparison.png", dpi=300)
plt.show()


# 3.4 Correlation between numerical measurements
numeric_data = df.select_dtypes(include=np.number)
correlation = numeric_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Air Pollution Indicators")
plt.tight_layout()
plt.savefig("Correlation_Heatmap.png", dpi=300)
plt.show()


# 3.5 Average AQI for each calendar month across the dataset
df["Month"] = df["Date"].dt.month
monthly_average_aqi = df.groupby("Month")["AQI"].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(
    x=monthly_average_aqi.index,
    y=monthly_average_aqi.values,
    marker="o"
)
plt.title("Average AQI by Calendar Month")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("Monthly_AQI_Trend.png", dpi=300)
plt.show()


# 3.6 Relationship between PM2.5 and AQI
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="PM2.5",
    y="AQI",
    alpha=0.4
)
plt.title("PM2.5 Concentration and AQI")
plt.xlabel("PM2.5 concentration")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("PM25_AQI_Relationship.png", dpi=300)
plt.show()

print("\nAll six EDA graphs have been created.")


# --------------------------------------------------
# 4. Train and compare five regression models
# --------------------------------------------------

# Reload the original file for the modelling section.
# The imputer below is fitted on the training data only.
ml_df = pd.read_csv("city_day.csv")

ml_df["Date"] = pd.to_datetime(
    ml_df["Date"], dayfirst=False, errors="coerce"
)
ml_df["Month"] = ml_df["Date"].dt.month

# Rows without a target value cannot be used for supervised learning.
ml_df = ml_df.dropna(subset=["AQI"])

features = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene", "Month"
]

X = ml_df[features]
y = ml_df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

model_results = []
trained_models = {}

for model_name, model in models.items():

    # Missing feature values are filled using medians calculated
    # from the training data as part of this pipeline.
    model_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])

    model_pipeline.fit(X_train, y_train)
    predicted_aqi = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predicted_aqi)
    mse = mean_squared_error(y_test, predicted_aqi)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicted_aqi)

    model_results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[model_name] = model_pipeline

    print("\nModel:", model_name)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 score:", round(r2, 4))


# Put the model with the lowest RMSE at the top.
results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\nModel comparison:")
print(results_df.to_string(index=False))

results_df.to_csv("Model_Comparison.csv", index=False)

plt.figure(figsize=(11, 6))
sns.barplot(
    data=results_df,
    x="RMSE",
    y="Model"
)
plt.title("Comparison of Five Machine Learning Models")
plt.xlabel("Root Mean Squared Error")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig("Model_Comparison.png", dpi=300)
plt.show()

print("\nAll five models have been trained and evaluated.")


# --------------------------------------------------
# 5. Examine the best model's predictions
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest model based on RMSE:", best_model_name)

best_predictions = best_model.predict(X_test)

# Save the actual and predicted AQI values for later review.
prediction_results = pd.DataFrame({
    "Actual AQI": y_test.values,
    "Predicted AQI": best_predictions
})

prediction_results.to_csv("AQI_Predictions.csv", index=False)


# 5.1 Actual AQI compared with predicted AQI
plt.figure(figsize=(9, 6))
plt.scatter(y_test, best_predictions, alpha=0.4)

line_min = min(y_test.min(), best_predictions.min())
line_max = max(y_test.max(), best_predictions.max())

plt.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle="--"
)

plt.title("Actual vs Predicted AQI - " + best_model_name)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.tight_layout()
plt.savefig("Actual_vs_Predicted.png", dpi=300)
plt.show()


# 5.2 Residuals show the difference between actual and predicted AQI.
residuals = y_test.values - best_predictions

plt.figure(figsize=(9, 6))
plt.scatter(best_predictions, residuals, alpha=0.4)
plt.axhline(y=0, linestyle="--")
plt.title("Residual Analysis - " + best_model_name)
plt.xlabel("Predicted AQI")
plt.ylabel("Residual (actual minus predicted)")
plt.tight_layout()
plt.savefig("Residual_Analysis.png", dpi=300)
plt.show()


# 5.3 Distribution of prediction errors
plt.figure(figsize=(9, 5))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction error")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.savefig("Prediction_Error_Distribution.png", dpi=300)
plt.show()


# 5.4 Feature importance for tree-based models
tree_model = best_model.named_steps["model"]

if hasattr(tree_model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": tree_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )
    plt.title("Feature Importance - " + best_model_name)
    plt.tight_layout()
    plt.savefig("Feature_Importance.png", dpi=300)
    plt.show()

    importance_df.to_csv("Feature_Importance.csv", index=False)

else:
    print("Feature importance is not available for this model.")


# Save the selected model so it can be loaded again later.
joblib.dump(best_model, "Best_AQI_Model.pkl")

print("\nBest model:", best_model_name)
print("Prediction analysis is complete.")
print("Trained model saved as Best_AQI_Model.pkl")


# --------------------------------------------------
# 6. Demonstrate a prediction using one test record
# --------------------------------------------------

# This is an example using a record already in the test set.
# It is not a new, independently collected air-quality reading.
sample_input = X_test.iloc[[0]].copy()

sample_prediction = best_model.predict(sample_input)

print("\nExample prediction using a test record:")
print(sample_input)

print("\nPredicted AQI:", round(sample_prediction[0], 2))
print("Actual AQI:", round(y_test.iloc[0], 2))

print("\nPROJECT RUN COMPLETED.")
