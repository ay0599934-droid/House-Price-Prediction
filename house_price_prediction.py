# ============================================
# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# Algorithm: Linear Regression
# Dataset: California Housing
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# STEP 1: LOAD DATASET

print("=" * 50)
print("HOUSE PRICE PREDICTION")
print("=" * 50)

print("\nLoading California Housing dataset...")

try:
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    print("Dataset loaded successfully!")

except Exception as error:
    print("Dataset loading failed:", error)
    print("Check your internet connection and try again.")
    raise SystemExit(1)


# STEP 2: EXPLORE DATASET

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())


# STEP 3: VISUALIZE DATA DISTRIBUTION

plt.figure(figsize=(8, 5))

plt.hist(df["MedHouseVal"], bins=30, edgecolor="black")

plt.title("Distribution of House Prices")
plt.xlabel("Median House Value (units of $100,000)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# STEP 4: SELECT FEATURES AND TARGET

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: MedHouseVal")


# STEP 5: SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# STEP 6: PREPROCESSING

# Fill missing values and standardize numerical features.
# Pipeline ensures preprocessing is fitted only on training data.

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# STEP 7: TRAIN LINEAR REGRESSION MODEL

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])

print("\nTraining Linear Regression model...")

model.fit(X_train, y_train)

print("Model trained successfully!")


# STEP 8: MAKE PREDICTIONS

y_pred = model.predict(X_test)


# STEP 9: EVALUATE MODEL

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R2 Score: {r2:.4f}")


# STEP 10: ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.4)

plt.xlabel("Actual House Value (units of $100,000)")
plt.ylabel("Predicted House Value (units of $100,000)")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()


# STEP 11: TEST A SAMPLE HOUSE

sample_house = X_test.iloc[[0]]

predicted_price = model.predict(sample_house)[0]

print("\nSample House Prediction")
print("-" * 30)

print("House features:")
print(sample_house.to_string(index=False))

print(f"\nPredicted house value: ${predicted_price * 100000:,.2f}")
print(f"Actual house value: ${y_test.iloc[0] * 100000:,.2f}")


# STEP 12: FINAL SUMMARY

print("\n" + "=" * 50)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 50)

print("Algorithm: Linear Regression")
print("Dataset: California Housing")
print(f"Test R2 Score: {r2:.4f}")
print(f"Test RMSE: {rmse:.4f} units of $100,000")