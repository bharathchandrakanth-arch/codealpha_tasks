# ==========================================
# SALES PREDICTION USING PYTHON
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("advertising.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# DATA CLEANING
# ==========================================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing values if any
df.fillna(df.mean(numeric_only=True), inplace=True)

# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

print("\nStatistical Summary:")
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pair Plot
sns.pairplot(df)
plt.show()

# ==========================================
# FEATURE SELECTION
# ==========================================

# Assuming dataset columns:
# TV, Radio, Newspaper, Sales

X = df.drop("Sales", axis=1)
y = df["Sales"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODEL BUILDING
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

importance = importance.sort_values(
    by='Coefficient',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8,5))
sns.barplot(
    x='Coefficient',
    y='Feature',
    data=importance
)
plt.title("Advertising Impact on Sales")
plt.show()

# ==========================================
# FUTURE SALES PREDICTION
# ==========================================

future_data = pd.DataFrame({
    'TV':[250],
    'Radio':[40],
    'Newspaper':[50]
})

future_sales = model.predict(future_data)

print("\nPredicted Future Sales:")
print(future_sales[0])

# ==========================================
# BUSINESS INSIGHTS
# ==========================================

print("\nBusiness Insights:")
print("1. Identify the advertising channel with highest coefficient.")
print("2. Increase budget allocation to the most influential channel.")
print("3. Reduce spending on low-impact channels.")
print("4. Use model predictions before launching campaigns.")
print("5. Continuously monitor campaign performance.")