# ===============================
# House Price Prediction Project
# Linear Regression
# ===============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 2. Load Dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target


# 3. Basic Exploration
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())


# 4. Visualization: Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.show()


# 5. Visualization: Feature vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['MedInc'], y=df['Price'], alpha=0.5)
plt.title("Median Income vs House Price")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.show()


# 6. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 7. Feature Selection
X = df.drop('Price', axis=1)
y = df['Price']


# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 9. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)


# 10. Predictions
y_pred = model.predict(X_test)


# 11. Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)


# 12. Visualization: Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


# 13. Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# 14. Feature Importance (Coefficients)
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Coefficients:\n", coeff_df)

plt.figure(figsize=(10,5))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df)
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.show()


# 15. Save Model
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved as house_price_model.pkl")


# 16. Load Model & Example Prediction
loaded_model = joblib.load("house_price_model.pkl")

sample_house = X.iloc[0].values.reshape(1, -1)
predicted_price = loaded_model.predict(sample_house)

print("\nExample Predicted House Price:", predicted_price[0])
