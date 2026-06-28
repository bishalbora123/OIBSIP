import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set visualization style
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Data Loading and EDA
# ==========================================

# Load the dataset (using index_col=0 to handle the row number column)
df = pd.read_csv(r"C:\Users\Admin\Downloads\Advertising.csv", index_col=0)

print("--- First 5 rows of the dataset ---")
display(df.head())

print("\n--- Null Value Check ---")
print(df.isnull().sum())

print("\n--- Descriptive Statistics ---")
display(df.describe())

# Pairplot of all features
sns.pairplot(df, diag_kind='kde')
plt.suptitle('Pairplot of Advertising Data', y=1.02)
plt.show()

# ==========================================
# 2. Individual Scatter Plots
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(data=df, x='TV', y='Sales', ax=axes[0], color='blue')
axes[0].set_title('Sales vs. TV Spend')

sns.scatterplot(data=df, x='Radio', y='Sales', ax=axes[1], color='green')
axes[1].set_title('Sales vs. Radio Spend')

sns.scatterplot(data=df, x='Newspaper', y='Sales', ax=axes[2], color='orange')
axes[2].set_title('Sales vs. Newspaper Spend')

plt.tight_layout()
plt.show()

# ==========================================
# 3. Correlation Matrix Heatmap
# ==========================================

plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# ==========================================
# 4. Train/Test Split
# ==========================================

# Define features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ==========================================
# 5. Train Models (Baseline & Additional)
# ==========================================

# Model 1: Linear Regression (Baseline)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# ==========================================
# 6. Evaluate Models
# ==========================================

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} Evaluation ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}\n")
    return r2

lr_r2 = evaluate_model(y_test, lr_predictions, "Linear Regression")
rf_r2 = evaluate_model(y_test, rf_predictions, "Random Forest Regressor")

# ==========================================
# 7. Residual Plot for the Best Model
# ==========================================

# Random Forest generally performs better on this dataset due to non-linear interactions
best_predictions = rf_predictions if rf_r2 > lr_r2 else lr_predictions
best_model_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"

residuals = y_test - best_predictions

plt.figure(figsize=(8, 6))
sns.scatterplot(x=best_predictions, y=residuals, color='purple', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title(f'Residual Plot ({best_model_name})')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals (Errors)')
plt.show()

# ==========================================
# 8. Interpretation
# ==========================================

print("--- Model Interpretation ---")

# Interpretation using Linear Regression Coefficients
print("\nLinear Regression Coefficients:")
coef_df = pd.DataFrame(lr_model.coef_, X.columns, columns=['Coefficient'])
display(coef_df)

# Interpretation using Random Forest Feature Importances
print("\nRandom Forest Feature Importances:")
importance_df = pd.DataFrame(rf_model.feature_importances_, X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
display(importance_df)

print("\nConclusion:")
print("Based on both the Linear Regression coefficients and Random Forest feature importances,")
print("TV advertising spend has the highest impact on product sales. Radio is the second most")
print("effective channel, while Newspaper spend has the least (and sometimes negligible) impact.")
