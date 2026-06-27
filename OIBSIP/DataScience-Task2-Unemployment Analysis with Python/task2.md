import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Load the dataset
# Replace with your actual file name if different
df = pd.read_csv(r"C:\Users\Admin\Downloads\Unemployment in India.csv")

# 2. Data loading and shape inspection
print(f"Dataset Shape: {df.shape}\n")

# Strip leading/trailing spaces from column names to avoid KeyErrors
df.columns = df.columns.str.strip()

# 3. Null value check
print("Null values per column:")
print(df.isnull().sum())

# Drop rows with null values (if any exist at the bottom of the CSV)
df = df.dropna()

# 4. Type conversion: Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y')

# Verify the cleaned data
df.info()

# Calculate region-wise average unemployment rate
region_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
print("Region-wise Average Unemployment Rate (%):\n", region_avg.head())

# Extract Month and Year for temporal EDA
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Calculate month-wise trends (aggregating across all states)
monthly_trend = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()

# Select 3 major states
major_states = ['Maharashtra', 'Delhi', 'Uttar Pradesh']
df_major = df[df['Region'].isin(major_states)]

# Plot time-series line chart
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_major, x='Date', y='Estimated Unemployment Rate (%)', hue='Region', marker='o')
plt.title('Unemployment Rate Over Time (Selected States)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Unemployment Rate (%)', fontsize=12)
plt.legend(title='State')
plt.show()

# Get top 10 states by average unemployment rate
top_10_states = region_avg.head(10).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_states, x='Estimated Unemployment Rate (%)', y='Region', palette='viridis')
plt.title('Top 10 States with Highest Average Unemployment Rate', fontsize=16)
plt.xlabel('Average Unemployment Rate (%)', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.show()

# Select the numeric columns for correlation
numeric_cols = ['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Employment Metrics', fontsize=16)
plt.show()

# Define the threshold date for the COVID-19 lockdown in India
lockdown_date = pd.to_datetime('2020-04-01')

# Split the data
pre_covid_df = df[df['Date'] < lockdown_date]
post_covid_df = df[df['Date'] >= lockdown_date]

# Calculate mean rates for each period
pre_covid_mean = pre_covid_df['Estimated Unemployment Rate (%)'].mean()
post_covid_mean = post_covid_df['Estimated Unemployment Rate (%)'].mean()

print(f"Average Unemployment Rate (Pre-COVID): {pre_covid_mean:.2f}%")
print(f"Average Unemployment Rate (Post-COVID): {post_covid_mean:.2f}%")

# Visualize the comparison
comparison_data = pd.DataFrame({
    'Period': ['Pre-COVID (Before Apr 2020)', 'Post-COVID (Apr 2020 onwards)'],
    'Average Unemployment Rate (%)': [pre_covid_mean, post_covid_mean]
})

plt.figure(figsize=(8, 5))
sns.barplot(data=comparison_data, x='Period', y='Average Unemployment Rate (%)', palette=['#3498db', '#e74c3c'])
plt.title('Impact of COVID-19 on Unemployment in India', fontsize=16)
plt.ylabel('Average Unemployment Rate (%)', fontsize=12)
plt.show()
