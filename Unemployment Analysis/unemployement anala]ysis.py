# ==========================================
# CODEALPHA TASK 2: UNEMPLOYMENT ANALYSIS
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ==========================================
# LOAD DATASET
# ==========================================

# Replace with your dataset file name
df = pd.read_csv("Unemployment in India.csv")

# ==========================================
# DATA CLEANING
# ==========================================

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove missing values
df.dropna(inplace=True)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Display basic information
print("="*50)
print("DATASET INFORMATION")
print("="*50)
print(df.info())

print("\n")
print("="*50)
print("FIRST 5 ROWS")
print("="*50)
print(df.head())

print("\n")
print("="*50)
print("STATISTICAL SUMMARY")
print("="*50)
print(df.describe())

# ==========================================
# MISSING VALUES CHECK
# ==========================================

print("\n")
print("="*50)
print("MISSING VALUES")
print("="*50)
print(df.isnull().sum())

# ==========================================
# OVERALL UNEMPLOYMENT TREND
# ==========================================

plt.figure(figsize=(14,6))
sns.lineplot(
    data=df,
    x='Date',
    y='Estimated Unemployment Rate (%)'
)
plt.title("Overall Unemployment Trend in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# COVID-19 IMPACT ANALYSIS
# ==========================================

covid_period = df[df['Date'] >= '2020-03-01']

plt.figure(figsize=(14,6))
sns.lineplot(
    data=covid_period,
    x='Date',
    y='Estimated Unemployment Rate (%)'
)
plt.title("Unemployment Rate During COVID-19 Period")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# AVERAGE UNEMPLOYMENT RATE BY STATE
# ==========================================

state_unemployment = (
    df.groupby('Region')['Estimated Unemployment Rate (%)']
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,8))
state_unemployment.head(10).plot(
    kind='bar',
    color='skyblue'
)
plt.title("Top 10 States with Highest Average Unemployment Rate")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# EMPLOYMENT DISTRIBUTION
# ==========================================

plt.figure(figsize=(10,6))
sns.histplot(
    df['Estimated Unemployment Rate (%)'],
    bins=20,
    kde=True
)
plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.show()

# ==========================================
# URBAN VS RURAL ANALYSIS
# ==========================================

if 'Area' in df.columns:
    plt.figure(figsize=(8,6))
    sns.boxplot(
        data=df,
        x='Area',
        y='Estimated Unemployment Rate (%)'
    )
    plt.title("Urban vs Rural Unemployment Rate")
    plt.show()

# ==========================================
# LABOUR PARTICIPATION RATE ANALYSIS
# ==========================================

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x='Estimated Labour Participation Rate (%)',
    y='Estimated Unemployment Rate (%)'
)
plt.title("Labour Participation vs Unemployment Rate")
plt.xlabel("Labour Participation Rate (%)")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ==========================================
# HEATMAP (CORRELATION ANALYSIS)
# ==========================================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ==========================================
# MONTHLY UNEMPLOYMENT TREND
# ==========================================

df['Month'] = df['Date'].dt.month_name()

monthly_avg = (
    df.groupby('Month')['Estimated Unemployment Rate (%)']
    .mean()
)

month_order = [
    'January','February','March','April',
    'May','June','July','August',
    'September','October','November','December'
]

monthly_avg = monthly_avg.reindex(month_order)

plt.figure(figsize=(12,6))
monthly_avg.plot(marker='o')
plt.title("Average Monthly Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# TOP 10 HIGHEST UNEMPLOYMENT RECORDS
# ==========================================

print("\n")
print("="*50)
print("TOP 10 HIGHEST UNEMPLOYMENT RECORDS")
print("="*50)

top10 = df.nlargest(
    10,
    'Estimated Unemployment Rate (%)'
)

print(top10[['Region',
             'Date',
             'Estimated Unemployment Rate (%)']])

# ==========================================
# INSIGHTS
# ==========================================

print("\n")
print("="*60)
print("PROJECT INSIGHTS")
print("="*60)

print("""
1. COVID-19 caused a significant increase in unemployment rates.
2. Some states experienced much higher unemployment than others.
3. Employment levels dropped during lockdown periods.
4. Labour participation rate showed noticeable fluctuations.
5. Monthly analysis helps identify seasonal unemployment patterns.
6. State-wise analysis can help policymakers target employment programs.
7. Skill development and job creation initiatives are crucial in high-unemployment regions.
""")

print("\nAnalysis Completed Successfully!")