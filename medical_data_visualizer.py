import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('medical_examination.csv')

# Calculate BMI and add overweight column
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['bmi'] > 25).astype(int)

# Normalize cholesterol and gluc columns
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Melt the DataFrame
df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
df_cat = df_cat.rename(columns={'variable': 'feature', 'value': 'value'})

# Group by cardio and feature
df_cat = df_cat.groupby(['cardio', 'feature', 'value']).size().reset_index(name='count')

# Create the categorical plot
sns.catplot(data=df_cat, kind='bar', x='feature', y='count', hue='value', col='cardio', height=5, aspect=1.5)

# Show the plot
plt.show()

# Filter out incorrect data
df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
             (df['height'] >= df['height'].quantile(0.025)) &
             (df['height'] <= df['height'].quantile(0.975)) &
             (df['weight'] >= df['weight'].quantile(0.025)) &
             (df['weight'] <= df['weight'].quantile(0.975))]

# Calculate correlation matrix
corr = df_heat.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Plot the heatmap
sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, cmap='coolwarm', cbar_kws={'shrink': 0.8})

# Show the plot
plt.show()
