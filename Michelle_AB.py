#### Michelle's code for part AB #### 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('/Users/michelle/Desktop/LSE BCs Economics 23:26/24:25 Y2/EC2B1 - Macroeconomics II/[EC2B1] Course Project/EC2B1_pwt100.xlsx', sheet_name='Data', header = 0)
# Subset relevant columns and compute per capita real GDP
data = data.loc[:, ("country", "year", "rgdpe", "pop")]
data["rgdpe_pc"] = data["rgdpe"] / data["pop"]

# Filter data for Indonesia
indonesia_data = data.loc[data['country'] == 'Indonesia', ["year", "rgdpe", "pop", "rgdpe_pc"]]
indonesia_data = indonesia_data.reset_index(drop=True)

# ------------------------------
# TREND CALCULATION (LOG LEVELS)
# ------------------------------

# Extract relevant variables
indonesia_data['ln_rgdpna'] = np.log(indonesia_data['rgdpna'])
indonesia_data['ln_rgdppc'] = np.log(indonesia_data['rgdpna'] / indonesia_data['pop'])

# Add a time trend variable
indonesia_data['time'] = np.arange(len(indonesia_data))

# Fit linear trends
trend_ln_rgdpna = np.polyfit(indonesia_data['time'], indonesia_data['ln_rgdpna'], 1)
trend_ln_rgdppc = np.polyfit(indonesia_data['time'], indonesia_data['ln_rgdppc'], 1)

# Calculate fitted values
indonesia_data['trend_ln_rgdpna'] = np.polyval(trend_ln_rgdpna, indonesia_data['time'])
indonesia_data['trend_ln_rgdppc'] = np.polyval(trend_ln_rgdppc, indonesia_data['time'])

# Plot results
plt.figure(figsize=(12, 6))

# Panel 1: Real GDP
plt.subplot(2, 1, 1)
plt.plot(indonesia_data['year'], indonesia_data['ln_rgdpna'], label='ln(Real GDP)')
plt.plot(indonesia_data['year'], indonesia_data['trend_ln_rgdpna'], label='Trend', linestyle='--')
plt.title('Indonesia: Natural Log of Real GDP and Trend')
plt.xlabel('Year')
plt.ylabel('ln(Real GDP)')
plt.legend()

# Panel 2: Real GDP per capita
plt.subplot(2, 1, 2)
plt.plot(indonesia_data['year'], indonesia_data['ln_rgdppc'], label='ln(Real GDP per capita)')
plt.plot(indonesia_data['year'], indonesia_data['trend_ln_rgdppc'], label='Trend', linestyle='--')
plt.title('Indonesia: Natural Log of Real GDP per Capita and Trend')
plt.xlabel('Year')
plt.ylabel('ln(Real GDP per capita)')
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------
# ROBUSTNESS: TREND REGRESSION USING LEVELS 
# -----------------------------------------

# For robustness, you might also fit trends using the level of real GDP.
trend_rgdpna = np.polyfit(indonesia_data['time'], indonesia_data['rgdpna'], 1)
indonesia_data['trend_rgdpna'] = np.polyval(trend_rgdpna, indonesia_data['time'])

plt.figure(figsize=(8, 5))
plt.plot(indonesia_data['year'], indonesia_data['rgdpna'], label='Real GDP (Levels)')
plt.plot(indonesia_data['year'], indonesia_data['trend_rgdpna'], label='Trend (Levels)', linestyle='--')
plt.title('Indonesia: Real GDP and Trend (Levels)')
plt.xlabel('Year')
plt.ylabel('Real GDP')
plt.legend()
plt.show()

# ------------------------------
# TFP EXTRACTION
# ------------------------------

# Define alpha (capital share)
alpha = 0.3

# Calculate TFP
indonesia_data['tfp'] = indonesia_data['rgdpna'] / ( (indonesia_data['rnna'] ** alpha) * (indonesia_data['emp'] ** (1 - alpha)))

# Plot TFP
plt.figure(figsize=(8, 5))
plt.plot(indonesia_data['year'], indonesia_data['tfp'], label='TFP')
plt.title('Indonesia: Total Factor Productivity (TFP)')
plt.xlabel('Year')
plt.ylabel('TFP')
plt.legend()
plt.show()

# Calculate the natural log of real GDP
indonesia_data['ln_rgdpna'] = np.log(indonesia_data['rgdpna'])