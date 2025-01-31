# Import modules
import pandas as pd
import numpy as np

from get_regression_coefs_general import get_regression_coefs

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme('talk', style = 'white')

import openpyxl
print(openpyxl.__version__)

# Set display options
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)  
pd.set_option('display.max_colwidth', None) 

# Load dataset
data = pd.read_excel('pwt100.xlsx', sheet_name = 'Data', header = 0)

## Part A 

# Clean data 

# Subset relevant columns and compute per capita real GDP
data = data.loc[:, ("country", "year", "rgdpna", "pop", "emp", "rnna")]
data["rgdpna_pc"] = data["rgdpna"] / data["pop"]

# Select Indonesia as country
data = data.loc[data["country"] == "Indonesia", ("year", "rgdpna_pc", "rgdpna", "emp", "rnna")]
data = data.reset_index(drop = True)

# Subset the RGDP per capita series
ymax = 2019
ymin = 1960
Y = data.loc[np.logical_and(data["year"] <= ymax, data["year"] >= ymin), "rgdpna_pc"]
y = np.log(Y)
data = data[data["year"] >= ymin] 


# Compute separate sample sizes for the subsample used for regression and the whole sample
T = len(Y) # sample size used for regression
T_all = data["year"].max() - (ymin - 1) # number of all years in the data 

# 1) Additive Linear Model
# First regressor x1 is T x 1 vector of ones, second regressor x2 is the vector 1, 2, ..., T
# The dependent variable is per capital GDP in levels 

x1 = np.empty(T) 
x2 = np.empty(T) 

for t in range(T):
    x1[t] = 1.
    x2[t] = t + 1 

a_add_lin, b_add_lin = get_regression_coefs(Y, x1, x2)

# Initialise predicted values yhat 
Yhat_add_lin = np.empty(T_all)

# Create loop to compute trend for all years
for t in range(T_all):
    Yhat_add_lin[t] = a_add_lin + b_add_lin * (t + 1) # recall that Python indexing starts at 0

# Convert into log units
yhat_add_lin = np.log(Yhat_add_lin)

lw = 4

# Plotting the figure
plt.figure()
plt.plot(data['year'],np.log(data['rgdpna_pc']))
plt.plot(data['year'],yhat_add_lin)

#labels
plt.xlabel("year")
plt.ylabel("ln(GDP)")

#title
plt.title("Linear Additive: GDP & trend")

plt.show()

# 2) Additive Quadratic Model
# First regressor x1 is T x 1 vector of ones, second regressor x2 is the vector 1, 2, ..., T, third regressor x3 is the vector 1, 4, 9, ..., T^2
# The dependent variable is per capital GDP in levels 
x1 = np.empty(T) 
x2 = np.empty(T) 
x3 = np.empty(T)

for t in range(T):
    x1[t] = 1.
    x2[t] = t + 1 
    x3[t] = (t + 1) ** 2

a_add_quad, b_add_quad, c_add_quad = get_regression_coefs(Y, x1, x2, x3)

# Initialise predicted values yhat 
Yhat_add_quad = np.empty(T_all)

# Create loop to compute trend for all years
for t in range(T_all):
    Yhat_add_quad[t] = a_add_quad + b_add_quad * (t + 1) + c_add_quad * (t + 1) ** 2

# Convert into log units
yhat_add_quad = np.log(Yhat_add_quad)

lw = 4

# Plotting the figure
plt.figure()
plt.plot(data['year'],np.log(data['rgdpna_pc']))
plt.plot(data['year'],yhat_add_quad)

#labels
plt.xlabel("year")
plt.ylabel("ln(GDP)")

#title
plt.title("Additive Quadratic: GDP & trend")
plt.legend()

plt.show()

# 3) Exponential Linear Model
# First regressor x1 is T x 1 vector of ones, second regressor x2 is the vector 1, 2, ..., T, third regressor x3 is the vector exp(1), exp(2), ..., exp(T)
# The dependent variable is per capital GDP in levels 
x1 = np.empty(T) 
x2 = np.empty(T) 
x3 = np.empty(T)

for t in range(T):
    x1[t] = 1.
    x2[t] = t + 1 
    x3[t] = np.exp(t + 1)

a_exp_lin, b_exp_lin, c_exp_lin = get_regression_coefs(Y, x1, x2, x3)

Yhat_exp_lin = np.empty(T_all)

for t in range(T_all):
    Yhat_exp_lin[t] = a_exp_lin + b_exp_lin * (t + 1) + c_exp_lin * np.exp(t + 1)

# Initialise predicted values yhat 
yhat_exp_lin = np.log(Yhat_exp_lin)

lw = 4

# Plotting the figure
plt.figure()
plt.plot(data['year'],np.log(data['rgdpna_pc']))
plt.plot(data['year'],yhat_exp_lin)

#labels
plt.xlabel("year")
plt.ylabel("ln(GDP)")

#title
plt.title("Exponential Linear: GDP & trend")
plt.legend()

plt.show()

# 4) Exponential Quadratic Model
# First regressor x1 is T x 1 vector of ones, second regressor x2 is the vector exp(1), exp(2), ..., exp(T) third regressor x3 is the vector 1, 4, 9, ..., T^2
# The dependent variable is per capital GDP in levels 
x1 = np.empty(T) 
x2 = np.empty(T) 
x3 = np.empty(T)

for t in range(T):
    x1[t] = 1.
    x2[t] = np.exp(t + 1)
    x3[t] = (t + 1) ** 2

a_exp_quad, b_exp_quad, c_exp_quad = get_regression_coefs(Y, x1, x2, x3)

# Initialise predicted values yhat 
Yhat_exp_quad = np.empty(T_all)

for t in range(T_all):
    Yhat_exp_quad[t] = a_exp_quad + b_exp_quad * np.exp(t + 1) + c_exp_quad * (t + 1) ** 2

yhat_exp_quad = np.log(Yhat_exp_quad)

lw = 4

# Plotting the figure
plt.figure()
plt.plot(data['year'],np.log(data['rgdpna_pc']))
plt.plot(data['year'],yhat_exp_quad)

#labels
plt.xlabel("year")
plt.ylabel("ln(GDP)")

#title
plt.title("Exponential Quadratic: GDP & trend")
plt.legend()

plt.show()

pd.set_option("display.float_format", "{:.2f}".format)  # set display setting

# Set parameteres
alpha = 0.3

# Extract Data as numpy arrays. 
year = data["year"].to_numpy()  
Y = data["rgdpna"].to_numpy()
K = data["rnna"].to_numpy()
L = data["emp"].to_numpy()

# construct TFP as the Solow residual
tmp = Y / (K**alpha * L ** (1 - alpha))
A = tmp ** (1 / (1 - alpha))

# Add TFP series to our dataframe and inspect the data.
data["TFP (A)"] = A
print(data)

# For convenience we define a function that calculate yearly growth rates using log growth rates
def compute_growth_rate(X):
    return np.log(X[1:] / X[:-1])

dY = compute_growth_rate(Y)
dA = compute_growth_rate(A)
dK = compute_growth_rate(K)
dL = compute_growth_rate(L)

# calculate contributions
contrib_dK = alpha * dK / dY
contrib_dL = (1 - alpha) * dL / dY
contrib_dA = (1 - alpha) * dA / dY


# Display growth rates
print("\t Year by Year Growth Rate \n")
print("\t year \t \t ln(Y_t/Y_{t-1}) ")
print("\t ------- \t ------ ")
for y1, y2, growth in zip(year[:-1], year[1:], dY):
    print(f"\t {y1:.0f}-{y2:.0f} \t {growth:.4f}")

# calculate contributions
contrib_dK = alpha * dK / dY
contrib_dL = (1 - alpha) * dL / dY
contrib_dA = (1 - alpha) * dA / dY

# To inspect the results we convert the numpy arrays to a pandas dataframe holding the contributions
# First we need to concatenate (stack horizontally) the contribution arrays into a matrix
# Note that in Python 1D-arrays are given as row vectors (row major). So stacking arrays horizontally would create a long row vector.
# X[:, None] is a simple way to convert a 1D row vector to a column vector

# Compute yearly intervals of the form "2000-2001"
intvls = np.array([f"{y1}-{y2}" for y1, y2 in zip(year[:-1], year[1:])])

# Fill in the dataframe
df_contribs = pd.DataFrame(
    data=np.hstack(
        (intvls[:, None], contrib_dK[:, None], contrib_dL[:, None], contrib_dA[:, None])
    ),
    columns=["year", "K contribution", "L contribution", "A contribution"],
)

# Specify the data types for specific columns (it would be more efficient to define dataypes already before creating dataframe)
df_contribs = df_contribs.astype(
    {
        "year": "object",
        "K contribution": "float64",
        "L contribution": "float64",
        "A contribution": "float64",
    }
)

# Set index
df_contribs = df_contribs.set_index("year")

# Set display options
pd.set_option("display.float_format", "{:.4f}".format)

# Display contribution table
print("\n\t == year by year contributions ==\n")
print(df_contribs)
print("\n")

avg_contribs = df_contribs.mean()
print(avg_contribs)

print("year \t \t K contrib \t L contrib \t A contrib \t Y growth")
print("------- \t --------- \t --------- \t --------- \t --------")

print(
    f"1970-79 \t {alpha * np.log(K[19]/K[10]) / np.log(Y[19]/Y[10]):.4f}",
    f"\t {(1-alpha) * np.log(L[19]/L[10]) / np.log(Y[19]/Y[10]):.4f}",
    f"\t {(1-alpha) * np.log(A[19]/A[10]) / np.log(Y[19]/Y[10]):.4f}",
    f"\t {np.log(Y[19] / Y[10]):.4f}",
)

print(
    f"1980-89 \t {alpha * np.log(K[29]/K[20]) / np.log(Y[29]/Y[20]):.4f}",
    f"\t {(1-alpha) * np.log(L[29]/L[20]) / np.log(Y[29]/Y[20]):.4f}",
    f"\t {(1-alpha) * np.log(A[29]/A[20]) / np.log(Y[29]/Y[20]):.4f}",
    f"\t {np.log(Y[29] / Y[20]):.4f}",
)

print(
    f"1990-99 \t {alpha * np.log(K[39]/K[30]) / np.log(Y[39]/Y[30]):.4f}",
    f"\t {(1-alpha) * np.log(L[39]/L[30]) / np.log(Y[39]/Y[30]):.4f}",
    f"\t {(1-alpha) * np.log(A[39]/A[30]) / np.log(Y[39]/Y[30]):.4f}",
    f"\t {np.log(Y[39] / Y[30]):.4f}",
)

print(
    f"2000-09 \t {alpha * np.log(K[49]/K[40]) / np.log(Y[49]/Y[40]):.4f}",
    f"\t {(1-alpha) * np.log(L[49]/L[40]) / np.log(Y[49]/Y[40]):.4f}",
    f"\t {(1-alpha) * np.log(A[49]/A[40]) / np.log(Y[49]/Y[40]):.4f}",
    f"\t {np.log(Y[49] / Y[40]):.4f}",
)

print(
    f"2010-19 \t {alpha * np.log(K[59]/K[50]) / np.log(Y[59]/Y[50]):.4f}",
    f"\t {(1-alpha) * np.log(L[59]/L[50]) / np.log(Y[59]/Y[50]):.4f}",
    f"\t {(1-alpha) * np.log(A[59]/A[50]) / np.log(Y[59]/Y[50]):.4f}",
    f"\t {np.log(Y[59] / Y[50]):.4f}",
)

# D: Series for labour productivity 
data["lbrpd"] = data["rgdpna"] / data["emp"]
print(data)