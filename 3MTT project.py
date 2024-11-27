#Step 1: IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

#Step 2: IMPORT AND PREPROCESSING DATA
df = pd.read_csv("C:\\Users\\DELL\\Downloads\\archive\\covid_19_clean_complete.csv")

#check the first 5 rows of the data
print(df.head())
print(df.columns)

#describe the data
print(df.describe())

#check for missing value
print(df.isnull().sum())

#STEP 3: HANDLING MISSING VALUES

df['Province/State'].fillna('Unknown', inplace=True)
# Active Case = confirmed - deaths - recovered
df['Active'] = df['Confirmed'] - df[ 'Deaths'] - df['Recovered']

#Transform date column to datetime format

from datetime import datetime as dt

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.normalize()
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

a = df.Date.value_counts().sort_index()
print(f"the first date is: {a.index[0]}")
print(f"the last date is: {a.index[-1]}")

total_cases = df.loc[:, ['Date', 'Confirmed']].groupby('Date').sum().reset_index()
print(total_cases)

#Step 4: FEARTURE ENGINEERING
#load a new dataset with the new features
df_1=pd.read_csv("C:\\Users\\DELL\\Downloads\\archive\\worldometer_data.csv")


# Find the common columns
common_columns = set(df) & set(df_1)


# Print the common columns
print("Common columns:")
for column in common_columns:
    print(column)


# Check if the values in the common columns are the same
for column in common_columns:
    if df[column].equals(df_1[column]):
        print(f"The values in the '{column}' column are the same in both datasets.")
    else:
        print(f"The values in the '{column}' column are different in the two datasets.")


# Check for inconsistencies in country names between the two datasets
inconsistent_countries = set(df['Country/Region']) ^ set(df_1['Country/Region'])
print(inconsistent_countries)


# Merge the datasets
merged_df = pd.merge(df, df_1, on='Country/Region', how='outer', suffixes=('_x', '_y'))

# Print the merged dataset
print(merged_df.head())
print(merged_df.columns)
print(merged_df.shape)


# Calculate daily growth rates
merged_df['daily_growth_rate_cases'] = merged_df.groupby('Country/Region')['Confirmed'].pct_change()
merged_df['daily_growth_rate_deaths'] = merged_df.groupby('Country/Region')['Deaths'].pct_change()

# Calculate mortality ratios
merged_df['mortality_ratio'] = merged_df['Deaths'] / merged_df['Confirmed']

# Calculate cases per Population (assuming 'Population' column exists)
merged_df['cases_per_population'] = merged_df['Confirmed'] / merged_df['Population'] * 100000

# Calculate deaths per Population (assuming 'Population' column exists)
merged_df['deaths_per_population'] = merged_df['Deaths'] / merged_df['Population'] * 100000


# Calculate recovery rate
merged_df['recovery_rate'] = merged_df['Recovered'] / merged_df['Confirmed']

# Calculate active case rate
merged_df['active_case_rate'] = merged_df['Active'] / merged_df['Confirmed']

#STEP 5 EXPLORATORY DATA ANALYSIS
# Line plot: COVID-19 cases over time
plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Confirmed', data=df)
plt.title('COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()

# Bar chart: COVID-19 cases by country
plt.figure(figsize=(10,6))
sns.barplot(x='Country/Region', y='Confirmed', data=df)
plt.title('COVID-19 Cases by Country')
plt.xlabel('Country')
plt.ylabel('Cases')
plt.show()

# Scatter plot: Relationship between cases and deaths
plt.figure(figsize=(10,6))
sns.scatterplot(x='Confirmed', y='Deaths', data=df)
plt.title('Relationship Between Cases and Deaths')
plt.xlabel('Cases')
plt.ylabel('Deaths')
plt.show()

# Heatmap: Correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(df[['Confirmed', 'Deaths', 'Recovered', 'Active']].corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Box plot: Distribution of cases by WHO region
plt.figure(figsize=(10,6))
sns.boxplot(x='WHO Region', y='Confirmed', data=df)
plt.title('Distribution of Cases by WHO Region')
plt.xlabel('WHO Region')
plt.ylabel('Cases')
plt.show()

# Identify top 10 countries with highest number of cases
top_10_countries = df.groupby('Country/Region')['Confirmed'].sum().sort_values(ascending=False).head(10)
print(top_10_countries)

# Identify top 10 countries with highest number of deaths
top_10_deaths = df.groupby('Country/Region')['Deaths'].sum().sort_values(ascending=False).head(10)
print(top_10_deaths)

plt.figure(figsize= (14,5))


ax = sns.pointplot(x=total_cases['Date'],
                   y=total_cases['Confirmed'],
                   color='r')
ax.set(xlabel='Dates', ylabel='Total cases')

plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=12)

plt.xlabel('Dates', fontsize=14)
plt.ylabel('Total cases', fontsize=14)
plt.title('Worldwide Confirmed Cases Over Time', fontsize=20)

with sns.axes_style('white'):

    g = sns.relplot(
        x="Date",
        y="Deaths",
        kind="line",
        data=df
)
    g.fig.autofmt_xdate()
    g.set_xticklabels(step=10)
    plt.title("Covid-19 Deaths, Year:2020", fontsize=16)

    top = df.loc[df['Date'] == df['Date'].max()]

top_casualities = top.groupby('Country/Region')['Confirmed'].sum().sort_values(ascending =False).head(10).reset_index()

print(top_casualities)

sns.set(style="darkgrid")
plt.figure(figsize= (15,10))

ax = sns.barplot(x=top_casualities['Confirmed'],
                 y=top_casualities['Country/Region'])

for i, (value, name) in enumerate(zip(top_casualities['Confirmed'], top_casualities['Country/Region'])):
    ax.text(value, i-.05, f'{value:,.0f}', size=10, ha='left', va='center')
ax.set(xlabel='Total cases', ylabel='Country/Region')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Total cases', fontsize=30)
plt.ylabel('Country', fontsize=30)
plt.title('Top 10 countries having most confirmed cases', fontsize=20)

US_data = df.loc[df['Country/Region'] == 'Brazil', ['Date', 'Recovered', 'Deaths', 'Confirmed', 'Active']]
US_data.head()

US_data = US_data.groupby('Date').sum().reset_index()
US_data = US_data.iloc[33:]
US_data.head()

plt.figure(figsize=(15,5))
sns.set_color_codes("pastel")

sns.pointplot(x=US_data.index, y=US_data['Active'], color='b')

plt.xlabel('No. of Days', fontsize=15)
plt.ylabel('Active cases', fontsize=15)
plt.title("US's Active Cases Over Time", fontsize=25)

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 5))

# Plot the total cases
sns.set_color_codes("pastel")
sns.barplot(x=US_data.index, y=US_data.Active + US_data.Recovered + US_data.Deaths,
            label="Active", color="b")

# Plot the recovered
sns.set_color_codes("muted")
sns.barplot(x=US_data.index, y=US_data.Recovered + US_data.Deaths,
            label="Recovered", color="g")

# Plot the Deaths
sns.set_color_codes("dark")
sns.barplot(x=US_data.index ,y=US_data.Deaths,
            label="Deaths", color="r")

plt.xlabel('No. of Days', fontsize=14)
plt.ylabel('No. of cases', fontsize=15)

# Add a legend and informative axis label
ax.legend(ncol=2, loc="upper left", frameon=True)
sns.despine(top=True)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#step 6: Train the model

# Prepare data (convert date to numerical format)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(pd.Timestamp.toordinal)  # Convert date to ordinal format (numeric)

# Define features (X) and target (y)
X = df[['Date']]  # Feature: Date
y = df['Confirmed']  # Target: Confirmed cases

# Split data into training and test sets (80% training, 20% testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model on the training data
dt_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt_model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Plot the original data and the predicted values
plt.figure(figsize=(15,5))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')  # Actual data points
plt.scatter(X_test, y_pred, color='red', label='Predicted Data')  # Predicted data points
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title('Actual vs Predicted Confirmed Cases using Decision Tree')
plt.legend()
plt.show()

# Time-Series Modeling (ARIMA)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
model = ARIMA(train_df['Confirmed'], order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test_df))
mse = mean_squared_error(test_df['Confirmed'], forecast)
print(f'RMSE: {np.sqrt(mse)}')

# Classification Modeling (Random Forest)
X = df[['Confirmed', 'Deaths', 'Recovered', 'Active']]
y = np.where(df['Confirmed'] > df['Recovered'], 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test,y_pred)}')
