import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./housing.csv"
df = pd.read_csv(DATA_PATH)

#new california housing data
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

df = df.drop(columns = ["ocean_proximity"])

cols_to_check = ["total_rooms", "total_bedrooms", "population"]
df = df[(df[cols_to_check] != 0).all(axis=1)]




X = df.drop("median_house_value", axis=1)
Y = df["median_house_value"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#lin regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, Y_train)

lin_predictions = lr_model.predict(X_test_scaled)

lin_r2 = r2_score(Y_test, lin_predictions)
lin_rmse = np.sqrt(mean_squared_error(Y_test, lin_predictions))
print("Linear Regression model metrics: ")
print ("-" * 50)
print (f"R square value: {lin_r2:.3f}")
print (f"Root Mean squared error of: ${lin_rmse:,.2f}")



rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(X_train_scaled, Y_train)

rf_predictions = rf_model.predict(X_test_scaled)

rf_r2 = r2_score(Y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(Y_test, rf_predictions))

print ("Random Forest Regressor model metrics: ")
print ("-" * 50)
print (f"R square value: {rf_r2:.3f}")
print (f"Root Mean squared error of: ${rf_rmse:,.2f}")



plt.figure(figsize=(14,8))
sns.scatterplot(x="latitude", y="longitude", data = df, hue = "median_house_value", palette= "coolwarm")
plt.show()
#
sns.scatterplot(x=Y_test, y=rf_predictions)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths= 0.1)
plt.show()

sns.displot(df["median_house_value"])
plt.show()



def predict_price():
    print("="*50)
    print("Welcome to HomeQuant Realty Price Predictor")
    print("Estimate California home values instantly using key property and neighborhood features.")
    print("="*50)
    try:
        longitude = float(input("Longitude (e.g., -122.23): "))
        latitude = float(input("Latitude (e.g., 37.88): "))
        housing_median_age = float(input("Median age of homes in area (years, e.g., 41): "))
        total_rooms = float(input("Total rooms in the block group (e.g., 880): "))
        total_bedrooms = float(input("Total bedrooms in the block group (e.g., 129): "))
        population = float(input("Population of the block group (e.g., 322): "))
        households = float(input("Number of households (e.g., 126): "))
        median_income = float(input("Median income (in $10,000s, e.g., 8.32 for $83,200): "))


        input_data = {
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
        }
        input_df = pd.DataFrame(input_data)
        input_df = input_df[X.columns]

        input_scaled = scaler.transform(input_df)
        pred = rf_model.predict(input_scaled)

        print("\n" + "-"*50)
        print(f"Predicted Median House Value: ${pred[0]:,.2f}")
        print(f"Estimate has an error range of: ${rf_rmse:.3f}")
        print("-"*50)
        print("Thank you for using HomeQuant Realty Price Predictor!")
        print("="*50)
    except Exception as e:
        print(f"\n Invalid input. Please enter numeric values only. Error: {e}")


predict_price()




