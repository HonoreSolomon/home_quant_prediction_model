from idlelib.debugobj import dispatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.preprocessing as preprocessing
from numpy.f2py.symbolic import number_types
from numpy.ma.core import negative
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error, \
    PredictionErrorDisplay
import seaborn as sns
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./housing.csv"
df = pd.read_csv(DATA_PATH)

#new california housing data
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

df = df.drop(columns = ["ocean_proximity"])

cols_to_check = ["total_rooms", "total_bedrooms", "population"]
df = df[(df[cols_to_check] != 0).all(axis=1)]

# df["rooms_per_household"] = df["total_room"] / df["households"]


X = df.drop("median_house_value", axis=1)
Y = df["median_house_value"]


#feature engineering - usa_housing.csv
# df["HouseAge"] = 2025 - df["YearBuilt"]
# df = df.drop(["YearBuilt"], axis=1)
#
# # df["Price_per_sqft"] = df ["Price"] / df["SquareFeet"]
# #
# # df["Price_per_bedroom"] = df["Price"] / df["Bedrooms"]
#
# df["Age_Lot_Interaction"] = df["HouseAge"] * df["LotSize"]
#
# df["Crime_School_Interaction"] = df["CrimeRate"] * df["SchoolRating"]
# df = df.drop(columns=["CrimeRate", "SchoolRating"])
#
# df["Age_Lot_Interaction"] = df["HouseAge"] * df["LotSize"]
# df = df.drop(columns=["LotSize", "HouseAge"])
#
# df=df.drop(["ZipCode"],axis=1)
#
# X = df.drop(["Price"], axis=1)
# Y = df["Price"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#lin regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, Y_train)
lin_predictions = lr_model.predict(X_test_scaled)
lin_r2 = r2_score(Y_test, lin_predictions)
lin_mse = mean_squared_error(Y_test, lin_predictions)
print (f"R square value: {lin_r2}")
print (f"MSE: {lin_mse}")



rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(X_train_scaled, Y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_r2 = r2_score(Y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(Y_test, rf_predictions))
# rf_mae = mean_absolute_error(Y_test, rf_predictions)
# rf_mape = mean_absolute_percentage_error(Y_test, rf_predictions)

print (f"R square value: {rf_r2}")
print (f"Root Mean squared error: {rf_rmse}")
# print (f"Mean absolute error: {rf_mae}")
# print (f"Mean absolute percentage error: {rf_mape}")



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
# #
# #
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths= 0.1)
plt.show()


def predict_price():
    print("="*50)
    print("🏠 Welcome to HomeQuant Realty Price Predictor 🏠")
    print("Estimate California home values instantly using key property and neighborhood features.")
    print("Please enter the following information. Type Ctrl+C at any time to exit.")
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

        # Feature engineering for ratios
        rooms_per_household = total_rooms / households if households else 0
        bedrooms_per_room = total_bedrooms / total_rooms if total_rooms else 0
        population_per_household = population / households if households else 0

        input_data = {
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'rooms_per_household': [rooms_per_household],
            'bedrooms_per_room': [bedrooms_per_room],
            'population_per_household': [population_per_household]
        }
        input_df = pd.DataFrame(input_data)
        input_df = input_df[X.columns]  # Ensure correct order

        input_scaled = scaler.transform(input_df)
        pred = rf_model.predict(input_scaled)

        print("\n" + "-"*50)
        print(f"🏡 Predicted Median House Value: ${pred[0]:,.2f}")
        print(f"Model R² (accuracy): {rf_r2:.3f}")
        print("-"*50)
        print("Thank you for using HomeQuant Realty Price Predictor!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Invalid input. Please enter numeric values only. Error: {e}")
        print("Tip: Use Ctrl+C to exit the program at any time.")






# def predict_price():
#     print("\n-- HomeQuant Realty: Predict House Price Model --")
#     try:
#         # Gather user input
#         bedrooms = float(input("Enter Bedrooms: "))
#         bathrooms = float(input("Enter Bathrooms: "))
#         square_feet = float(input("Enter Square Feet: "))
#         garage_space = float(input("Enter Garage Spaces: "))
#
#
#         # Year built and lot size are needed for interactions
#         year_built = float(input("Enter Year Built: "))
#         lot_size = float(input("Enter Lot Size: "))
#         crime_rate = float(input("Enter Crime Rate: "))
#         school_rating = float(input("Enter School Rating: "))
#
#         # Feature engineering to match training data
#
#
#
#
#         # house_age = 2025 - year_built
#         # price_per_sqft = 0  # Not used for prediction, only for EDA
#         # price_per_bedroom = 0  # Not used for prediction, only for EDA
#         # age_lot_interaction = house_age * lot_size
#         # crime_school_interaction = crime_rate * school_rating
#
#         # Build the input DataFrame with the correct columns/order
#         input_data = {
#             'Bedrooms': [bedrooms],
#             'Bathrooms': [bathrooms],
#             'SquareFeet': [square_feet],
#             'GarageSpaces': [garage_space],
#             # 'Price_per_sqft': [price_per_sqft],  # Placeholder, not used in prediction
#             # 'Price_per_bedroom': [price_per_bedroom],  # Placeholder, not used in prediction
#             'Age_Lot_Interaction': [age_lot_interaction],
#             'Crime_School_Interaction': [crime_school_interaction]
#         }
#         input_df = pd.DataFrame(input_data)
#
#         # Only keep columns that are in X (model input features)
#         input_df = input_df[X.columns]
#
#         # Scale the input
#         input_scaled = scaler.transform(input_df)
#
#         # Predict using the trained Random Forest model
#         pred = rf_model.predict(input_scaled)
#
#         print(f"\nPredicted House Price: ${pred[0]:,.2f}")
#         print(f"Model R² (accuracy): {rf_r2:.3f}")
#
#     except Exception as e:
#         print(f"Invalid input. Please enter numeric values only. Error: {e}")


# To use:
predict_price()




