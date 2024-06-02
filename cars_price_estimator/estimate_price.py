import joblib
import pandas as pd

model = joblib.load('./cars_price_estimator.joblib')
scaler = joblib.load('./scaler.joblib')
encoded_columns = joblib.load('columns.joblib')

data = pd.DataFrame({
    "id": 1,
    "Brand": "Mercedes",
    "Model": "E-Class",
    "Year": 2017,
    "Kilometers_Driven": 30000,
    "Fuel_Type": "Diesel",
    "Transmission": "Automatic",
    "Owner_Type": "First",
    "Mileage": 16,
    "Engine":  1950,
    "Power": 191,
    "Seats": 5,
}, index=['id'])

data.drop(columns=['id'], inplace=True)

column_to_encode = ['Brand', "Fuel_Type",
                    "Transmission", 'Owner_Type', "Model"]
column_to_scale = ["Year", "Kilometers_Driven",
                   "Mileage", "Engine", "Power", "Seats"]


scaled_data = scaler.transform(data[column_to_scale])

encoded_data = pd.get_dummies(data[column_to_encode]).astype('int')

# Ensure all necessary columns are present
for col in encoded_columns:
    if col not in encoded_data.columns:
        encoded_data[col] = 0

# Reorder columns to match the training data
encoded_data = encoded_data[encoded_columns]

# Perform scaling
scaled_data = scaler.transform(data[column_to_scale])

# Combine encoded and scaled data
X = pd.concat([encoded_data.reset_index(drop=True), pd.DataFrame(
    scaled_data, columns=column_to_scale)], axis=1)

predicted = model.predict(X)

print('Harga mobil adalah: ')
print('{:20,.2f}'.format(predicted[0]))
