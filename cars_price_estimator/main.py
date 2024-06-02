import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Car_ID,Brand,Model,Year,Kilometers_Driven,Fuel_Type,Transmission,Owner_Type,Mileage,Engine,Power,Seats,Price
cars_data = pd.read_csv("./data/cars.csv")

# pengecekan apakah data ada yang kosong
# print(cars_data.isnull().sum())

# melakukan pre processing input

column_to_encode = ['Brand', "Fuel_Type",
                    "Transmission", 'Owner_Type', "Model"]
column_to_scale = ["Year", "Kilometers_Driven",
                   "Mileage", "Engine", "Power", "Seats"]

encoded_data = pd.get_dummies(cars_data[column_to_encode]).astype('int')

scaler = StandardScaler().fit(cars_data[column_to_scale])
scaled_data = scaler.transform(cars_data[column_to_scale])


X = pd.concat([pd.DataFrame(encoded_data), pd.DataFrame(
    scaled_data, columns=column_to_scale)], axis=1)
Y = cars_data['Price']

# memecah data untuk pelatihan dan pengujian
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)


# melatih model
clf = LinearRegression()

model = clf.fit(X_train, Y_train)
joblib.dump(model, 'cars_price_estimator.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(encoded_data.columns, 'columns.joblib')

# pengujian

score = model.score(X_test, Y_test)

print(f"model telah mempelajari {int(score * 100)}% data")

Y_pred = model.predict(X_test)

r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print(f"R2 Score : {r2}")
print(f"MSE Score : {mse}")
print(f"MAE Score : {mae}")
