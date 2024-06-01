import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

housing = pd.read_csv("./data/housing.csv")

# melakukan one hot encoding untuk data categorical

encoded_neighborhood = pd.get_dummies(housing['neighborhood']).astype(int)

data_encoded = pd.concat([housing, encoded_neighborhood],
                         axis=1).drop(['neighborhood'], axis=1)

# melakukan standarisasi scala data di column tertentu

data_scaled = StandardScaler().fit_transform(
    data_encoded[data_encoded.columns[0:6]])


pd_scaled = pd.DataFrame(data_scaled, columns=data_encoded.columns[0:6])

X = pd.concat([pd_scaled, data_encoded[data_encoded.columns[7:]]], axis=1)
Y = housing['price']

# pembagian dataset pelatihan dan pengujian

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# menggunakan model random forest

clf = RandomForestRegressor(n_estimators=100)

model = clf.fit(X_train, Y_train)

joblib.dump(model, 'home_price_estimator.joblib')

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R^2): {r2}")
