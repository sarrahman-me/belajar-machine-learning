import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Data baru untuk prediksi (tanpa kolom 'price')
new_data = {
    'size': [1.678021],
    'bedrooms': [1.416090],
    'bathrooms': [-0.004766],
    'age': [1.687082],
    'garage': [-0.465320],
    'school_rating': [-1.217215],
    'Central': [1],
    'East': [0],
    'North': [0],
    'South': [0],
    'West': [0]
}


# Buat DataFrame dari data baru
df_new = pd.DataFrame(new_data)

# Load model yang telah dilatih sebelumnya
loaded_model = joblib.load('home_price_estimator.joblib')

# Lakukan prediksi harga rumah pada data baru
predicted_price = loaded_model.predict(df_new)

# Cetak hasil prediksi
print("Hasil Prediksi Harga Rumah:")
print("Prediksi Harga Rumah:", predicted_price[0])
