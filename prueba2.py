import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset (asegúrate de cargar tu archivo CSV en Kaggle)
# Reemplaza 'laptops.csv' con el nombre de tu archivo
data = pd.read_csv('laptop_data.csv')

# Inspeccionar los datos
print(data.head())
print(data.info())

# Preprocesamiento de datos
# Eliminar filas con valores nulos
data = data.dropna()

# Codificar variables categóricas (como marca, tipo de procesador, etc.)
label_encoders = {}
categorical_columns = ['brand', 'processor_type']  # Reemplaza con las columnas categóricas de tu dataset

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Seleccionar las características (X) y la variable objetivo (y)
# Reemplaza 'price' con el nombre de la columna que contiene los precios
X = data[['processor_type', 'ram_size', 'storage_capacity', 'brand']]  # Ajusta según las columnas de tu dataset
y = data['price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE del modelo: {rmse}")

# Opcional: Guardar el modelo entrenado
import joblib
joblib.dump(model, 'laptop_price_model.pkl')

# Opcional: Predicción con nuevos datos
new_data = pd.DataFrame({'processor_type': [1], 'ram_size': [16], 'storage_capacity': [512], 'brand': [2]})
prediction = model.predict(new_data)
print(f"Predicción del precio: {prediction}")