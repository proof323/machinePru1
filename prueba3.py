import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt, log

# Cargar datos
df = pd.read_csv("laptop_data.csv", index_col=0)

# Inspect the data for NaN values
print("Dataset info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# ====================================================
# 1. Preprocesamiento de datos
# ====================================================

# Limpiar columna Ram
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

# Limpiar columna Weight
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

# Función corregida para extraer almacenamiento
def extract_storage(cell):
    # Handle NaN values
    if pd.isna(cell):
        return 0
    
    total_gb = 0
    parts = cell.split("+")
    
    for part in parts:
        part = part.strip()
        if 'TB' in part:
            num = part.split('TB')[0].strip()
            if num.replace('.', '', 1).isdigit():
                total_gb += float(num) * 1000
        elif 'GB' in part:
            num = part.split('GB')[0].strip()
            if num.replace('.', '', 1).isdigit():
                total_gb += float(num)
    
    return int(total_gb)

df["Storage_GB"] = df["Memory"].apply(extract_storage)

# Extraer resolución de pantalla corregida
def extract_resolution(res_str):
    if pd.isna(res_str):
        return 0
    
    # Handle non-string values
    if not isinstance(res_str, str):
        return 0
        
    # Buscar patrones comunes
    match = pd.Series(res_str).str.extract(r'(\d+)\s*x\s*(\d+)')
    if not match.empty and match.iloc[0].notnull().all():
        return int(match.iloc[0, 0]) * int(match.iloc[0, 1])
    
    # Manejar casos especiales (4K/UHD)
    if '4K' in res_str or 'UHD' in res_str:
        return 3840 * 2160
    if 'QHD' in res_str:
        return 2560 * 1440
    
    return 0

df["Resolution"] = df["ScreenResolution"].apply(extract_resolution)

# Extraer características del CPU
# Handle NaN values in CPU Brand
df["Cpu_Brand"] = df["Cpu"].apply(lambda x: x.split()[0] if pd.notna(x) else "Unknown")

# Extract CPU speed and handle NaN values
cpu_speed = df["Cpu"].str.extract(r'(\d+\.\d+)GHz')
df["Cpu_Speed"] = cpu_speed[0].astype(float, errors='ignore')
df["Cpu_Speed"] = df["Cpu_Speed"].fillna(df["Cpu_Speed"].median())

# Extract CPU generation
def extract_cpu_gen(cpu_str):
    if pd.isna(cpu_str):
        return -1  # Default value for NaN
    
    # Extract Intel generation (e.g., i7-8550U -> 8)
    intel_match = re.search(r'i\d-(\d{1,2})\d{3}[A-Z]?', cpu_str)
    if intel_match:
        return int(intel_match.group(1))
    
    # Extract AMD generation (roughly estimated)
    amd_match = re.search(r'AMD\s+(\d)', cpu_str)
    if amd_match:
        return int(amd_match.group(1))
    
    return -1  # Default for unknown pattern

df["Cpu_Gen"] = df["Cpu"].apply(extract_cpu_gen)

# Extraer características del GPU
# Handle NaN values in GPU Brand
df["Gpu_Brand"] = df["Gpu"].apply(lambda x: x.split()[0] if pd.notna(x) else "Unknown")

# ====================================================
# 2. Definir variables y dividir datos
# ====================================================

# Create additional engineered features
df["Ram_Storage_Ratio"] = df["Ram"] / (df["Storage_GB"] + 1)  # Adding 1 to avoid division by zero
df["Screen_Density"] = df["Resolution"] / (df["Inches"] ** 2)
df["Ram_Weight_Ratio"] = df["Ram"] / df["Weight"]
df["Performance_Score"] = df["Cpu_Speed"] * df["Ram"] * (df["Cpu_Gen"] + 5) / 10

categorical_cols = ["Company", "TypeName", "Cpu_Brand", "Gpu_Brand"]
numerical_cols = [
    "Inches", "Ram", "Weight", "Storage_GB", "Resolution", "Cpu_Speed", 
    "Cpu_Gen", "Ram_Storage_Ratio", "Screen_Density", "Ram_Weight_Ratio", 
    "Performance_Score"
]

features = [
    "Company", "TypeName", "Inches", "Ram", "Weight",
    "Resolution", "Cpu_Brand", "Cpu_Speed", "Gpu_Brand", "Storage_GB",
    "Cpu_Gen", "Ram_Storage_Ratio", "Screen_Density", "Ram_Weight_Ratio",
    "Performance_Score"
]

X = df[features]

# Apply log transformation to price to create normalized target variable
df["Log_Price"] = np.log1p(df["Price"])
y_raw = df["Price"]
y = df["Log_Price"]

# Plot the price distribution before and after transformation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df["Price"], bins=50)
plt.title("Original Price Distribution")
plt.xlabel("Price")

plt.subplot(1, 2, 2)
plt.hist(df["Log_Price"], bins=50)
plt.title("Log-Transformed Price Distribution")
plt.xlabel("Log(Price)")

plt.tight_layout()
plt.savefig("price_transformation.png")
plt.close()

# Fill remaining NaN values in numerical columns with median
for col in numerical_cols:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())
        
# Fill remaining NaN values in categorical columns with mode
for col in categorical_cols:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mode()[0])

# Split the data, keeping both raw and log-transformed targets aligned
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42
)

# Create preprocessing pipeline with explicit NaN handling
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', PowerTransformer(method='yeo-johnson'))  # Better for skewed data
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Try both Random Forest and Gradient Boosting
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=500, random_state=42))
])

gb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=300, random_state=42))
])

# Use Gradient Boosting as our primary model as it often performs better on regression tasks
pipeline = gb_pipeline

# ====================================================
# 4. Optimización de hiperparámetros
# ====================================================

param_grid = {
    "regressor__n_estimators": [200, 300],
    "regressor__max_depth": [5, 7, 9],
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__subsample": [0.8, 1.0],
    "regressor__min_samples_split": [2, 4]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

print("Fitting model to log-transformed prices...")
grid_search.fit(X_train, y_train)

# ====================================================
# 5. Evaluación del modelo
# ====================================================

best_model = grid_search.best_estimator_

# Make predictions on log-transformed scale
log_y_pred = best_model.predict(X_test)

# Calculate RMSE on log scale
log_rmse = np.sqrt(mean_squared_error(y_test, log_y_pred))

# Transform predictions back to original scale
y_pred_raw = np.expm1(log_y_pred)

# Calculate RMSE on original scale
raw_rmse = np.sqrt(mean_squared_error(y_raw_test, y_pred_raw))

# Calculate normalized RMSE (divide by range or mean)
price_range = y_raw_test.max() - y_raw_test.min()
normalized_rmse = raw_rmse / price_range

# Calculate R² score on both scales
log_r2 = r2_score(y_test, log_y_pred)
raw_r2 = r2_score(y_raw_test, y_pred_raw)

print(f"\n{'='*50}")
print(f"Log-scale RMSE: {log_rmse:.4f}")
print(f"Original scale RMSE: {raw_rmse:.4f}")
print(f"Normalized RMSE: {normalized_rmse:.4f}")
print(f"Log-scale R²: {log_r2:.4f}")
print(f"Original scale R²: {raw_r2:.4f}")
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"{'='*50}\n")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_raw_test, y_pred_raw, alpha=0.5)
plt.plot([y_raw_test.min(), y_raw_test.max()], [y_raw_test.min(), y_raw_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Laptop Prices')
plt.savefig("price_predictions.png")
plt.close()

# ====================================================
# 6. Generar predicciones para Kaggle
# ====================================================

# Cargar datos de test
# test_df = pd.read_csv("test.csv")

# # Aplicar mismo preprocesamiento
# test_df["Ram"] = test_df["Ram"].str.replace("GB", "").astype(int)
# test_df["Weight"] = test_df["Weight"].str.replace("kg", "").astype(float)
# test_df["Storage_GB"] = test_df["Memory"].apply(extract_storage)
# test_df["Resolution"] = test_df["ScreenResolution"].apply(extract_resolution)
# test_df["Cpu_Brand"] = test_df["Cpu"].apply(lambda x: x.split()[0])
# test_df["Cpu_Speed"] = test_df["Cpu"].str.extract(r'(\d+\.\d+)GHz').astype(float)
# test_df["Gpu_Brand"] = test_df["Gpu"].apply(lambda x: x.split()[0])

# # Hacer predicciones
# final_predictions = best_model.predict(test_df[features])

# # Generar archivo de submission
# submission = pd.DataFrame({
#     "id": test_df["id"],
#     "Price": final_predictions
# })

# submission.to_csv("submission.csv", index=False)
# print("Archivo submission.csv generado exitosamente!")

# ====================================================
# 7. Mejoras recomendadas (Implementar según necesidad)
# ====================================================
"""
1. Ingeniería de características adicional:
   - Extraer generación del CPU (ej: i5-1035G1 -> 10)
   - Calcular relación rendimiento/precio por marca
   - Agregar interacción entre RAM y almacenamiento

2. Probár modelos alternativos:
   - XGBoost
   - LightGBM
   - Redes Neuronales

3. Optimización avanzada:
   - Usar Optuna para búsqueda de hiperparámetros
   - Ensamblaje de modelos
   - Stacking de regresores

4. Tratamiento de outliers:
   - Análisis de valores extremos en precios
   - Transformaciones logarítmicas
"""