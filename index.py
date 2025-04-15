import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# -------------------------------
# Carga y preprocesamiento de datos
# -------------------------------
df = pd.read_csv('laptop_data.csv', index_col=0)

# Limpieza básica
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# Función mejorada para procesar 'Memory'
def split_memory(text):
    text = text.replace('Flash Storage', 'SSD').replace('GB', '').replace('TB', '*1000')
    parts = text.split('+')
    total_amount = 0
    types = []
    
    for part in parts:
        part = part.strip()
        amount = ''.join(filter(lambda x: x.isdigit() or x == '*', part))
        amount = eval(amount.replace('*', '')) if '*' in amount else float(amount)
        type_ = ''.join(filter(str.isalpha, part)).strip()
        
        total_amount += amount
        types.append(type_ if type_ else 'Unknown')
    
    return total_amount, '_'.join(sorted(set(types)))

df[['Memory_Amount', 'Memory_Type']] = df['Memory'].apply(split_memory).apply(pd.Series)

# Procesamiento de CPU
df['Cpu_Brand'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[:2]))

# -------------------------------
# División de datos
# -------------------------------
X = df[['Company', 'TypeName', 'Inches', 'Cpu_Brand', 'Ram', 
        'Memory_Amount', 'Memory_Type', 'Gpu', 'OpSys', 'Weight']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Pipeline de preprocesamiento
# -------------------------------
categorical_features = ['Company', 'TypeName', 'Cpu_Brand', 
                        'Memory_Type', 'Gpu', 'OpSys']
numeric_features = ['Inches', 'Ram', 'Memory_Amount', 'Weight']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# -------------------------------
# Modelo con optimización de hiperparámetros
# -------------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 15, 25],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# -------------------------------
# Evaluación del modelo
# -------------------------------
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n📊 RMSE optimizado: {rmse:.4f}")
print(f"Mejores parámetros: {grid_search.best_params_}")

# -------------------------------
# Interfaz de predicción con validación
# -------------------------------
def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options:
            return user_input
        print(f"Opción inválida. Valores permitidos: {', '.join(valid_options)}")

print("\n----- PREDICTOR DE PRECIO DE PORTÁTILES -----")
try:
    company = get_valid_input(f"Marca ({'/'.join(X['Company'].unique())}): ", X['Company'].unique())
    typename = get_valid_input(f"Tipo ({'/'.join(X['TypeName'].unique())}): ", X['TypeName'].unique())
    inches = float(input("Tamaño de pantalla (pulgadas): "))
    cpu_brand = get_valid_input(f"Procesador ({'/'.join(X['Cpu_Brand'].unique())}): ", X['Cpu_Brand'].unique())
    ram = int(input("RAM (GB): "))
    memory_amount = float(input("Almacenamiento total (GB): "))
    memory_type = get_valid_input(f"Tipo de almacenamiento ({'/'.join(X['Memory_Type'].unique())}): ", X['Memory_Type'].unique())
    gpu = get_valid_input(f"Tarjeta gráfica ({'/'.join(X['Gpu'].unique())}): ", X['Gpu'].unique())
    os = get_valid_input(f"Sistema operativo ({'/'.join(X['OpSys'].unique())}): ", X['OpSys'].unique())
    weight = float(input("Peso (kg): "))

    input_data = pd.DataFrame([{
        'Company': company,
        'TypeName': typename,
        'Inches': inches,
        'Cpu_Brand': cpu_brand,
        'Ram': ram,
        'Memory_Amount': memory_amount,
        'Memory_Type': memory_type,
        'Gpu': gpu,
        'OpSys': os,
        'Weight': weight
    }])

    prediction = best_model.predict(input_data)[0]
    print(f"\n💰 Precio estimado: ${prediction:,.2f}")

except ValueError as e:
    print(f"\n❌ Error: {str(e)} - Ingrese valores numéricos válidos")
except Exception as e:
    print(f"\n❌ Error inesperado: {str(e)}")
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session