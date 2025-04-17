import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

# Configuración inicial
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

## =============================================
## 1. Carga y Preprocesamiento de Datos
## =============================================

def load_and_preprocess(filepath):
    """Carga y preprocesa los datos de laptops"""
    print("Cargando y preprocesando datos...")
    df = pd.read_csv(filepath, index_col=0)
    
    # Conversiones básicas
    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)
    
    # Función para convertir a GB
    def to_gb(text):
        if pd.isna(text): return 0
        text = str(text).lower()
        if 'tb' in text: return float(text.split('tb')[0].strip()) * 1000
        elif 'gb' in text: return float(text.split('gb')[0].strip())
        return 0
    
    # Procesamiento de memoria mejorado
    df['Memory'] = df['Memory'].str.lower()
    
    def get_ssd(memory_str):
        return sum(to_gb(part) for part in str(memory_str).split('+') 
               if any(s in part for s in ['ssd', 'flash', 'solid state']))
    
    def get_hdd(memory_str):
        return sum(to_gb(part) for part in str(memory_str).split('+') 
               if 'hdd' in part)
    
    df['SSD'] = df['Memory'].apply(get_ssd)
    df['HDD'] = df['Memory'].apply(get_hdd)
    df['Total_Storage'] = df['SSD'] + df['HDD']
    df['Storage_Type'] = np.where(df['HDD'] > 0, 
                                np.where(df['SSD'] > 0, 'Hybrid', 'HDD Only'), 
                                'SSD Only')
    
    # Características de pantalla
    df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', na=False).astype(int)
    df['IPS'] = df['ScreenResolution'].str.contains('IPS', na=False).astype(int)
    
    try:
        res = df['ScreenResolution'].str.extract(r'(?P<x_res>\d+)\s*x\s*(?P<y_res>\d+)')
        df['ppi'] = ((res['x_res'].astype(float)**2 + res['y_res'].astype(float)**2)**0.5 / df['Inches']).round(2)
        df['Aspect_Ratio'] = res['x_res'].astype(float) / res['y_res'].astype(float)
    except:
        df['ppi'] = df['Inches'].apply(lambda x: 220 if x > 12 else 300)
        df['Aspect_Ratio'] = 16/9
    
    # Procesamiento de CPU
    df['Cpu_Cores'] = df['Cpu'].str.extract(r'(\d+)Core').fillna(1).astype(int)
    df['Cpu brand'] = df['Cpu'].apply(
        lambda x: 'Intel Core' if 'Intel Core' in x else 
                 ('Intel' if 'Intel' in x else 'AMD'))
    
    # Procesamiento de GPU
    df['Gpu brand'] = df['Gpu'].str.split().str[0]
    df['Gpu_Memory'] = df['Gpu'].str.extract(r'(\d+)GB').fillna(0).astype(int)
    
    # Codificación inteligente de categorías
    def smart_encode(col, threshold=0.05):
        counts = df[col].value_counts(normalize=True)
        return df[col].apply(lambda x: x if counts.get(x, 0) > threshold else 'Other')
    
    df['Company'] = smart_encode('Company')
    df['TypeName'] = smart_encode('TypeName')
    df['Gpu brand'] = smart_encode('Gpu brand', threshold=0.03)
    
    # Sistema operativo
    df['os'] = df['OpSys'].apply(
        lambda x: 'Windows' if 'Windows' in str(x) else 
                 ('Mac' if 'Mac' in str(x) or 'macOS' in str(x) else 
                 ('Linux' if 'Linux' in str(x) else 'Other')))
    
    # Columnas a eliminar (solo las que existen)
    cols_to_drop = ['Cpu', 'Memory', 'ScreenResolution', 'OpSys', 'Gpu']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    # Eliminar columnas no necesarias
    df.drop(columns=cols_to_drop, inplace=True)
    
    print("Preprocesamiento completado.")
    return df

## =============================================
## 2. Preparación de Datos para Modelado
## =============================================

print("\nPreparando datos para modelado...")
df = load_and_preprocess("laptop_data.csv")

# Definir características y objetivo
X = df.drop(columns=['Price'])
y = np.log1p(df['Price'])  # Transformación logarítmica

# Dividir datos (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identificar columnas numéricas y categóricas
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocesamiento
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

## =============================================
## 3. Configuración y Entrenamiento de Modelos
## =============================================

print("\nConfigurando y entrenando modelos...")

# Configuración optimizada para GridSearch
param_grids = {
    'XGB': {
        'model__n_estimators': [300, 500],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [5, 6],
        'model__subsample': [0.8, 0.9],
        'model__colsample_bytree': [0.8, 0.9],
        'model__reg_alpha': [0, 0.1],
        'model__reg_lambda': [0, 0.1]
    },
    'LGBM': {
        'model__n_estimators': [300, 500],
        'model__learning_rate': [0.05, 0.1],
        'model__num_leaves': [31, 63],
        'model__max_depth': [5, 7],
        'model__feature_fraction': [0.8, 0.9],
        'model__min_child_samples': [10, 20]
    }
}

def train_and_evaluate(model_type, params):
    """Entrena y evalúa un modelo con GridSearchCV"""
    print(f"\nEntrenando modelo {model_type}...")
    
    if model_type == 'XGB':
        model = XGBRegressor(random_state=42, n_jobs=-1)
    else:
        model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    search = GridSearchCV(
        pipe, 
        params, 
        cv=5, 
        scoring='neg_root_mean_squared_error',
        n_jobs=4,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    # Evaluación
    train_pred = search.predict(X_train)
    test_pred = search.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"\nResultados {model_type}:")
    print(f"Mejores parámetros: {search.best_params_}")
    print(f"RMSE entrenamiento: {train_rmse:.5f}")
    print(f"RMSE prueba: {test_rmse:.5f}")
    
    return search.best_estimator_, test_rmse

# Entrenar modelos individuales
best_models = {}
for model_type, params in param_grids.items():
    best_model, rmse = train_and_evaluate(model_type, params)
    best_models[model_type] = (best_model, rmse)

## =============================================
## 4. Ensamblado de Modelos (Si es necesario)
## =============================================

# Seleccionar el mejor modelo individual
best_single_model, best_rmse = min(best_models.values(), key=lambda x: x[1])

if best_rmse > 0.07:
    print("\nProbando modelo ensamblado para mejorar resultados...")
    
    # Crear ensemble con los mejores modelos individuales
    xgb_model = best_models['XGB'][0].named_steps['model']
    lgbm_model = best_models['LGBM'][0].named_steps['model']
    
    ensemble = VotingRegressor([
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ], weights=[1, 1.5])
    
    ensemble_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ])
    
    ensemble_pipe.fit(X_train, y_train)
    
    # Evaluar ensemble
    train_pred = ensemble_pipe.predict(X_train)
    test_pred = ensemble_pipe.predict(X_test)
    
    ensemble_train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"\nResultados Ensemble:")
    print(f"RMSE entrenamiento: {ensemble_train_rmse:.5f}")
    print(f"RMSE prueba: {ensemble_test_rmse:.5f}")
    
    if ensemble_test_rmse < best_rmse:
        best_single_model = ensemble_pipe
        best_rmse = ensemble_test_rmse

## =============================================
## 5. Evaluación Final y Guardado del Modelo
## =============================================

print("\n" + "="*50)
print("Resultado Final del Mejor Modelo")
print("="*50)

# Entrenar el mejor modelo con todos los datos
if best_rmse <= 0.07:
    print("\n¡Objetivo alcanzado! RMSE < 0.07")
    
    # Reentrenar con todos los datos
    if 'ensemble' in best_single_model.named_steps:
        final_model = Pipeline([
            ('preprocessor', preprocessor),
            ('ensemble', VotingRegressor([
                ('xgb', XGBRegressor(**best_models['XGB'][0].named_steps['model'].get_params())),
                ('lgbm', LGBMRegressor(**best_models['LGBM'][0].named_steps['model'].get_params()))
            ], weights=[1, 1.5]))
        ])
    else:
        final_model = best_single_model
    
    final_model.fit(X, y)
    
    # Guardar modelo
    joblib.dump(final_model, 'best_laptop_price_model.pkl')
    print("\nModelo guardado como 'best_laptop_price_model.pkl'")
    
    # Importancia de características
    if 'ensemble' not in final_model.named_steps:
        feature_importances = final_model.named_steps['model'].feature_importances_
    else:
        xgb_imp = final_model.named_steps['ensemble'].estimators_[0].feature_importances_
        lgbm_imp = final_model.named_steps['ensemble'].estimators_[1].feature_importances_
        feature_importances = (xgb_imp * 1 + lgbm_imp * 1.5) / 2.5
    
    # Obtener nombres de características
    num_features = num_cols
    cat_features = final_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_features = np.concatenate([num_features, cat_features])
    
    # Visualizar importancia
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Top 20 Características Más Importantes')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("\nGráfico de importancia de características guardado como 'feature_importance.png'")
    
else:
    print("\nRecomendaciones para mejorar el modelo:")
    print("1. Recolectar más datos o características adicionales")
    print("2. Experimentar con técnicas de feature engineering más avanzadas")
    print("3. Probar modelos de stacking más sofisticados")
    print("4. Realizar una búsqueda más exhaustiva de hiperparámetros")
    print("5. Considerar transformaciones alternativas de la variable objetivo")

print("\nProceso completado exitosamente!")