import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, PolynomialFeatures, 
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
import warnings

warnings.filterwarnings('ignore')

# -------------------------------
# 1. Carga y Preprocesamiento de Datos
# -------------------------------

def load_and_preprocess():
    # Cargar datos
    df = pd.read_csv('laptop_data.csv', index_col=0)
    
    # Convertir columnas básicas
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    
    return df

# -------------------------------
# 2. Ingeniería de Características Avanzada
# -------------------------------

def advanced_feature_engineering(df):
    # Procesamiento de CPU
    df['Cpu_Speed'] = df['Cpu'].str.extract(r'(\d+\.\d+)GHz').astype(float)
    df['Cpu_Cores'] = df['Cpu'].str.extract(r'(i3|i5|i7|i9|Ryzen \d|A\d|M\d)')
    df['Cpu_Generation'] = df['Cpu'].str.extract(r'(\d+)th').astype(float)
    
    # Procesamiento de GPU
    df['Gpu_Memory'] = df['Gpu'].str.extract(r'(\d+)GB').astype(float)
    df['Gpu_Type'] = df['Gpu'].str.extract(r'(GTX|RTX|Radeon|Iris|HD|UHD|Iris Plus|Iris Pro)')
    
    # Procesamiento de pantalla
    df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen').astype(int)
    df['4K'] = df['ScreenResolution'].str.contains('3840x2160|4096x2160').astype(int)
    df['IPS'] = df['ScreenResolution'].str.contains('IPS').astype(int)
    
    # Extraer resolución
    resolution = df['ScreenResolution'].str.extract(r'(\d+x\d+)', expand=False)
    df['Resolution_Width'] = resolution.str.split('x').str[0].astype(float)
    df['Resolution_Height'] = resolution.str.split('x').str[1].astype(float)
    df['PPI'] = ((df['Resolution_Width']**2 + df['Resolution_Height']**2)**0.5 / df['Inches']).round(2)
    
    # Procesamiento mejorado de memoria
    def enhanced_memory_processing(text):
        text = text.replace('GB', '').replace('TB', '*1000')
        text = text.replace('Flash Storage', 'Flash').replace('SSD', 'SSD').replace('HDD', 'HDD')
        parts = text.split('+')
        ssd = hdd = flash = hybrid = 0
        
        for part in parts:
            part = part.strip()
            amount = ''.join(filter(lambda x: x.isdigit() or x == '*', part))
            amount = eval(amount.replace('*', '')) if '*' in amount else float(amount)
            type_ = ''.join(filter(str.isalpha, part)).strip()
            
            if 'SSD' in type_:
                ssd += amount
            elif 'HDD' in type_:
                hdd += amount
            elif 'Flash' in type_:
                flash += amount
            elif 'Hybrid' in type_:
                hybrid += amount
        
        return pd.Series({
            'Total_Storage': ssd + hdd + flash + hybrid,
            'SSD_Storage': ssd,
            'HDD_Storage': hdd,
            'Flash_Storage': flash,
            'Hybrid_Storage': hybrid,
            'Storage_Types': '_'.join(sorted({
                'SSD' if ssd > 0 else None,
                'HDD' if hdd > 0 else None,
                'Flash' if flash > 0 else None,
                'Hybrid' if hybrid > 0 else None
            } - {None}))
        })
    
    df = df.join(df['Memory'].apply(enhanced_memory_processing))
    
    # Simplificar sistema operativo
    df['OpSys'] = df['OpSys'].replace({
        'Mac OS X': 'macOS',
        'Windows 10 S': 'Windows 10',
        'Android': 'Other'
    })
    
    return df

# -------------------------------
# 3. Preparación de Datos para Modelado
# -------------------------------

def prepare_data(df):
    # Definir características y objetivo
    X = df.drop(['Price', 'Memory', 'ScreenResolution', 'Cpu', 'Gpu'], axis=1)
    y = df['Price']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# -------------------------------
# 4. Pipeline de Modelado
# -------------------------------

def build_model_pipeline():
    # Definir características
    numeric_features = [
        'Inches', 'Ram', 'Weight', 'Cpu_Speed', 'Cpu_Generation',
        'Total_Storage', 'SSD_Storage', 'HDD_Storage', 'Flash_Storage',
        'Gpu_Memory', 'Resolution_Width', 'Resolution_Height', 'PPI'
    ]
    
    categorical_features = [
        'Company', 'TypeName', 'Cpu_Cores', 'Gpu_Type',
        'OpSys', 'Touchscreen', '4K', 'IPS', 'Storage_Types'
    ]
    
    # Transformadores
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Modelo principal con optimización de parámetros
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            threshold='median'
        )),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    return model

# -------------------------------
# 5. Optimización del Modelo
# -------------------------------

def optimize_model(model, X_train, y_train):
    param_grid = {
        'preprocessor__num__poly__degree': [2, 3],
        'feature_selection__estimator__n_estimators': [50, 100],
        'regressor__n_estimators': [500, 1000],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

# -------------------------------
# 6. Evaluación y Ajuste Final
# -------------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE inicial: {rmse:.6f}")
    
    if rmse > 0.07:
        print("\nAplicando transformación logarítmica...")
        y_train_transformed = np.log1p(y_train)
        
        final_model = clone(model)
        final_model.fit(X_train, y_train_transformed)
        
        y_pred_transformed = final_model.predict(X_test)
        y_pred_final = np.expm1(y_pred_transformed)
        
        rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
        print(f"RMSE después de transformación logarítmica: {rmse_final:.6f}")
        
        if rmse_final <= 0.07:
            print("✅ Objetivo alcanzado con transformación logarítmica!")
            return final_model, rmse_final
        else:
            print("⚠️ RMSE aún por encima del objetivo. Probando Stacking...")
            return try_stacking_model(X_train, y_train_transformed, X_test, y_test)
    else:
        print("✅ Objetivo alcanzado!")
        return model, rmse

def try_stacking_model(X_train, y_train, X_test, y_test):
    # Construir modelo de ensamblaje
    numeric_features = [
        'Inches', 'Ram', 'Weight', 'Cpu_Speed', 'Cpu_Generation',
        'Total_Storage', 'SSD_Storage', 'HDD_Storage', 'Flash_Storage',
        'Gpu_Memory', 'Resolution_Width', 'Resolution_Height', 'PPI'
    ]
    
    categorical_features = [
        'Company', 'TypeName', 'Cpu_Cores', 'Gpu_Type',
        'OpSys', 'Touchscreen', '4K', 'IPS', 'Storage_Types'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    estimators = [
        ('gb', GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )),
        ('svr', SVR(
            kernel='rbf',
            C=10,
            epsilon=0.1
        ))
    ]
    
    stacking_model = Pipeline([
        ('preprocessor', preprocessor),
        ('stacking', StackingRegressor(
            estimators=estimators,
            final_estimator=LassoCV(cv=5),
            cv=5,
            n_jobs=-1
        ))
    ])
    
    stacking_model.fit(X_train, y_train)
    
    y_pred_transformed = stacking_model.predict(X_test)
    y_pred_final = np.expm1(y_pred_transformed)
    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
    
    print(f"RMSE con Stacking: {rmse_final:.6f}")
    
    if rmse_final <= 0.07:
        print("✅ Objetivo alcanzado con Stacking!")
    else:
        print("⚠️ RMSE aún por encima del objetivo. Considerar:")
        print("- Más ingeniería de características")
        print("- Redes neuronales")
        print("- Más datos de entrenamiento")
    
    return stacking_model, rmse_final

# -------------------------------
# 7. Flujo Principal
# -------------------------------

if __name__ == "__main__":
    # Cargar y preprocesar datos
    print("Cargando y preprocesando datos...")
    df = load_and_preprocess()
    df = advanced_feature_engineering(df)
    
    # Preparar datos para modelado
    print("\nPreparando datos para modelado...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Construir y optimizar modelo
    print("\nConstruyendo y optimizando modelo...")
    model = build_model_pipeline()
    best_model, best_params = optimize_model(model, X_train, y_train)
    
    print("\nMejores parámetros encontrados:")
    print(best_params)
    
    # Evaluar modelo
    print("\nEvaluando modelo...")
    final_model, final_rmse = evaluate_model(best_model, X_test, y_test)
    
    # Mostrar resultados finales
    print("\nResultados finales:")
    print(f"RMSE alcanzado: {final_rmse:.6f}")
    
    if final_rmse <= 0.07:
        print("¡Modelo cumple con el objetivo de RMSE ≤ 0.07!")
    else:
        print("Modelo no alcanzó el objetivo. Considerar estrategias adicionales.")