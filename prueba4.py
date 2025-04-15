import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from math import sqrt, log
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Cargar datos
df = pd.read_csv("laptop_data.csv", index_col=0)

# Inspect the data for NaN values
# print("Dataset info:")
# print(df.info())
# print("\nMissing values per column:")
# print(df.isnull().sum())

# Limpiar columna Ram
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

# Limpiar columna Weight
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)
# print(df["Memory"].unique())
# Función para convertir a GB (manejando TB y GB)
def to_gb(text):
    if pd.isna(text):
        return 0
    if 'TB' in text:
        num = float(text.split('TB')[0].strip())
        return num * 1000
    elif 'GB' in text:
        return float(text.split('GB')[0].strip())
    else:
        return 0

# Inicializar columnas SSD y HDD con 0
df['SSD'] = 0
df['HDD'] = 0

# Procesar cada entrada
for idx, row in df.iterrows():
    storage = row['Memory']
    parts = [part.strip() for part in storage.split('+')]  # Dividir por "+"
    
    for part in parts:
        if 'SSD' in part or 'Flash Storage' in part or 'Hybrid' in part:
            df.at[idx, 'SSD'] += to_gb(part)
        elif 'HDD' in part:
            df.at[idx, 'HDD'] += to_gb(part)
    
# Mostrar el DataFrame resultante
#print(df[['Memory', 'SSD', 'HDD']])


# print(df["Storage_GB"].unique()) [ 128  256  512  500 1000   32 1128   64 1256 2256 2000 1512  756 2128 1024   16  768 2512 1064  180  240    8  508]

# if df["Storage_GB"]==8:
#     print(df["Storage_GB"]==8)
""" import seaborn as sns
sns.histplot(df['Price']) """


df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)#ver si es tactil
# df['Touchscreen'].value_counts().plot(kind='bar')
#df['ScreenResolution'].value_counts()
df['IPS']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
# df.sample(6)
#Separar la resolucion y
new=df['ScreenResolution'].str.split('x',n=1,expand = True)
df['x_res'] = new[0]
df['y_res'] = new[1]

#Extraer resolucion x tambien
df['x_res'] = df['x_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

df['x_res'] = df['x_res'].astype(int)
df['y_res'] = df['y_res'].astype(int)
df['ppi'] = (((df['x_res']**2) + (df['y_res']**2))**0.5/df['Inches']).astype(float)
#eliminar columnas no usadas
df.drop(columns=['ScreenResolution'],inplace = True)
# df.corr()['Price']
#print(df['OpSys'].unique())
#Sistema op
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Other/Linux/No OS'
df['os'] = df['OpSys'].apply(cat_os)
#df.drop(columns=['OpSys'],inplace = True)

#Eliminar columnas no usadas
#df.drop(columns=['ScreenResolution','x_res','y_res'],inplace = True)
#seccion cpu
def fetch_processor(text):
    if text =='Intel Core i5' or text  == 'Intel Core i7' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)

df.drop(columns=['Cpu','Cpu Name','OpSys','Memory'],inplace=True)
#print(df.sample(5))
#print(df["Cpu Name"].unique())
# print(df["ScreenResolution"].unique(),len(df["ScreenResolution"].unique()))
# print(df["ScreenResolution"].value_counts())
# # print(df.corr()['Price'])

# print(df.sample(5))
# print(df["ScreenResolution"].unique(),len(df["ScreenResolution"].unique()))
# print(df["ScreenResolution"].value_counts())


#Gpu brand
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
df.drop(columns=['Gpu'],inplace=True)
print(df.sample(10)) 
# print(df["Gpu brand"].unique())
# print(df[df["Cpu Name"] == 'Samsung Cortex A72&A53'])
#print(df.corr()['Price'])
x = df.drop(columns=['Price'])
y = np.log(df['Price'])

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.15, random_state=6227)

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 12, 13, 14])  # Ajustado a columnas válidas
], remainder='passthrough')

step2 = RandomForestRegressor(
    n_estimators=200,
    random_state=3,
    max_samples=0.5,
    max_features=0.75,
    max_depth=15
)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

# Métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("MAE Score:", mae)
print("RMSE Score:", rmse)
#Ridge Regresssion
print("Ridge Regresssion")
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0, 1, 12, 13, 14])
],remainder='passthrough')

step2 = Ridge(alpha=10)
pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
print("r2 Score ",r2_score(y_test,y_pred))
print("MAE Score ",mean_absolute_error(y_test,y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE Score:", rmse)
print("linear regresion")
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0, 1, 12, 13, 14])
],remainder='passthrough')

step2 = LinearRegression()
pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# scores.append(r2_score(y_test,y_pred))
print("r2 Score ",r2_score(y_test,y_pred))
print("MAE Score ",mean_absolute_error(y_test,y_pred))
# print(scores[np.argmax(scores)])
# print(np.argmax(scores))
print("RMSE Score:", rmse)
print("linear regresion")
print("xgboost")
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0, 1, 12, 13, 14])
],remainder='passthrough')

step2 = XGBRegressor(max_depth=5,learning_rate=0.5)
pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("MAE Score:", mae)
print("RMSE Score:", rmse)

# import pandas as pd
# import numpy as np

# # Librerías de sklearn
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Modelos
# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression, Ridge

# ===============================
# 1. CARGA DE DATOS Y PREPARACIÓN
# ===============================

# # Asegúrate de tener cargado tu DataFrame 'df'
# df = df.dropna()  # Eliminar valores nulos si hay

# x = df.drop(columns=['Price'])
# y = np.log(df['Price'])  # Regresión logarítmica

# # ========================
# # 2. TRAIN TEST SPLIT
# # ========================
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=6227)

# # ===========================
# # 3. TRANSFORMADOR DE COLUMNAS
# # ===========================

# categorical_cols = ['Company', 'TypeName', 'os', 'Cpu brand', 'Gpu brand']
# numerical_cols = [col for col in x.columns if col not in categorical_cols]

# preprocessor = ColumnTransformer(transformers=[
#     ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols),
#     ('num', StandardScaler(), numerical_cols)
# ])

# # =================================
# # 4. XGBOOST PIPELINE + GRIDSEARCH
# # =================================

# xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# xgb_pipe = Pipeline([
#     ('pre', preprocessor),
#     ('model', xgb_model)
# ])

# param_grid = {
#     'model__n_estimators': [300],
#     'model__max_depth': [4, 6],
#     'model__learning_rate': [0.1, 0.05],
#     'model__subsample': [0.7, 1.0],
#     'model__colsample_bytree': [0.7, 1.0]
# }

# grid = GridSearchCV(xgb_pipe, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
# grid.fit(x_train, y_train)

# # =================================
# # 5. EVALUACIÓN DEL MEJOR MODELO
# # =================================

# best_model = grid.best_estimator_
# y_pred = best_model.predict(x_test)

# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))

# # Convertir a escala real para evaluar real RMSE
# y_test_real = np.exp(y_test)
# y_pred_real = np.exp(y_pred)
# rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
# mae_real = mean_absolute_error(y_test_real, y_pred_real)

# print("\n===== MEJOR MODELO (XGBoost + GridSearch) =====")
# print("R2 (log):", r2)
# print("MAE (log):", mae)
# print("RMSE (log):", rmse_log)
# print("MAE (real):", mae_real)
# print("RMSE (real):", rmse_real)

# # Mostrar mejores hiperparámetros
# print("\nMejores hiperparámetros:", grid.best_params_)
 