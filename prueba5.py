# label_encoder = LabelEncoder()
# for col in df.select_dtypes(include = ["object"]).columns:
#   df[col] = label_encoder.fit_transform(df[col])

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


df = pd.read_csv("laptop_data.csv", index_col=0)

df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

# Limpiar columna Weight
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)
# print(df["Memory"].unique())

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


df['SSD'] = 0
df['HDD'] = 0

# Procesar cada entrada
for idx, row in df.iterrows():
    storage = row['Memory']
    parts = [part.strip() for part in storage.split('+')]  
    
    for part in parts:
        if 'SSD' in part or 'Flash Storage' in part or 'Hybrid' in part:
            df.at[idx, 'SSD'] += to_gb(part)
        elif 'HDD' in part:
            df.at[idx, 'HDD'] += to_gb(part)

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

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Other/Linux/No OS'
df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'],inplace = True)

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

df.drop(columns=['Cpu','Cpu Name','Memory'],inplace=True) 
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
df.drop(columns=['Gpu'],inplace=True)
# print(df["Gpu brand"].unique())
# print(df[df["Cpu Name"] == 'Samsung Cortex A72&A53'])
#print(df.corr()['Price'])
from sklearn.preprocessing import OrdinalEncoder
def agrupar_valores_poco_frecuentes(columna, top_n=5):
    valores_principales = df[columna].value_counts().nlargest(top_n).index
    return df[columna].apply(lambda x: x if x in valores_principales else 'Other')
# Agrupar las categorías menos frecuentes como 'Other'
df['Company'] = agrupar_valores_poco_frecuentes('Company', top_n=5)
df['TypeName'] = agrupar_valores_poco_frecuentes('TypeName', top_n=5)
df['Gpu brand'] = agrupar_valores_poco_frecuentes('Gpu brand', top_n=4)
df['Cpu brand'] = agrupar_valores_poco_frecuentes('Cpu brand', top_n=4)
print(df['Company'].value_counts())
print(df['TypeName'].value_counts())
print(df['Gpu brand'].value_counts())
print(df['Cpu brand'].value_counts())
drop_cols = ['x_res', 'y_res']
df = df.drop(columns=drop_cols)
print(df.sample(10)) 
cat_cols = ['Company', 'TypeName', 'os', 'Cpu brand', 'Gpu brand']

# Creamos el codificador
ordinal_encoder = OrdinalEncoder()

# Ajustamos y transformamos las columnas categóricas
df[cat_cols] = ordinal_encoder.fit_transform(df[cat_cols])
x = df.drop(columns=['Price'])
y = np.log(df['Price'])
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.15, random_state=6227)

step1 = ColumnTransformer(transformers=[
    ('ord_enc', OrdinalEncoder(), [0, 1, 12, 13, 14])
], remainder='passthrough')

step2 = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5)


pipe = Pipeline([
    ('transformer', ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), cat_cols),
        ('num', 'passthrough', [col for col in df.columns if col not in cat_cols + ['Price']])
    ])),
    ('model', LinearRegression())
])

pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)

# Métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  

print("R2 Score:", r2)
print("MAE Score:", mae)
print("RMSE Score:", rmse)
from sklearn.metrics import mean_squared_log_error

msle = mean_squared_log_error(np.exp(y_test), np.exp(y_pred))
print("MSLE:", msle)


# Company        0.140371
# TypeName      -0.127313
# Inches         0.068197
# Ram            0.743007
# Weight         0.210370
# Price          1.000000
# SSD            0.619883
# HDD           -0.096441
# Touchscreen    0.191226
# IPS            0.252208
# x_res          0.556529
# y_res          0.552809
# ppi            0.473487
# os             0.177681
# Cpu brand      0.246240
# Gpu brand      0.322535