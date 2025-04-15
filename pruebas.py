import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# -------------------------------
# Carga y preprocesamiento de datos
# -------------------------------
df = pd.read_csv('laptop_data.csv', index_col=0)

# Limpieza básica
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)



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

print(df.head())

#print(df['Memory_Amount'])
# print(df['Memory_Type'])
#print(df['Memory_Type'].unique())

# print(df['Cpu'].unique())
# df['Cpu_Brand'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[:2]))
# print(df['Cpu_Brand'].unique())

# print(df['Gpu'].unique())

#df.groupby("Company")["TypeName"].value_counts()
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for column in df.columns:
    df[column] = LE.fit_transform(df[column])

df.info()

final_res = []
# Define data to X and y 
X = df.drop('Price',axis = 1)
y = df.Price
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


LR = LinearRegression()
LR.fit(x_train,y_train)

y_pred = LR.predict(x_test)
print(mean_squared_error(y_test,y_pred))
from sklearn.metrics import r2_score,mean_squared_error
r2_score_LR = r2_score(y_test,y_pred)
r2_score_LR
final_res.append(r2_score_LR)
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()
DTR.fit(x_train,y_train)
y_pred2 = DTR.predict(x_test)
r2_score_DTR = r2_score(y_test,y_pred2)
r2_score_DTR
final_res.append(r2_score_DTR)
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()
RFR.fit(x_train,y_train)
y_pred3 = RFR.predict(x_test)
r2_score_RFR = r2_score(y_test,y_pred3)
r2_score_RFR
final_res.append(r2_score_RFR)
from sklearn.svm import SVR
SV = SVR()
SV.fit(x_train,y_train)
y_pred1 = SV.predict(x_test)
r2_score_SV = r2_score(y_test,y_pred1)
r2_score_SV
final_res.append(r2_score_SV)
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor()
KNN.fit(x_train,y_train)
y_pred4 = KNN.predict(x_test)
r2_score_KNN = r2_score(y_test,y_pred4)
r2_score_KNN
final_res.append(r2_score_KNN)
from xgboost import XGBRegressor
XGR = XGBRegressor()
XGR.fit(x_train,y_train)
y_pred5 = XGR.predict(x_test)
r2_score_XGR = r2_score(y_test,y_pred5)
r2_score_XGR
final_res.append(r2_score_XGR)
final = np.array(final_res)
result = final.reshape(-1,1)
columns = ['R2_score']
index = ['Linear Regression','Decision Tree Regressor','Random Forest Regressor','Support Vector Regressor','KNeighbors Regressor','XGBRegressor']
#index = ['Linear Regression','SVR','DecisionTreeRegressor','RandomForestRegressor','KNeighborsRegressor','XGB','a','b']
final_result = pd.DataFrame(result,index = index,columns = columns)
print(final_result)
from sklearn.metrics import mean_squared_error
import numpy as np

rmse_LR = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_DTR = np.sqrt(mean_squared_error(y_test, y_pred2))
rmse_RFR = np.sqrt(mean_squared_error(y_test, y_pred3))
rmse_SV = np.sqrt(mean_squared_error(y_test, y_pred1))
rmse_KNN = np.sqrt(mean_squared_error(y_test, y_pred4))
rmse_XGR = np.sqrt(mean_squared_error(y_test, y_pred5))

rmse_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'KNN', 'XGBoost'],
    'RMSE': [rmse_LR, rmse_DTR, rmse_RFR, rmse_SV, rmse_KNN, rmse_XGR]
})

print(rmse_results)
