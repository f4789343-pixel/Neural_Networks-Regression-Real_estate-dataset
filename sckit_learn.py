from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import pandas as pd
import numpy as np

df = pd.read_excel('Real estate valuation data set.xlsx')

X = df.drop(columns='Y house price of unit area')
y = df['Y house price of unit area']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(df.shape)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = MLPRegressor(hidden_layer_sizes=(4,),activation='relu', solver='adam',max_iter=5000,random_state=42)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
#print('predictions:',predictions)
#print('y test', np.array(y_test))
mae = mean_absolute_error(y_test,predictions)
mse = mean_squared_error(y_test,predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print('MAE:',mae)
print("MSE:", mse)
print("RMSE:", rmse)
print('R2 Score:', r2)
