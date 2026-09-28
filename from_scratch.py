import pandas as pd
import numpy as np

df = pd.read_excel('Real estate valuation data set.xlsx')
print(df.head())
print(df.info())
print(df.columns)

X = df.drop(columns='Y house price of unit area')
y = df['Y house price of unit area']

np.random.seed(42)

indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print(x_train.shape)
print(x_test.shape)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
mean = np.mean(x_train,axis=0)
std = np.std(x_train,axis=0)

y_mean = np.mean(y_train,axis=0)
y_std = np.std(y_train,axis=0)

x_train_scaled = (x_train - mean) / std
x_test_scaled = (x_test - mean) / std

y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std
w1 = np.random.randn(4,len(x_train[0]))
b1 = np.random.randn(4)

w2 = np.random.randn(1,4)
b2 = np.zeros(1)
lr = 0.0000001

def ReLU(z):
  soft = []
  e = []
  t = 0
  for i in z:
    x = np.exp(i)
    t += x
    e.append(x)
  for i in e:
   s = i / t
   soft.append(s)
  return soft

for epoch in range(1000):
  for i in range(len(x_train_scaled)):
    z = w1 @ x_train_scaled[i] + b1
    a = ReLU(z)
    y_pred = w2 @ a + b2
    error = y_pred - y_train_scaled[i]
    loss = np.mean((error)**2)
    dw2 = a * error
    db2 = error

    delta = (w2.T@error)*ReLU(z)
    dw1 = delta[:,None] * x_train_scaled[i][None,:]
    db1 = delta

    w2 -= lr*dw2
    b2 -= lr*db2

    w1 -= lr*dw1
    b1 -= lr*db1

predictions = []
for i in range(len(x_test)):
  z_t = w1 @ x_test_scaled[i] + b1
  a_t = ReLU(z_t)
  y_test_pred = w2 @ a_t + b2
  predictions.append(y_test_pred)
print('predictions:',predictions)
print('y test:',y_test_scaled)

mse = np.mean((predictions - y_test)**2)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
    
