import pandas as pd
data=pd.read_csv('https://github.com/hrishikesh2005/codsoft-task/raw/e0ca7119c117837929849787ac39966d719c395f/advertising.csv')
data.head()
data.info()

y=data['Sales']
X=data.drop(['Sales'], axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)

print(X_train.shape , X_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_squared_error, r2_score
mse=mean_squared_error(y_test,y_pred)
print(mse)
r2=r2_score(y_test,y_pred)
print(r2)