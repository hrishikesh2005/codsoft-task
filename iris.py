import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
data = pd.read_csv('https://github.com/hrishikesh2005/codsoft-task/raw/498c524de8e0948673bf64d1482150e23461c0e2/IRIS.csv')


print("First few rows of the dataset:")
print(data.head())


X = data.drop('species', axis=1)  
y = data['species']               


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
