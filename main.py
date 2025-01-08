import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head()

X = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']
print(X.shape)
print(y.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

import joblib

joblib.dump(model, 'my_model.joblib')
