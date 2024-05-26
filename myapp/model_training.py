import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('datasets/cleaned_data.csv')
print(data.head())
print(data.info())

X = data.drop('result_label', axis=1)
y = data['result_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print('Model Accuracy:', accuracy)
print(classification_report(y_test, predictions))
