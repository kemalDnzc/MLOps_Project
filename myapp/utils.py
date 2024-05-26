import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gelecekteki downcasting davranışını etkinleştirme
pd.set_option('future.no_silent_downcasting', True)

def load_and_prepare_data(csv_file):
    data = pd.read_csv(csv_file)
    data.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)
    data = pd.get_dummies(data, columns=['EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep'])
    X = data.drop(['MathScore', 'ReadingScore', 'WritingScore'], axis=1)
    y = data[['MathScore', 'ReadingScore', 'WritingScore']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
