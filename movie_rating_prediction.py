import pandas as pd
df = pd.read_csv('movies.csv', encoding='latin-1')
print(df.head())
print(df.columns)
from sklearn.preprocessing import LabelEncoder
df = df.dropna()
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df ['Director'] = le.fit_transform(df['Director'])
df['Actor 1'] = le.fit_transform(df['Actor 1'])
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
X = df[['Genre', 'Director', 'Actor 1']]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')