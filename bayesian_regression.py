from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Salary_Data[1].csv")
df.dropna(inplace=True)
for col in ['Gender', 'Education Level', 'Job Title']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Salary', axis=1)
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BayesianRidge()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, preds))
print("R^2:", r2_score(y_test, preds))
