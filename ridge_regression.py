import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Salary_Data[1].csv")
df = df.dropna()
# df
for col in ['Gender', 'Education Level', 'Job Title']:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])


X = df.drop(columns = ['Salary'])
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

predictions = ridge.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'meqan squared error: ', mse)

r2_s = r2_score(y_test, predictions)
print(f'r2_score: ', r2_s)
