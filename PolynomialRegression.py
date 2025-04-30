import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('Salary_Data[1].csv')
df = df.dropna()

for col in ['Gender', 'Education Level', 'Job Title']:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns=['Salary'])
Y = df['Salary']

poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(x_poly, Y, test_size = 0.2, random_state=42)



model = LinearRegression()
model.fit(x_train, y_train)

predict = model.predict(x_test)

mse = mean_squared_error(y_test, predict)
r2 = r2_score(y_test, predict)

print(f'mse: ', mse)
print(f'r2_score: ', r2)
