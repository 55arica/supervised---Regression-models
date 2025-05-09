import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Salary_Data[1].csv")
df = df.dropna()

for col in ['Gender', 'Education Level', 'Job Title']:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns=['Salary'])
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio: 0 = Ridge, 1 = Lasso, 0.5 = mix
elastic.fit(X_train, y_train)

predictions = elastic.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2_s = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2_s}")
