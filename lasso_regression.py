import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Salary_Data[1].csv')
df = df.dropna()

for col in ['Gender', 'Education Level', 'Job Title']:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    
X = df.drop(columns=['Salary'])
y = df['Salary']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)

lasso.fit(x_train, 
          y_train)


pre = lasso.predict(x_test)

mse = mean_squared_error(y_test, pre)
r2 = r2_score(y_test, pre)
print(f'mse: ', mse)
print(f'r2_score: ', r2)
