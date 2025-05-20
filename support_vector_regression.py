import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Salary_Data[1].csv')
df.dropna(inplace=True)

for col in ['Gender', 'Education Level', 'Job Title']:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
# X = df.drop('Salary', axis=1)
X = df.drop(columns=['Salary'])
y = df['Salary']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = SVR(kernel='rbf', C=100, epsilon=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, predictions)) # Evaluation
print("R^2:", r2_score(y_test, predictions))
