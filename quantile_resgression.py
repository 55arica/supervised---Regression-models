import pandas as pd
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm


df = pd.read_csv("Salary_Data[1].csv")
df = df.dropna()

for col in ['Gender', 'Education Level', 'Job Title']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Salary'])
y = df['Salary']
X = sm.add_constant(X)

model =  sm.QuantReg(y, X)
res = model.fit(q=0.9)

print(res.summary())
