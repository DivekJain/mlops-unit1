import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.head())


from sklearn.model_selection import train_test_split

# Select useful columns
df = df[['Survived', 'Pclass', 'Age', 'Fare']].dropna()

X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib

joblib.dump(model, "model.joblib")
print("Model saved successfully!")