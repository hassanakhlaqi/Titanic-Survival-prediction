import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the dataset
try:
    df = pd.read_csv('titanic.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'titanic.csv' not found in the directory.")
    exit()

# 2. Data Cleaning
# Drop irrelevant columns
df = df.drop(['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'], axis=1)

# Convert 'Sex' to numerical values (0 for female, 1 for male)
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

# Convert 'embarked' to numerical values (S:0, C:1, Q:2)
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Fill missing Age values with mean age per sex
mean_age_men = df[df['sex'] == 1]['age'].mean()
mean_age_women = df[df['sex'] == 0]['age'].mean()
df.loc[(df.age.isnull()) & (df['sex'] == 0), 'age'] = mean_age_women
df.loc[(df.age.isnull()) & (df['sex'] == 1), 'age'] = mean_age_men

# Drop rows with null embarked
df.dropna(subset=['embarked'], inplace=True)

# Fill missing Fare with mean
df['fare'] = df['fare'].fillna(df['fare'].mean())

# Feature Scaling
df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
df['fare'] = (df['fare'] - df['fare'].min()) / (df['fare'].max() - df['fare'].min())

# 3. Feature Selection
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]
y = df['survived']

# 4. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# 5. Build and Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluation
predictions = model.predict(X_test)
print("\n--- Model Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))