# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the Salary dataset.
2. Separate the dataset into input feature (X) and target variable (Salary).
3. Split the dataset into training and testing data and train the Decision Tree Regressor model.
4. Predict the salary values and visualize the Decision Tree model.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SHAJIVE KUMAR J
RegisterNumber:  212225230258
```
~~~
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r"C:/Users/acer/Downloads/Salary.csv")

print(data.head())

# Convert text columns to numbers
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeRegressor(random_state=0)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Predicted Salary Values:")
print(y_pred)

# Tree Diagram
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, filled=True)
plt.title("Decision Tree Regressor")
plt.show()
~~~
## Output:
![alt text](<Screenshot 2026-02-25 094233.png>)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.