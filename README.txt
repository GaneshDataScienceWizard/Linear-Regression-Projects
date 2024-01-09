# Salary Prediction with Machine Learning

## Introduction
This project focuses on predicting salaries based on job experience using machine learning techniques. We leverage a dataset that includes information about job experience and corresponding salaries. The goal is to analyze the relationship between these variables and build a regression model for salary prediction.

### Dataset
You can download the dataset [here](https://statso.io/wp-content/uploads/2022/10/Salary_Data.csv)

The dataset contains two columns:
- `YearsExperience`: The years of job experience.
- `Salary`: The corresponding salary.


## Getting Started
To start working with the project, follow the steps outlined below:

#### Installation
Make sure you have the required Python libraries installed. You can use the following commands to install them:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```

#### Code Overview

Summary of the dataset

```python
print(data.info())
```
Check if the dataset has any null values:

```python
print(data.isnull().sum())
```

No null values are present in the dataset.

#### Data Visualization
Explore the relationship between salary and job experience:

```python
plt.scatter(x=YearsExperience, y="Salary")
plt.show()
```

The scatter plot indicates a perfect linear relationship, suggesting that more job experience corresponds to a higher salary.

### Training the Machine Learning Model
Since this is a regression analysis problem, we will use a Linear Regression model to predict salary. Here's how to split the data and train the model:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = pd.DataFrame(data["YearsExperience"])   # independent variable
y = pd.DataFrame(data["Salary"])    # dependent variable

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
```

Train the Machine Learning model:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

## Making Predictions on New data
Predict the salary of a person using the trained model:

```python
new_data = int(input("Years of Experience : "))
features = np.array([[new_data]])
print("Predicted Salary = ", model.predict(features))
```

Enter the years of experience, and the model will predict the corresponding salary.

## Summary
In conclusion, this project demonstrates the process of salary prediction using machine learning techniques, specifically linear regression. We discovered a perfect linear relationship between job experience and salary, indicating that more experience leads to a higher salary. This project serves as a beginner-friendly introduction to data science.

Feel free to ask any questions or provide feedback in the comments section. Happy coding!