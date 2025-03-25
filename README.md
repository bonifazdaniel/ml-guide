# ML Guide - Basic Machine Learning Project

This guide will walk you through the steps to build a basic machine learning model using the Iris dataset. You will learn how to set up the environment, prepare the data, build a model, improve the model, and deploy it using Flask.

## Step 1: Setting Up the Environment

### 1.1 Installing Python

Make sure you have Python installed on your system. If you don't have it yet, download it from [python.org](https://www.python.org/).

### 1.2 Creating a Virtual Environment

To avoid conflicts with other libraries, we’ll create a virtual environment for this project. Open your terminal or command prompt and run:
python -m venv env


This will create a virtual environment called `env`.

### 1.3 Activating the Virtual Environment

- **On Windows:**
.\env\Scripts\activate


- **On macOS/Linux:**
source env/bin/activate


### 1.4 Installing Required Libraries

The main libraries we’ll use for this project are:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `flask`

Install them by running the following command:
pip install numpy pandas matplotlib scikit-learn flask


## Step 2: Preparing the Data

In this step, we’ll guide you on how to load and prepare the data for machine learning.

### 2.1 Selecting a Dataset

For this project, we'll use the popular **Iris dataset**. It's a simple dataset that contains information about three species of Iris flowers, with features like sepal length, sepal width, petal length, and petal width.

To load the dataset in Python, use the following code:

from sklearn.datasets import load_iris

Load the Iris dataset
data = load_iris()

Get the features and labels
X = data.data # Features y = data.target # Labels

### 2.2 Exploratory Data Analysis (EDA)

Before building the model, let's explore the data a little.

import pandas as pd

Convert the data to a pandas DataFrame
df = pd.DataFrame(X, columns=data.feature_names) df['species'] = y

Display the first few rows of the DataFrame
print(df.head())

Statistical summary of the data
print(df.describe())

### 2.3 Data Visualization
Visualizing the data helps to understand how the features relate to each other. Let’s create a scatter plot to visualize the distribution of the Iris flowers.

import matplotlib.pyplot as plt

Create a scatter plot between two features
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['species']) plt.xlabel('Sepal Length') plt.ylabel('Sepal Width') plt.title('Iris Flowers - Sepal Length vs Sepal Width') plt.show()


## Step 3: Building the Machine Learning Model
In this step, we will create and train a machine learning model using the dataset.

### 3.1 Splitting the Data
We need to split the dataset into training and testing sets to evaluate the model’s performance on unseen data.

from sklearn.model_selection import train_test_split

Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

Initialize the model
model = LogisticRegression(max_iter=200)

Train the model on the training data
model.fit(X_train, y_train)

### 3.3 Making Predictions
Once the model is trained, we can use it to make predictions on the test set.

Make predictions on the test set
y_pred = model.predict(X_test)

Display the predicted values
print("Predictions:", y_pred)

### 3.4 Evaluating the Model
We’ll use accuracy as the evaluation metric.

from sklearn.metrics import accuracy_score

Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred) print(f"Model Accuracy: {accuracy * 100:.2f}%")


## Step 4: Improving the Model
Now, let’s improve the model by fine-tuning it and trying different algorithms.

### 4.1 Hyperparameter Tuning
We’ll use **GridSearchCV** to search for the best combination of hyperparameters.

from sklearn.model_selection import GridSearchCV

Define the parameter grid
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

Initialize GridSearchCV with logistic regression
grid_search = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5)

Fit the grid search to the data
grid_search.fit(X_train, y_train)

Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)


### 4.2 Trying a Different Model
We can also try **K-Nearest Neighbors (KNN)** for this classification task.

from sklearn.neighbors import KNeighborsClassifier

Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

Train the KNN model
knn.fit(X_train, y_train)

Make predictions
y_pred_knn = knn.predict(X_test)

Calculate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn) print(f"KNN Model Accuracy: {accuracy_knn * 100:.2f}%")

### 4.3 Comparing Model Performance
Now, let’s compare the performance of both models (Logistic Regression and KNN).

print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%") print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")


## Step 5: Deploying the Model
In this step, we’ll deploy the trained machine learning model using **Flask**.

### 5.1 Setting Up Flask
Install Flask using pip:

pip install flask


### 5.2 Creating a Flask API
Create a new Python file, `app.py`, and add the following code:

from flask import Flask, request, jsonify import pickle from sklearn.datasets import load_iris from sklearn.linear_model import LogisticRegression

Initialize the Flask app
app = Flask(name)

Load the Iris dataset and train a model
data = load_iris() X, y = data.data, data.target model = LogisticRegression(max_iter=200) model.fit(X, y)

Create a route to predict the species of a flower
@app.route('/predict', methods=['POST']) def predict(): data = request.get_json() features = [data['features']] prediction = model.predict(features) return jsonify({'prediction': int(prediction[0])})

if name == 'main': app.run(debug=True)

### 5.3 Running the Flask API
To run the Flask API, execute the following command:

python app.py


This will start a local server at `http://127.0.0.1:5000/`. You can send a POST request to the `/predict` endpoint to make predictions.

### 5.4 Testing the API
To test the API, you can use **Postman** or **cURL**. Here’s an example using cURL:

curl -X POST -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}' http://127.0.0.1:5000/predict
This will return a response with the predicted class of the Iris flower.

## Step 6: Finalizing and Sharing the Project

### 6.1 Saving the Model
It’s important to save the trained model so that it can be used later without needing to retrain it every time.

import pickle

Save the trained model to a file
with open('model.pkl', 'wb') as model_file: pickle.dump(model, model_file)

To load the model later:
with open('model.pkl', 'rb') as model_file:
loaded_model = pickle.load(model_file)

### 6.2 Documenting the Project
Make sure to add detailed documentation in the `README.md` file, explaining:

- Setting up the environment
- Loading and preprocessing the data
- Training the model
- Testing the model
- Using the API

### 6.3 Pushing the Project to GitHub
Once your changes are made, push them to GitHub:

git add . git commit -m "Final version of the project" git push origin main


### 6.4 Sharing the Project
Once your project is on GitHub, you can share the link with others so they can clone and try the guide. It’s a good idea to also create a demo or tutorial video to walk users through the entire process.






