# ml-guide
Machine learning basic guide

Step 1: Setting Up the Environment
1.1 Installing Python
Make sure you have Python installed on your system. If you don’t have it yet, download it from python.org.

1.2 Creating a Virtual Environment
To avoid conflicts with other libraries, we’ll create a virtual environment for this project. Open your terminal or command prompt and run:
python -m venv env
This will create a virtual environment called env.

1.3 Activating the Virtual Environment
On Windows:
.\env\Scripts\activate
On macOS/Linux:
source env/bin/activate

1.4 Installing Required Libraries
The main libraries we’ll use for this project are:
numpy
pandas
matplotlib
scikit-learn

Install them by running the following command:
pip install numpy pandas matplotlib scikit-learn

Step 2: Preparing the Data
In this step, we'll guide users on how to load and prepare the data for machine learning.

2.1 Selecting a Dataset
For this project, we'll use the popular Iris dataset. It's a simple dataset that contains information about three species of Iris flowers, with features like sepal length, sepal width, petal length, and petal width.

To load the dataset in Python, use the following code:
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()

# Get the features and labels
X = data.data  # Features
y = data.target  # Labels

2.2 Exploratory Data Analysis (EDA)
Before we build any models, it's crucial to understand the data. Let’s start with some basic exploration.
import pandas as pd

# Convert the data to a pandas DataFrame
df = pd.DataFrame(X, columns=data.feature_names)
df['species'] = y

# Display the first few rows of the DataFrame
print(df.head())

# Statistical summary of the data
print(df.describe())

2.3 Data Visualization
Visualizing the data helps in understanding how the features relate to each other. Let's create a scatter plot to visualize the distribution of the Iris flowers.
import matplotlib.pyplot as plt

# Create a scatter plot between two features
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['species'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Flowers - Sepal Length vs Sepal Width')
plt.show()








