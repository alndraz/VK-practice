import pandas as pd

# Uploading data
file_path = './intern_task.csv'
data = pd.read_csv(file_path)

# Viewing the first few rows of data
print(data.head())

# Basic information about the data
print(data.info())

# Descriptive statistics
print(data.describe())
