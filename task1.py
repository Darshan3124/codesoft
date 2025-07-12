#steps to perform
# load data
# preprocess the dataset
# buiid the neural network classifier
# model training
# model prediction

# Import necessary libraries
import pandas as pd   #to load and maintain data like the excel
import numpy as np   #it is used to to the numeric operation like averge
import matplotlib.pyplot as plt   # used to create the graphs and charts
import seaborn as sns   # to learn the machine learning models and evaluate them
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset read the titanic csv file and covert it into dataframe=data in a table structure
df = pd.read_csv('C:/Users/darsh/OneDrive/Desktop/titanic survival prediction/Titanic-Dataset.csv')

# Display the first few rows only to see what thge data is present in the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Show basic info ,column names and the missing vales that are present in the dataset
print("\nDataset Info:")
print(df.info())

# Checking missing values  and how much missing values are threre
print("\nMissing values in each column:")
print(df.isnull().sum())

# Drop unnecessary columns like thsese columns have the missing values so we drop them
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing Age with median value
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode 'Sex' and 'Embarked' using LabelEncoder for the undersatanding of the machine learnig 
le = LabelEncoder()   #converts the text value into the numbers
df['Sex'] = le.fit_transform(df['Sex'])       # male = 1, female = 0
df['Embarked'] = le.fit_transform(df['Embarked'])

# Feature and Target selection in this x basically input the features like age,sex,fare and y will classify that 0=not survived and 1= survived
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data into train and test sets (80-20 split) 80 for training and 20 for the testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier=builds multiple decision tree and combines them to make a storng prediction
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) #fit learns pattern from the training data

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model in this tehy did not survive,model predicted tehy survive but they actually dont,model predicted they not survive but they actually survived,model correctly surived
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:") #it shows the precision,recall,f1 score and the support
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")  #means the model correctly predicted surival 
print(accuracy_score(y_test, y_pred))

# Feature importance plot it basically shows the which feature were the most important in predictions okayyyy
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh')
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
