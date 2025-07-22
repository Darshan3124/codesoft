# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 2: Load the dataset
df = pd.read_csv("C:/Users/darsh/OneDrive/Desktop/erand/codesoft/iris_flower_calssifier/IRIS.csv")

# Preview the dataset
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 3: Encode species if not already numeric
if df['species'].dtype == 'object':
    df['species'] = df['species'].astype('category').cat.codes  # setosa=0, versicolor=1, virginica=2

# Step 4: Split features and target
X = df.drop('species', axis=1)
y = df['species']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)

# Evaluation
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Setosa', 'Versicolor', 'Virginica'], 
            yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Optional: Visualize pairplot
df_plot = df.copy()
df_plot['species'] = df_plot['species'].replace({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
sns.pairplot(df_plot, hue='species', corner=True)
plt.suptitle("Pairplot of IRIS Data", y=1.02)
plt.show()
