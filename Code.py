import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'bank-full.csv'
df = pd.read_csv(file_path, delimiter=';')

# Display the first few rows of the dataframe
print(df.head())

# Handle missing values (if any)
df = df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include='object').columns

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

#Defining the features
X = df.drop('poutcome', axis=1)
y = df['poutcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Training the classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

#Predictions
y_pred = clf.predict(X_test)

#Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Plot the decision tree
plt.figure(figsize=(200,100))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["unknown","other","failure","success"], rounded=True)
plt.show()

