# Assigment-6-7
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Data
file_path = r'C:\Users\Danie\Downloads\baseball.xlsx'  # Replace 'path_to_baseball_spreadsheet.xlsx' with the actual file path
data = pd.read_excel(file_path)

# Step 2: Data Exploration (optional)
print(data.head())  # Display the first few rows of the dataframe
print(data.info())  # Display information about the dataframe, including data types and missing values

# Step 3: Data Preprocessing (if necessary)
# Check for missing values and handle them appropriately if needed

# Step 4: Feature Selection
# Select features (RS, RA, W, OBP, SLG, BA) and target variable (Playoffs)
X = data[['RS', 'RA', 'W', 'OBP', 'SLG', 'BA']]
y = data['Playoffs']

# Step 5: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Selection and Training
model = LogisticRegression()  # Using logistic regression for classification
model.fit(X_train, y_train)  # Train the model on the training data

# Step 7: Model Evaluation
y_pred = model.predict(X_test)  # Make predictions on the testing data
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix
classification_rep = classification_report(y_test, y_pred)  # Classification report

# Print the evaluation results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

print("GO PACKERS HHHHHHHHHHHHHHHHHHHHHAAAAAAAAAAAAAAAAAAAAAAHAHAHAHAHA")


# Step 8: Interpretation
# Discuss the predictive power of the model and any insights gained from the analysis

