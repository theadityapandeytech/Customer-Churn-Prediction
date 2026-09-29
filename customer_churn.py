import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("customer_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Remove customer ID if present
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Convert TotalCharges to numeric
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

# Remove missing values
data = data.dropna()

# Convert target column
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical columns into numerical values
X = data.drop("Churn", axis=1)
y = data["Churn"]

X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nCustomer Churn Prediction")
print("-------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))