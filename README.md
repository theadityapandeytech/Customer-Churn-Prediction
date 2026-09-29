Customer Churn Prediction

📌 Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave a service.

The project uses customer information such as contract type, tenure, monthly charges, internet service, technical support and other service-related features to predict customer churn.

🎯 Objectives

- Predict whether a customer will churn.
- Identify factors related to customer churn.
- Train a machine learning classification model.
- Evaluate the performance of the model.

🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- Machine Learning
- Data Preprocessing

📂 Project Structure

Customer-Churn-Prediction/
│
├── customer_churn.py
├── customer_churn.csv
├── requirements.txt
└── README.md

📊 Dataset

The project uses a customer churn dataset containing information about customer demographics, services, contracts and billing.

The target variable is:

Churn

where:

- "Yes" = Customer is likely to churn
- "No" = Customer is likely to stay

⚙️ Data Preprocessing

The project performs the following preprocessing steps:

1. Removes the customer ID.
2. Converts "TotalCharges" into numerical format.
3. Handles missing values.
4. Converts categorical variables into numerical variables.
5. Splits the dataset into training and testing sets.
6. Applies feature scaling.

🤖 Machine Learning Model

A Random Forest Classifier is used to predict customer churn.

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

▶️ How to Run

1. Install Python

Make sure Python is installed on your computer.

2. Install required libraries

Open Command Prompt inside the project folder and run:

pip install -r requirements.txt

3. Run the project

python customer_churn.py

📌 Expected Output

The program displays:

Dataset loaded successfully!
Dataset shape: ...

Customer Churn Prediction
-------------------------
Accuracy: ... %

Classification Report:
...

Confusion Matrix:
...

The exact accuracy depends on the dataset and preprocessing.

🚀 Future Improvements

- Add a graphical user interface.
- Add customer churn probability.
- Deploy the model as a web application.
- Compare multiple machine learning algorithms.
- Add interactive data visualizations.

👨‍💻 Author

Aditya Pandey

B.Tech – Data Science