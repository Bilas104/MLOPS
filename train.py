import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import mlflow
import mlflow.sklearn

# Set the experiment name
mlflow.set_experiment("Diabetes_Experiment", artifact_location="/home/bilas/mlflow_artifacts")

# Load the California Housing dataset directly from Scikit-learn
diabetes = load_diabetes(as_frame=True)

# Convert to pandas DataFrame
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data['target'] = diabetes.target  # Add the target variable

# Prepare the data
X = data.drop(columns=["target"])
y = data["target"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_mse = float('inf')
best_model = None
best_model_name = ""

# Define a function to train models and log experiments in MLflow
def train_and_log_model(model, model_name):
    global best_mse, best_model, best_model_name
    
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, model_name)
        print(f"{model_name} - MSE: {mse}")

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = model_name

# Train Linear Regression model
linear_model = LinearRegression()
train_and_log_model(linear_model, "Linear_Regression")

# Train Random Forest model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
train_and_log_model(random_forest_model, "Random_Forest")

if best_model:
    mlflow.sklearn.log_model(best_model, f"Best_Model_{best_model_name}")
    print(f"Best model '{best_model_name}' saved with MSE: {best_mse}")
