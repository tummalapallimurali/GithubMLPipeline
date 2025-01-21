# create a regression model using the training data using mlflow logging, tracking and model registry

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/mlflow/mlflow/0.7.0/examples/sklearn_elasticnet_wine/train.csv')



# Split the data into training and test sets
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]
alpha = 0.5

# Start a new MLflow run
with mlflow.start_run():
    # Train the model
    lr = LinearRegression(alpha=alpha)
    lr.fit(train_x, train_y)

    # Make predictions
    predicted_qualities = lr.predict(test_x)

    # Log model
    mlflow.sklearn.log_model(lr, "model")

    # Log model parameters
    mlflow.log_param("alpha", alpha)

    # Log metrics
    mse = mean_squared_error(test_y, predicted_qualities)
    mae = mean_absolute_error(test_y, predicted_qualities)
    r2 = r2_score(test_y, predicted_qualities)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    print("Alpha: ", alpha)
    print("  mse: %s" % mse)
    print("  mae: %s" % mae)
    print("  r2: %s" % r2)

    # Register the model
    mlflow.sklearn.log_model(lr, "model")
    mlflow.register_model("runs:/<run_id>/model", "wine-quality-regression-model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    # run this in conda environment
    # conda_env = { 
    #     'name': 'mlflow-env',
    #     'channels': ['defaults'],
    #     'dependencies': ['python=3.7.3', 
    #                         'scikit-learn=0.21.2', 
    #                         'pandas=0.24.2', 
    #                         'numpy=1.16.4']
    # }
    # mlflow.pyfunc.log_model("model", conda_env=conda_env)
    # print("Model saved in run %s" % mlflow.active_run().info.run_uuid)