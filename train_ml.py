from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Experiment
from azureml.core import Workspace
from azureml.train.estimator import Estimator
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.train.hyperdrive import choice, loguniform
from azureml.core import ScriptRunConfig

from azureml.widgets import RunDetails
from azureml.train.automl import AutoMLConfig
from azureml.core import Workspace, Experiment
from azureml.core.experiment import Experiment
from azureml.core import Environment


ws = Workspace.from_config('quick-starts-ws-276120')
env = Environment.get(workspace=ws, name='JJ_ML_Project2')  # Replace with your environment name


def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    data_url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    Tabular_dataset = TabularDatasetFactory.from_delimited_files(path=data_url)

    print(Tabular_dataset.to_pandas_dataframe().head())

    # TODO: Split data into train and test sets.
    x, y  = clean_data(Tabular_dataset)
    #X = Tabular_dataset.drop('job', axis=1)  # Features
    #y = Tabular_dataset['age']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    print("Training set:", X_train.shape, y_train.shape)
    print("Testing set:", X_test.shape, y_test.shape)

# AUTOML Config
    automl_config = AutoMLConfig(
        task='classification',  # Specify the task type
        primary_metric='accuracy',  # Metric to optimize
        training_data=Tabular_dataset,  # Your training dataset
        label_column_name='age',  # Replace with your actual target column name
        experiment_timeout_minutes=30,  # Total time for the experiment
        n_cross_validations=5,  # Number of cross-validation folds
        enable_early_stopping=True,  # Enable early stopping
        featurization='auto',  # Automatic feature engineering
        compute_target='MLProj'  # Compute target
)

# Create and submit the experiment
    experiment_name = 'AutoML_Run'  # Replace with your experiment name
    experiment = Experiment(workspace=ws, name=experiment_name)

# Submit the AutoML run
    automl_run = experiment.submit(automl_config)
    
    # Wait for completion
    automl_run.wait_for_completion(show_output=True)

    # Retrieve the best run
    best_run = automl_run.get_best_run()

# save the best model
    model = best_run.register_model(model_name='The_best_run_AutoML',  # Replace with your desired model name
                                 model_path='outputs/model.pkl')  # Path where the model is saved
    print("Best model registered:", model.name, model.id)


 #   model.fit(X_train, y_train)  # Train the model
  #  joblib.dump(model, 'outputs/model2.pkl')  # Save the model




    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()

