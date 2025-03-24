
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
from azureml.train.hyperdrive import HyperDriveRun, HyperDriveConfig, RandomParameterSampling, EarlyTerminationPolicy, BanditPolicy
from azureml.train.hyperdrive import uniform, choice, PrimaryMetricGoal
from azureml.core import Environment
from azureml.widgets import RunDetails
import joblib
from azureml.core import Workspace, Experiment
from azureml.train.automl import AutoMLConfig
from sklearn.ensemble import RandomForestClassifier 
from azureml.core.experiment import Experiment
from azureml.core import Environment

ws = Workspace.from_config('quick-starts-ws-276120')
env = Environment.get(workspace=ws, name='JJ_ML_Project2')  # Replace with your environment name


compute_targets = ws.compute_targets
for name, target in compute_targets.items():
    print(f"Name: {name}, Type: {target.type}")

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

    #X = Tabular_dataset['job']  # Features
    #y = Tabular_dataset['age']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # print("Training set:", X_train.shape, y_train.shape)
    # print("Testing set:", X_test.shape, y_test.shape)

    #PARAMETERS
#{}
    learning_rate = uniform(0.05,0.1)
    batch_size = choice(16, 32, 64, 128)

    parameter_sampling = RandomParameterSampling({
    'learning_rate': learning_rate,
    'batch_size': batch_size
})

    # Define a Bandit policy
    policy = BanditPolicy(
        slack_factor=0.1,  # The slack factor to allow for some variability
        evaluation_interval=1  # How often to evaluate the performance
        #delay_evaluation=5  # Number of iterations to wait before evaluating
)
    #Estimator
    src = ScriptRunConfig(source_directory='.',  # Directory where train.py is located
                      script='train.py',  # Your training script
                      compute_target= 'MLProj',
                      environment=env)  # Your environment

    #estimator = Estimator(
    #source_directory=data_url,  # Directory where train.py is located
    #script_params={
    #    '--learning_rate': 0.01,
    #    '--batch_size': 32,},  # Replace with your actual arguments
    #compute_target='ML_Proj',  # Replace with your compute target
    #environment_definition=env  # Your environment)

    
    #HyperdriveConfig
    hyperdrive_config = HyperDriveConfig(
        run_config= src,  # Your script run configuration
        hyperparameter_sampling=parameter_sampling,
        policy=policy,  # Include the early stopping policy
        primary_metric_name='Accuracy',  # Metric to optimize
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,  # Goal to optimize
        max_total_runs=20,  # Total number of runs
        max_concurrent_runs=4, # Number of concurrent runs
)

    #Submit hyperdrive
    experiment_name = 'Hyperdrive'  # Replace with your experiment name
    experiment = Experiment(workspace=ws, name=experiment_name)
    hyperdrive_run = experiment.submit(hyperdrive_config)

    #Rundetails 
    RunDetails(hyperdrive_run).show()

    #Get the best run
    hyperdrive_run.wait_for_completion(show_output=True)  # This will block until the run is complete
    best_run_metrics = best_run.get_metrics()
    best_run = hyperdrive_run.get_best_run_by_primary_metric()
    print("Best Run ID:", best_run.id)
    print('\n Accuracy:', best_run_metrics['accuracy'])


# Register and save the best model
    model = best_run.register_model(model_name='The_best_run_Hyperdrive',  # Replace with your desired model name
                                 model_path='outputs/model.pkl')  # Path where the model is saved
    print("Best model registered:", model.name, model.id)



if __name__ == '__main__':
    main()

# Evaluate the model
    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))