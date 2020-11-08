
# Predicting car mileage
In this project, we will first compare the efficacy of AutoML vs HyperConfig tuning of a GradientBoostingRegressor in predicting car mileage given certain details 
about the car. We will then deploy the most accurate model.

## Project Set Up and Installation
To set up this project in AzureML, please note the following:
(1) Download the Auto MPG dataset and upload to Registered Datasets
(2) Download all .ipynb and .py files - note that both .ipynb files reference the train.py which is in the training folder.

## Dataset

### Overview
The data that I'm using is the Auto MPG dataset available from the UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/auto+mpg

### Task
Using regression, we will use the following attributes of cars to predict their mileage:
- number of cylinders
- displacement
- horsepower
- weight
- acceleration
- model year
- origin i.e. year of car
- car name: includes make and model

### Access
I'm loading the .csv file into the Registered datasets and then copying/pasting the code to access the dataset directly from the notebook.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
The model that I chose when tuning hyperparameters was the ensemble model GradientBoostingRegressor, which - like any other ensemble technique - uses a meta learner 
to start and then gets benefit of different regressors to improve on initial weaknesses. This model is superior to a single logistic or linear regressor, which 
doesn't offer advantage of many models at once.
The hyperparameters surveyed and ranges selected for tuning are:
max_depth from 3 to 11 - this hyperparameter 
learning_rate from 0.1 to 100

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
