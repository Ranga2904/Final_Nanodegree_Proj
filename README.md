
# Predicting car mileage
In this project, we will first compare the accuracy of AutoML vs HyperConfig's hyperparameter tuning of a GradientBoostingRegressor in predicting car mileage given certain details about the car. We will then deploy the most accurate model.

## Project Set Up and Installation
To set up this project in AzureML, please:
- download the Auto MPG dataset and register as a dataset in Azure ML studio under the name 'mpg'
- download all .ipynb and .py files - note that the HyperConfig notebook uses train.py as an entry script while AutoML uses score.py as a 
scoring script.

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

This model permits car manufacturers to optimize their products for minimal fuel consumption, which is not only good for the environment but can also provide credits.

### Access
I'm loading the .csv file as a registered dataset and then copying/pasting the code to access the registered dataset directly from the notebook.

## Automated ML
As this is a regression problem, the task was chosen appropriately and since the hyperparameter tuning was done on GradientBoostingRegressor - whose default 
model.score is the R-squared parameter - I chose the 'R2_score' as the primary metric for AutoML. 3-fold cross validation was appropriate given timing limits, with another default model being k=10. 

Since we're trying to predict mileage, 'mpg' was selected as the target variable.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The best AutoML model was a hard VotingClassifier with an R-square of 0.868. It didn't employ any regularization and weighed the predictions of 100 different estimators. Options to improve this model include:
- encoding car model/make



*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
The model that I chose when tuning hyperparameters was the ensemble model GradientBoostingRegressor, which - like any other ensemble technique - uses a meta learner 
to start and then gets benefit of different regressors to improve on initial weaknesses. This model is superior to a single logistic or linear regressor, which 
doesn't offer advantage of many models at once.
The hyperparameters surveyed and ranges selected for tuning are:
- max_depth values of 2 and 5 - this hyperparameter limits the depth of a tree and avoids an overly complex fitting and excessive variance i.e. poor model generalization to unseen data.
- learning_rate of 1 and 10 - this hyperparameter shrinks tree size and as learning rate shrinks, the model needs more trees to maintain performance.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
I got a R-squared score of 0.82 with a learning rate of 1 and max_depth of 2. Options to improve this model include:
- setting learning rate lower to 0.1 and increasing max depth to permit improved training performance
- encoding car model/make

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.


## Screen Recording
A screencast with the following details can be found at this link - https://youtu.be/dQridKJM0uY
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Opportunities to improve
Opportunities for the future include:
- adjusting features in dataset to address cardinality issue that AutoML had to work through
- deploying best model to ONNX format
