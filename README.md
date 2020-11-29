
# Predicting car mileage
In this project, we will first compare the accuracy of AutoML vs HyperConfig's hyperparameter tuning of a GradientBoostingRegressor in predicting car mileage given certain details about the car. We will then deploy the most accurate model to get an active endpoint that can be queried using those details to return a predicted mileage. 

The schematic below illustrates the path that is detailed in the rest of this write-up
![Architecture schematic](https://github.com/Ranga2904/Final_Nanodegree_Proj/blob/main/Capstone_Pipeline.png)


## Project Set Up and Installation
To set up this project in AzureML, please:
- download the Auto MPG dataset and register as a dataset in Azure ML studio under the name 'mpg'
- download all .ipynb and .py files - note that the hyperparameter_11_29_FINAL.ipynb notebook uses train_11_29.py as an entry script while automl_11_29_FINAL.ipynb uses score_11_29_FINAL.py  as a scoring script.


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
As this is a regression problem, the task was chosen as such and since the hyperparameter tuning was done on GradientBoostingRegressor - whose default 
model.score is the R-squared parameter - I chose the 'R2_score' as the primary metric for AutoML. 3-fold cross validation was appropriate given timing limits, with another default model being k=10. 

Since we're trying to predict mileage, 'mpg' was selected as the target variable.

### Results
The best AutoML model was a hard VotingClassifier with an R-square of 0.868. It didn't employ any regularization and weighed the predictions of 100 different estimators. Options to improve this model include:
- encoding car model/make rather than dropping this column
- averaging feature values to impute missing data rather than dropping those instances

Below are snapshots of the RunDetails widget for AutoML and a screenshot of the best model and run ID
###### RunDetails widget showing run details and progress
![RunDetails widget 1](https://github.com/Ranga2904/Final_Nanodegree_Proj/blob/main/Screenshot_1_AutoML_runs_1.png)
![RunDetails widget 2](https://github.com/Ranga2904/Final_Nanodegree_Proj/blob/main/Screenshot_1_AutoML_runs_2.png)
###### Best model and runID
![Best model and runID](https://github.com/Ranga2904/Final_Nanodegree_Proj/blob/main/Screenshot_2_AutoML_bestmodel_runID.png)


## Hyperparameter Tuning
The model that I chose when tuning hyperparameters was the ensemble model GradientBoostingRegressor, which - like any other ensemble technique - uses a meta learner 
to start and then gets benefit of different regressors to improve on initial weaknesses. Like any ensemble approach, this is superior to any single regressor
The hyperparameters surveyed and ranges selected for tuning are:
- max_depth values of 2 and 5 - this hyperparameter limits the depth of a tree and avoids an overly complex fitting and excessive variance i.e. poor model generalization to unseen data.
- learning_rate of 1 and 10 - this hyperparameter shrinks tree size and as learning rate shrinks, the model needs more trees to maintain performance.

### Results
I got a R-squared score of 0.82 with a learning rate of 1 and max_depth of 2. Options to improve this model include:
- setting learning rate lower to 0.1 and increasing max depth to permit improved training performance
- encoding car model/make rather than dropping this column
- averaging feature values to impute missing data rather than dropping those instances

###### RunDetails widget showing progress of training runs from experiments
![RunDetails widget showing child runs](https://github.com/Ranga2904/Final_Nanodegree_Proj/blob/main/Screenshot_4_HyperDrive_child_runs.png)
###### Best Hyperdrive model trained with its hyperparameter settings
![Best model and runID](https://github.com/Ranga2904/Final_Nanodegree_Proj/blob/main/Screenshot_4_HyperDrive_bestrunID_hyperparams.png)


## Model Deployment
The healthy endpoint is seen below
![Healthy endpoint](https://github.com/Ranga2904/Final_Nanodegree_Proj/blob/main/Screenshot_3_AutoML_healthy_endpoint.png)

The deployed model is the best performer from AutoML - a VotingEnsemble regressor that has been registered - and as seen above, has an active endpoint at the specified scoring URI. To query the endpoint:
- first replace existing values for cylinders, displacement, horsepower, weight, acceleration, model yr, and origin parameters
- these values are now part of a dictionary that is converted to json for inputting to a scoring script
- run the cell with this code. This results in the scoring script 'score_11_29_FINAL' running the saved registered model and producing a predicted mileage
- that predicted mileage is the output.


## Screen Recording
A screencast with the following details can be found at this link - https://youtu.be/sjcRizr4XVE
- A working model
- Demo of the deployed model
- Demo of a sample request sent to the endpoint and its response

## Opportunities to improve
Opportunities for the future include:
- converting registered model to ONNX format (my next step once I confirm that my deployment is done accurately)
- adjusting features in dataset to address cardinality issue that AutoML had to work through
- implement suggestions above around data imputation and encoding
