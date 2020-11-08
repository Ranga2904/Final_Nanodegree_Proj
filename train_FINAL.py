
from sklearn.ensemble import GradientBoostingRegressor
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

from azureml.core import Workspace, Dataset

subscription_id = 'b4a122b5-b4d5-40e7-9878-57b87adf4a8b'
resource_group = 'aml-quickstarts-125265'
workspace_name = 'quick-starts-ws-125265'

workspace = Workspace(subscription_id, resource_group, workspace_name)

data = Dataset.get_by_name(workspace, name='mpg')
data = data.to_pandas_dataframe()

data.columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']

def clean_data(data):
    
    # Clean and one hot encode data
    x_df = data.dropna()
    x_df = x_df.replace('?',0)

    x_df.drop(columns=['car name'],inplace=True)
    x_df['mpg'] = x_df['mpg'].astype(float)

    y_df = x_df.pop('mpg')
    
    return x_df,y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=3, help="maximum depth of each tree that limits number of nodes")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Factor by which each tree's contribution shrinks")

    args = parser.parse_args()

    x, y = clean_data(data)

    # TODO: Split data into train and test sets.

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

    model = GradientBoostingRegressor(max_depth=args.max_depth, learning_rate=args.learning_rate).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    
    run = Run.get_context()
    run.log("Accuracy", np.float(accuracy))
    run.log("Max depth:", np.float(args.max_depth))
    run.log("Learning rate:", np.int(args.learning_rate))

if __name__ == '__main__':
    main()
