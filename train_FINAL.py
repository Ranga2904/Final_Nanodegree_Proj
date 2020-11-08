
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

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
webpath = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'

data = pd.read_csv('mileage.csv')

def clean_data(data):
    
    # Clean and one hot encode data
    x_df = data.dropna()
    
    x_df['car make'] = x_df['car name']
    x_df['car make'] = x_df['car name'].apply(lambda x: x.split()[0]) 

    x_df.drop(columns=['car name'],inplace=True)

    x_df = pd.get_dummies(x_df,columns=['car make'])
    
    #Adjusting skewness of target variable
    x_df['mpg'] = np.log(1 + 100*x_df['mpg'])


    y_df = x_df.pop("mpg")
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
