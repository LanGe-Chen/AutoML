# AutoML - Hyperparameter Optimization

Hyperparameters play a crucial role for machine learning algorithms as they directly control the behavior and expressiveness of the model to fit the data. However, tuning each parameter manually can be a labor-intensive and time-consuming process, especially for larger spaces, and can sometimes require expert knowledge. 

This project uses popular techniques including Random Search, Grid Search and Bayesian Optimization(without using exsiting libraries from sklearn)to perform hyperparameter opimization for the Support Vector Machine model. There are 5 data sets provided for the investigations of those 3 techniques. These data sets are representative of problems that a company face and every data set presents a classification problem containing a number of input-output examples. 

## Prerequisites

Have Python3, sklearn and scipy installed

## Running the tests

1. Adjust the parameter search space in 'Hyperparameter_Optimization.py'
2. Run 'Hyperparameter_Optimization.py' to optimize hyperparameters via random search, grid Search and Bayesian Optimization and get the plots for the performance of the model on dataset0-4.npy

## Acknowledgments

The experiment results and discussion are shown in the 'Report_hyperparameter_optimization.pdf'