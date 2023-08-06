# Model Tree for Regression

This repository contains code for developing and evaluating regression models using a novel approach called Model Tree. The Model Tree is a custom regression model that combines a base regression model with a decision tree to form a powerful ensemble.

## Project Structure

Model_Tree_for_Regression/  
|-- data/  
|-- |-- data_file.csv  
|-- |-- data_file.xlsx  
|  
|-- models/  
|-- |-- gp_regr.py  
|-- |-- lasso_regr.py  
|-- |-- linear_regr.py  
|-- |-- logistic_regr.py  
|-- |-- mean_regr.py  
|-- |-- NN_regr.py  
|-- |-- ridge_regr.py  
|-- |-- rt_regr.py  
|-- |-- svm_regr.py  
|  
|-- src/  
|-- |-- ModelTree.py  
|-- |-- utils.py  
|  
|-- output/  
|-- |-- model_1/  
|-- |-- |-- model_1_train.xlsx  
|-- |-- |-- model_1_test.xlsx  
|-- |-- |-- model_1.pkl  
|-- |-- model_2/  
|-- |-- |-- model_2_train.xlsx  
|-- |-- |-- model_2_test.xlsx  
|-- |-- |-- model_2.pkl  
|-- |-- ...  
|  
|-- notebooks/  
|-- |-- data_exploration.ipynb  
|-- |-- model_evaluation.ipynb  
|-- |-- ...  
|  
|-- requirements.txt  
|-- README.md  
|-- main.py  


The project is organized into the following folders and files:

- **data**: Contains input data files in CSV or Excel format.
- **src**: Contains the main source code files for the regression models.
- **models**: Contains separate files for different regression models, such as Gaussian Process, LASSO, Linear Regression, Logistic Regression, Mean Regression, Neural Network Regression, Ridge Regression, Decision Tree Regression, and Support Vector Machine Regression.
- **output**: Will be used to store output files and model results.

## Model Tree

The main feature of this project is the Model Tree, which is a hybrid model that combines a base regression model with a decision tree. The decision tree is used to split the data into segments, and each segment is then fitted with a separate regression model. The final prediction is a weighted average of the predictions from different segments, resulting in improved performance and interpretability.

## Regression Models

The `models` folder contains separate Python files for different regression models. Each file represents a specific regression model and contains its implementation. Here's a brief description of each model:

- Gaussian Process Regression (`gp_regr.py`): Implementation of Gaussian Process regression, a non-parametric regression technique.
- LASSO Regression (`lasso_regr.py`): Implementation of LASSO regression, a linear regression technique with L1 regularization for feature selection.
- Linear Regression (`linear_regr.py`): Simple linear regression implementation.
- Logistic Regression (`logistic_regr.py`): Implementation of logistic regression for binary or multi-class classification problems.
- Mean Regression (`mean_regr.py`): A basic regression model that predicts the mean of the target values.
- Neural Network Regression (`NN_regr.py`): Implementation of a neural network-based regression model.
- Ridge Regression (`ridge_regr.py`): Implementation of Ridge regression, a linear regression technique with L2 regularization.
- Decision Tree Regression (`rt_regr.py`): Implementation of decision tree-based regression models, such as Random Forest or Decision Tree Regression.
- Support Vector Machine Regression (`svm_regr.py`): Implementation of Support Vector Machine regression.

## Source Code

The main source code for developing and evaluating regression models is located in the `src` folder. The `ModelTree.py` file contains the implementation of the Model Tree class. The `DevelopModel.py` file is used to develop and evaluate regression models based on the specified configurations.

## Utils

The `utils.py` file in the `src` folder contains utility functions used for data loading, cross-validation, data scaling, and other tasks related to regression model development.

## Data

Input data for the regression models should be placed in the `data` folder. Data can be in CSV or Excel format.

## Output

The `output` folder will be used to store the results of the trained models, evaluation metrics, and other output files.

## How to Use

To develop and evaluate a regression model, follow these steps:

1. Place the input data in the `data` folder.
2. Specify the desired regression model and configuration in the `DevelopModel` class in the `DevelopModel.py` file.
3. Run the `DevelopModel.py` script to train and evaluate the selected model.
4. Results and output files will be saved in the `output` folder.

## Dependencies

The project relies on the following Python libraries:

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

Make sure to install these libraries before running the code.

## Author

This project is created and maintained by Ehsan Mehryaar. Feel free to contact the author for any questions or suggestions related to the project.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for personal or commercial purposes. Please see the [LICENSE](LICENSE) file for more details.
