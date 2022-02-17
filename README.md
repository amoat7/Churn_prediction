# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project trains and compares the results of two models that identify credit card customers that are most likely to churn. 
The Project contains a Python library written using PEP8 standards and can be run intercatively or using the command line. 

## Files and folders in this repo
- `/data/` folder: Contains the bank_data.csv file 

- `/images/` : folder location for EDA plots

- `/results/`: folder location for training and testing results

- `/logs/`: folder location for logs produced by `churn_script_logging_and_tests.py`

- `/models/`: folder location for saved logistic regression and random forest model

- `churn_library.py`: Main python module  for churn prediction. Performs the following functions:
    - `import_data(pth)`: returns dataframe for the csv found at pth
    - `perform_eda(df_eda)`: performs eda on df_eda and save figures to images folder
    - `encoder_helper(df_data, category_lst)`: helper function to turn each categorical column into a new column with
    propotion of churn for each category. 
    - `perform_feature_engineering(df_eng)`: helper function to perform feature engineering tasks such as test and train data splits. 
    - `train_models(x_train, x_test, y_train, y_test)`: train, store model results: images + scores, and store models

- `churn_scripts_logging_and_tests.py`: Testing module for churn library. 

- `churn_notebook.ipynb`: Interactive jupyter notebook for churn prediction.

## Folder Structure

- data
    - bank_data.csv
- images
    - eda
        - churn_distribution.png
        - customer_age_distribution.png
        - heatmap.png
        - marital_status_distribution.png
        - total_transaction_distribution.png
- results
    - feature_importance.png
    - logistics_results.png
    - rf_results.png
    - roc_curve_result.png
- logs
    - churn_library.log
- models
    - logistic_model.pkl
    - rfc_model.pkl
- churn_library.py
- churn_notebook.ipynb
- churn_script_logging_and_tests.py
- README.md



## How to run files

- Clone this repo. 

- Install Anconda and create an environment 

- Run `pip install -r requirements.txt`

- Run `python churn_library.py` to train models and check outputs

- Run `python churn_script_logging_and_tests.py` to test library and see output logs.



