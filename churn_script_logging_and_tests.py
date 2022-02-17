import os
import logging
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
import glob
import pandas as pd 
import numpy as np
from math import ceil

# Logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	# Test data import
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

    # Test data shape
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	# Test if images were plotted
	try:
		perform_eda(import_data("./data/bank_data.csv"))
		image_files = glob.glob('images/eda/*.png')
		assert len(image_files)>0
		logging.info("Testing perform_eda: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_eda: Images for EDA not plotted")
		raise err

	try:
		assert os.path.isfile("images/eda/age_distribution.png") is True
		logging.info('Age distribution plotted')
	except AssertionError as err:
		logging.error("Age distribution not plotted")
		raise err 
	try:
		assert os.path.isfile("images/eda/churn_distribution.png") is True
		logging.info('Churn distribution plotted')
	except AssertionError as err:
		logging.error("Churn distribution not plotted")
		raise err 
	try:
		assert os.path.isfile("images/eda/marital_status_distribution.png") is True
		logging.info('Marital distribution plotted')
	except AssertionError as err:
		logging.error("Marital distribution not plotted")
		raise err 
	try:
		assert os.path.isfile("images/eda/heat_map.png") is True
		logging.info('Heat map plotted')
	except AssertionError as err:
		logging.error("Heat map not plotted")
		raise err 



	


def test_encoder_helper():
	'''
	test encoder helper
	'''
	category_lst =  [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'  ]
	len_cols = len(category_lst)

	# test if length of categorical columns is right
	try:
		df = import_data("./data/bank_data.csv")
		enc_cols_size = sum([len(np.unique(df[cat])) for cat in category_lst]) - len_cols
		df_encoded = encoder_helper(import_data("./data/bank_data.csv"), category_lst)
		assert df_encoded.shape[1] == enc_cols_size
		logging.info("Testing encoder_helper: SUCCESS")
	except AssertionError as err:
		logging.info("Testing encoder_helper: The dataframe does not appear to have the correct number of encoded categorical columns")
		raise err

def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''

	
	try:
		df = import_data("./data/bank_data.csv") 
		(_, X_test, _, _) = perform_feature_engineering(df)
		logging.info('Testing perform_feature_engineering. Data split successful: SUCCESS')
	except Exception as err:
		logging.error('Testing perform_feature_engineering failed. Error msg: %s', str(err))
    
	# Test if dataframe was split correctly
	try: 
		assert X_test.shape[0] == ceil(df.shape[0]*0.3)
		logging.info('Testing perform_feature_engineering. Data sizes are ok: SUUCESS')
	except AssertionError as err:
		logging.error('Testing perform_feature_engineering. Data sizes are incorrect: FAILED')
		raise err


def test_train_models():
	'''
	test train_models
	'''
	# train models
	try:
		df = import_data("./data/bank_data.csv") 
		(X_train, X_test, y_train, y_test) = perform_feature_engineering(df)
		train_models(X_train, X_test, y_train, y_test)
	except Exception as err:
		logging.error('Testing test_train_models failed. Error msg: %s', str(err))

	# test if models were saved
	try:
		assert os.path.isfile("models/logistic_model.pkl") is True
		logging.info('Logistic Regression model was saved.')
	except AssertionError as err:
		logging.error("Logistic Regression model NOT saved.")
		raise err 

	try:
		assert os.path.isfile("models/rfc_model.pkl") is True
		logging.info('Random Forest model was saved.')
	except AssertionError as err:
		logging.error("Random Forest model NOT saved.")
		raise err 
	
	# test if model results were saved
	try:
		assert os.path.isfile("images/results/feature_importances.png") is True
		logging.info('Feature importances plot saved.')
	except AssertionError as err:
		logging.error("Feature importances plot NOT saved.")
		raise err 
	
	try:
		assert os.path.isfile("images/results/logistic_results.png") is True
		logging.info('Logistic regression results plot saved.')
	except AssertionError as err:
		logging.error("Logistic regression results plot NOT saved.")
		raise err
	
	try:
		assert os.path.isfile("images/results/rf_results.png") is True
		logging.info('Random Forest plot saved.')
	except AssertionError as err:
		logging.error("Random Forest plot NOT saved.")
		raise err
	
	try:
		assert os.path.isfile("images/results/roc_curve.png") is True
		logging.info('ROC plot saved.')
	except AssertionError as err:
		logging.error("ROC plot NOT saved.")
		raise err

if __name__ == "__main__":
	test_import()
	test_eda()
	test_encoder_helper()
	test_perform_feature_engineering()
	test_train_models()








