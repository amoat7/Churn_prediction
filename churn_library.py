'''
Churn prediction library

Author: David
Date: 15-02-22
'''

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def plot_figures(plot_object, title, plot_name):
    '''
    plots eda figures

    inputs:
        plot_type: matplotlib object
        title: (str) title of plot
        plot_name: (str) filename of saved plot

    output:
        None
    '''
    
    plt.rcParams["figure.figsize"] = (20, 10)
    plot_object
    plt.title(title)
    plt.savefig(f'images/eda/{plot_name}.png', dpi=600, bbox_inches='tight')
    plt.show(block=False)
    plt.close()


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Plot churn distribution
    plot_figures(
        df['Churn'].hist(
            figsize=(
                20,
                10)),
        'Histogram of Attribute Flag',
        'churn_distribution')

    # Plot customer age distribution
    plot_figures(
        df['Customer_Age'].hist(),
        'Histogram of Customer Age',
        'age_distribution')
    
    # Plot marital status distribution
    plot_figures(
        df['Marital_Status'].value_counts(
            normalize=True).plot(
            kind='bar'),
        'Bar plot of Marital Status',
        'marital_status_distribution')
    # Plot heat map
    plot_figures(
        sns.heatmap(
            df.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2),
        'Heat map of variable correlation',
        'heat_map')


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    # one hot encoding of categorical features
    df = pd.get_dummies(df[category_lst],drop_first=True)
    return df 


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    category_lst =  [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'  ]

    quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio']

    # concatenate categorical and numerical features
    cat_df = encoder_helper(df, category_lst)
    quant_df = df[quant_columns]
    total_df = pd.concat([cat_df, quant_df], axis=1)


    # Get response
    response = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    X_train, X_test, y_train, y_test = train_test_split(total_df, response, test_size= 0.3, random_state=42)
    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # plot and save classification report of Random Forest
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/rf_results.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    # plot and save classification report of Logistic Regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/logistic_results.png', dpi=600, bbox_inches='tight')
    plt.close()
    


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # determine feature importances
    importances = model.best_estimator_.feature_importances_

    indices = np.argsort(importances)[::-1]


    names = [X_data.columns[i] for i in indices]

    # save feature importtances to output file
    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    
    plt.bar(range(X_data.shape[1]), importances[indices])

    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, dpi=600, bbox_inches='tight')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # model initialization
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    # model training and parameter search
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)
    
    # save models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # save feature importances
    feature_importance_plot(cv_rfc, X_train, 'images/results/feature_importances.png')
    
    # save ROC curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(lrc, X_test, y_test,ax=ax, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test,ax=ax, alpha=0.8)
    plt.show(block=False)
    plt.savefig('images/results/roc_curve.png', dpi=600, bbox_inches='tight')

if __name__ == '__main__':
    # Import data 
    DF = import_data(pth='data/bank_data.csv')

    # Perform EDA
    EDA = perform_eda(df=DF)

    # Feature Engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(df=DF)

    # Model Training and results
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)


