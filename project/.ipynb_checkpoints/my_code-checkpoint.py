from ucimlrepo import fetch_ucirepo 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt  
import scipy.stats as stats
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.metrics import r2_score
import itertools

def prepare_wine_data():
    """
    fetches the wine dataset and prepares the feature and target variables.
    """
    wine = fetch_ucirepo(id=109)
    
    X = wine.data.features
    y = wine.data.targets
    
    y = X['Alcohol']
    X = X.drop(columns=['Alcohol'])
    return X, y

def create_pairwise_plot(X, title='Pairwise Plot of Predictors', alpha=0.7):
    """
    creates and displays a pairwise plot for the given dataframe.
    """
    sns.pairplot(X, diag_kind="kde")
    plt.suptitle(title, y=1.02) 
    plt.show()

def fit_model_with_intercept(X, y):
    """
    fits a linear regression model with an intercept.
    """
    X_with_intercept = sm.add_constant(X)
    model = sm.OLS(y, X_with_intercept).fit()
    
    return model

def model_diagnostics(model, X, y):
    """
    prints the model summary, calculates residuals, and plots QQ plot and histogram.
    """

    print(model.summary())
    
    residuals = model.resid
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, color='blue', bins=30)
    plt.title('Histogram of Residuals')
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('QQ Plot of Residuals')
    
    plt.tight_layout()
    plt.show()

def backward_elimination(X, y, threshold_in=0.05):
    """
    performs backward elimination based on AIC and returns a list of the selected parameters.
    """

    X_with_intercept = sm.add_constant(X)
    model = sm.OLS(y, X_with_intercept).fit()

    while True:
        # find the feature with the highest p-value
        max_p_value = model.pvalues.max()
        
        # ff the highest p-value is greater than the threshold, remove that feature
        if max_p_value > threshold_in:
            # Get the name of the feature with the highest p-value
            feature_to_remove = model.pvalues.idxmax()
            
            # remove the feature with the highest p-value
            X_with_intercept = X_with_intercept.drop(columns=[feature_to_remove])
            
            # refit the model
            model = sm.OLS(y, X_with_intercept).fit()
        else:
            break
    
    selected_features = X_with_intercept.columns.tolist()[1:] 
    
    return selected_features, model

def calculate_vif_for_model(model, X):
    """
    calculates the (VIF) for the features in the given model.
    """

    features = X[model.params.index[1:]
    X_with_intercept = sm.add_constant(features)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_intercept.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_intercept.values, i) 
                       for i in range(X_with_intercept.shape[1])]
    
    
    return vif_data





















