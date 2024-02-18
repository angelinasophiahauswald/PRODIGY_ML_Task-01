import sys
import pandas as pd
import numpy as np

def read_data(data):
    df = pd.read_csv(data)
    return df

def multivariate_regression(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p   

if __name__ == "__main__":
    data = sys.argv[1]
    print(read_data(data))
    