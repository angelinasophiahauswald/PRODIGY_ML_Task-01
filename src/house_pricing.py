import sys
import pandas as pd
import numpy as np

def read_data(data):
    df = pd.read_csv(data)
    return df

def multivariate_regression(x, w, b): 
    reg = np.dot(x, w) + b     
    return reg 

def cost_function(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

if __name__ == "__main__":
    data = sys.argv[1]
    x = np.array([1, 1, 1])
    w = np.array([2, 2, 2])
    print(multivariate_regression(x, w, 1))    
    