import sys
import pandas as pd
import numpy as np
import math

def read_data(data, mode):
  df = pd.read_csv(data)
  x_temp = df[["Id", "LotArea", "BedroomAbvGr", "FullBath"]]
  x = df[["LotArea", "BedroomAbvGr", "FullBath"]]
  df_dict = x_temp.to_dict()
  if mode == "train":
    y = df["SalePrice"]
    y = np.array(y)
    return x, y
  return x, df_dict

def normalise(df):
  normalized_df = (df-df.min())/(df.max()-df.min())
  x = np.array(normalized_df)
  return x

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

def compute_gradient(X, y, w, b): 
  """
  Computes the gradient for linear regression 
  Args:
    X (ndarray (m,n)): Data, m examples with n features
    y (ndarray (m,)) : target values
    w (ndarray (n,)) : model parameters  
    b (scalar)       : model parameter
      
  Returns:
    dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
    dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
  """
  m,n = X.shape           #(number of examples, number of features)
  dj_dw = np.zeros((n,))
  dj_db = 0.

  for i in range(m):                             
      err = (np.dot(X[i], w) + b) - y[i]   
      for j in range(n):                         
          dj_dw[j] = dj_dw[j] + err * X[i, j]    
      dj_db = dj_db + err                        
  dj_dw = dj_dw / m                                
  dj_db = dj_db / m                                
        
  return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
  """
  Performs batch gradient descent to learn w and b. Updates w and b by taking 
  num_iters gradient steps with learning rate alpha
    
  Args:
    X (ndarray (m,n))   : Data, m examples with n features
    y (ndarray (m,))    : target values
    w_in (ndarray (n,)) : initial model parameters  
    b_in (scalar)       : initial model parameter
    alpha (float)       : Learning rate
    num_iters (int)     : number of iterations to run gradient descent
      
  Returns:
    w (ndarray (n,)) : Updated values of parameters 
    b (scalar)       : Updated value of parameter 
    """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
  J_history = []
  w = w_in  #avoid modifying global w within function
  b = b_in
    
  for i in range(num_iters):
      # Calculate the gradient and update the parameters
      dj_db,dj_dw = compute_gradient(X, y, w, b)
      # Update Parameters using w, b, alpha and gradient
      w = w - alpha * dj_dw
      b = b - alpha * dj_db
      # Save cost J at each iteration
      if i<100000:      # prevent resource exhaustion 
          J_history.append( cost_function(X, y, w, b))
      # Print cost every at intervals 10 times or as many iterations if < 10
      if i% math.ceil(num_iters / 10) == 0:
          print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
  return w, b, J_history #return final w,b and J history for graphing

if __name__ == "__main__":
  # TRAINING
  data = sys.argv[1]
  x_train , y_train = read_data(data, "train")
  x_train_scaled = normalise(x_train)

  # initialize parameters
  init_w = np.random.rand(3)
  init_b = 0
  # gradient descent settings
  iterations = 1000
  # alpha = .0e-7
  alpha = 0.1
  # run gradient descent 
  w_final, b_final, J_hist = gradient_descent(x_train_scaled, y_train, init_w, init_b, alpha, iterations)
    
  print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
  m,_ = x_train_scaled.shape
  for i in range(m):
    print(f"prediction: {np.dot(x_train_scaled[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")  
    if i > 10:
      break

# ESTIMATION
test = sys.argv[2]
x_test, df_dict = read_data(test, "estimate")
print(df_dict)
x_test_scaled = normalise(x_train)
'''for i in x_test_scaled:
  print(multivariate_regression(i, w_final, b_final))'''