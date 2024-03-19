import sys
import pandas as pd
import numpy as np
import math
import os


def read_data(data, mode):
    df = pd.read_csv(data)
    x_test = df[["Id", "LotArea", "BedroomAbvGr", "FullBath"]]
    df_dict = x_test.set_index("Id").agg(list, 1).to_dict()
    x_train = df[["LotArea", "BedroomAbvGr", "FullBath"]]
    if mode == "train":
        y = df["SalePrice"]
        y = np.array(y)
        return x_train, y
    elif mode == "estimate":
        return x_test, df_dict


def normalise(df):
    normalised_df = df / df.max()
    x = np.array(normalised_df)
    return x


def normalise_test_features(x, max_vals):
    normalised_features = []
    for i in range(len(x)):
        normalised_features.append(x[i] / max_vals[i])
    return normalised_features


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
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost


def compute_gradient(X, y, w, b):
    print(X)
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
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.0

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
    w = w_in  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history  # return final w,b and J history for graphing


def write_output(result):
    path = "output"
    is_exist = os.path.exists(path)
    if is_exist == False:
        os.makedirs(path)

    if os.path.exists("output/output.txt") == False:
        file = open("output/output.txt", "a")
        file.write("Id,SalePrice\n")
        for id in result:
            file.write("{},{}\n".format(id, result[id]))
    else:
        os.remove("output/output.txt")
        file = open("output/output.txt", "a")
        file.write("Id,SalePrice\n")
        for id in result:
            file.write("{},{}\n".format(id, result[id]))


if __name__ == "__main__":
    # TRAINING
    data = sys.argv[1]
    x_train, y_train = read_data(data, "train")
    x_train_scaled = normalise(x_train)

    # initialize parameters
    init_w = np.random.rand(3)
    init_b = 0
    # gradient descent settings
    iterations = 1000
    # alpha = 5.0e-7
    alpha = 0.1
    # run gradient descent
    w_final, b_final, J_hist = gradient_descent(
        x_train_scaled, y_train, init_w, init_b, alpha, iterations
    )

    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m, _ = x_train_scaled.shape
    for i in range(m):
        print(
            f"prediction: {np.dot(x_train_scaled[i], w_final) + b_final:0.2f}, target value: {y_train[i]}"
        )
        if i > 100:
            break

    # ESTIMATION
    test = sys.argv[2]
    x_test, df_dict = read_data(test, "estimate")
    temp = x_test[["LotArea", "BedroomAbvGr", "FullBath"]]
    result = {}

    # finding max_vals for each feature in the df for feature scalings
    max_vals = []
    for column in temp:
        max_vals.append(temp[column].max())

    for j in df_dict:
        # scale features
        x_test_scaled = normalise_test_features(np.array(df_dict[j]), max_vals)
        # predict house price
        result[j] = multivariate_regression(x_test_scaled, w_final, b_final)

    write_output(result)
