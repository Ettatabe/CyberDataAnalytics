import numpy as np # linear algebra
from numpy import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv("data_for_student_case.csv")
data = data[data['simple_journal'] != 'Refused']
data['simple_journal'] = data['simple_journal'].map({'Chargeback': 1, 'Settled': 0})

data['card_id'] = [x.strip().replace('card', '') for x in data['card_id']]
data['ip_id'] = [x.strip().replace('ip', '') for x in data['ip_id']]
data['mail_id'] = [x.strip().replace('email', '') for x in data['mail_id']]

print(data.columns.values)
# data.drop(data.columns[[1, 12]], axis=1, inplace=True)


# data2 = pd.get_dummies(data, columns=['txvariantcode', 'shopperinteraction', 'issuercountrycode',
#                                        'cardverificationcodesupplied', 'shoppercountrycode', 'currencycode',
#                                       'cvcresponsecode', 'accountcode'])
# print(data2.columns.values)
# print(data.columns.values)
# data.drop(data.columns[[1, 2, 5, 6, 7, 9, 10, 11]], axis=1, inplace=True)
# data = data.merge(data2)


# print(data2.isnull().sum())

columns = "txid bookingdate issuercountrycode txvariantcode bin amount currencycode shoppercountrycode shopperinteraction cardverificationcodesupplied" \
          " cvcresponsecode creationdate accountcode mail_id ip_id card_id ".split()
X = pd.DataFrame.as_matrix(data,columns=columns)
Y = data.simple_journal
Y = Y.reshape(Y.shape[0],1)
X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.06)
X_test, X_dev, Y_test, Y_dev = train_test_split(X_test,Y_test, test_size=.5)

print(np.where(Y_train == 1))
print(np.where(Y_test == 1))
print(np.where(Y_dev == 1))

print("No of training Examples : "+str(X_train.shape[0]))  # 94% data
print("No of test Examples : "+str(X_test.shape[0]))       # 3% data
print("No of dev Examples : "+str(X_dev.shape[0]))         # 3% data
print("Shape of training data : "+str(X_train.shape))
print("Shape of test data : "+str(X_test.shape))
print("Shape of dev data : "+str(X_dev.shape))
print("Shape of Y test data : "+str(Y_test.shape))
print("Shape of Y dev data : "+str(Y_dev.shape))

X_train_flatten = X_train.reshape(X_train.shape[0],-1).T
Y_train_flatten = Y_train.reshape(Y_train.shape[0],-1).T
X_dev_flatten = X_dev.reshape(X_dev.shape[0],-1).T
Y_dev_flatten = Y_dev.reshape(Y_dev.shape[0],-1).T
X_test_flatten = X_test.reshape(X_test.shape[0],-1).T
Y_test_flatten = Y_test.reshape(Y_test.shape[0],-1).T

print("No of training Examples : "+str(X_train_flatten.shape))
print("No of test Examples : "+str(Y_train_flatten.shape))
print("No of X_dev Examples : "+str(X_dev_flatten.shape))
print("No of Y_dev test Examples : "+str(Y_dev_flatten.shape))
print("No of X_test Examples : "+str(X_test_flatten.shape))
print("No of Y_test Examples : "+str(Y_test_flatten.shape))
print("No of Sanity_test : "+str(X_train_flatten[0:5,0]))

# X_train_flatten = X_train_flatten[~np.isnan(X_train_flatten)]
# X_train_set = preprocessing.normalize(np.array(X_train_flatten).reshape(-1, 1))
# X_train_set = preprocessing.normalize(X_train_flatten)

# X_train_flatten = np.nan_to_num(X_train_flatten)
X_train_flatten = X_train_flatten.astype(int)

X_train_set = preprocessing.normalize(X_train_flatten)
Y_train_set = Y_train_flatten



print("No of X_train_set shape : "+str(X_train_set.shape))
print("No of Y_train_set shape : "+str(Y_train_set.shape))


def intialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

# parameters = intialize_parameters([20,30,12,5,2])
# print("W1 =" + str(parameters["W1"]))
# print("b1 =" + str(parameters["b1"]))
# print("W2 =" + str(parameters["W2"]))
# print("b2 =" + str(parameters["b2"]))
# print("W3 =" + str(parameters["W3"]))
# print("b3 =" + str(parameters["b3"]))
# print("W4 =" + str(parameters["W4"]))
# print("b4 =" + str(parameters["b4"]))

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    cache = z
    return s, cache

print(sigmoid(np.array(([2,7]))))


def relu(z):
    r = np.maximum(0, z)
    cache = z
    return r, cache
print(relu([1,-1,21]))

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)


    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A, cache = linear_activation_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches


def cost_function(AL, Y):
    m = Y.shape[1]

    cost = ((-1 / m) * ( np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(1,L+1):
        parameters["W"+str(l)]=parameters["W" + str(l)]-learning_rate*grads["dW" + str(l)]
        parameters["b"+str(l)]=parameters["b" + str(l)]-learning_rate*grads["db" + str(l)]
    return parameters

layer_dims = [16, 30, 20,  10, 5, 1]  # 5 Layer model with 3 hidden layers

# Deep Learning network to classify frauds and normal
layer_dims = [16, 30, 20, 10, 5, 1]  # 5 Layer model with 3 hidden layers


# Deep Learning network to classify frauds and normal #.0065
def nn_model(X, Y, layer_dims, learning_rate=.0065, num_iterations=2500, print_cost=True):
    costs = []

    # initialize parameters
    parameters = intialize_parameters(layer_dims)
    # for loop for iterations/epoch
    for i in range(0, num_iterations):
        # forward_propagation
        AL, caches = forward_propagation(X, parameters)

        # compute cost
        cost = cost_function(AL, Y)

        # backward_propagation
        grads = backward_propagation(AL, Y, caches)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

print(X_train_set.shape)
print(Y_train_set.shape)

parameters = nn_model(X_train_set[1,:0], Y_train_set, [16, 14, 13, 10, 8,1 ],learning_rate=.0065,num_iterations = 2500, print_cost = True)
def predict(X, y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


pred_train = predict(X_train_set, Y_train_set, parameters)

pred_test = predict(X_test_flatten, Y_test_flatten, parameters)

pred_dev = predict(X_dev_flatten, Y_dev_flatten, parameters)

