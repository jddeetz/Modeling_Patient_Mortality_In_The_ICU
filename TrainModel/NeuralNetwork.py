#Fit a neural network to the dataset
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.preprocessing import StandardScaler 

#Change directory to where the data is
import os
os.chdir("../chartlabdata-any")

#Some parameters
iterations=20000
l2_weight=0.02
feature_scaling=1

#Import the data to be fit
pickle_jar=pickle.load(open("smallerfeatures.pkl",'rb'))
X=pickle_jar[0]
y=np.array(pickle_jar[1])

if feature_scaling==1:
    scaler = StandardScaler() 
    X=scaler.fit_transform(X) 

# Splitting the dataset randomly into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Reshape some variables to keep from being annoying
y_train=y_train.reshape(y_train.shape[0],1).T
y_valid=y_valid.reshape(y_valid.shape[0],1).T
X_train=X_train.T
X_valid=X_valid.T

#Define Architecture of Network
num_nodes=[np.shape(X_train)[0],80,40,30,20,15,1]

#Create placeholders for X and Y values, so that we can later pass training data to them
def create_placeholders(n_x,n_y):
    """Creates the placeholders for the tensorflow session
    n_x and n_y are scalars. n_x is the size of the input vector.
    n_y is the number of outputs.
    
    Returns X, a placeholder for the data input of shape [n_x,None]
    Returns Y, a placeholder for the outputs of shape [n_y,None]
    Use None here because it allows flexibility on the number of examples 
    used for the placeholders
    """
    X = tf.placeholder(tf.float32, shape=(n_x,None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y,None), name="Y")
    
    return X, Y

#Function for initializing the parameters
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow.
    Returns a dictionary of tensors containing W and b parameters
    """
    W1 = tf.get_variable("W1", [num_nodes[1],num_nodes[0]], initializer = tf.contrib.layers.xavier_initializer(seed = 100))
    b1 = tf.get_variable("b1", [num_nodes[1],1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [num_nodes[2],num_nodes[1]], initializer = tf.contrib.layers.xavier_initializer(seed = 2))
    b2 = tf.get_variable("b2", [num_nodes[2],1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [num_nodes[3],num_nodes[2]], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b3 = tf.get_variable("b3", [num_nodes[3],1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [num_nodes[4],num_nodes[3]], initializer = tf.contrib.layers.xavier_initializer(seed = 4))
    b4 = tf.get_variable("b4", [num_nodes[4],1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [num_nodes[5],num_nodes[4]], initializer = tf.contrib.layers.xavier_initializer(seed = 5))
    b5 = tf.get_variable("b5", [num_nodes[5],1], initializer = tf.zeros_initializer())
    W6 = tf.get_variable("W6", [num_nodes[6],num_nodes[5]], initializer = tf.contrib.layers.xavier_initializer(seed = 42))
    b6 = tf.get_variable("b6", [num_nodes[6],1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6}
    
    return parameters

#Forward propagation function
def forward_propagation(X, parameters):
    """
    Implements forward propagation for the model LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> RELU ... ->SIGMOID
    
    Takes X, the input dataset placeholder, of shape (inputsize, m)
    Takes parameters: a dictionary of the W and b parameters
    
    Returns: A6 --- the output of the last unit
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    
    Z1 = tf.matmul(W1, X)+b1                               
    A1 = tf.nn.relu(Z1)                                    
    Z2 = tf.matmul(W2, A1)+b2                              
    A2 = tf.nn.relu(Z2)                                    
    Z3 = tf.matmul(W3, A2)+b3                              
    A3 = tf.nn.relu(Z3)
    Z4 = tf.matmul(W4, A3)+b4                              
    A4 = tf.nn.relu(Z4)
    Z5 = tf.matmul(W5, A4)+b5
    A5 = tf.nn.relu(Z5)
    Z6 = tf.matmul(W6, A5)+b6                              
    A6 = tf.nn.sigmoid(Z6)
    
    return A6

#Compute the cost function
def compute_cost(A6, Y, parameters):
    """
    Computes mean squared error cost function
    
    Takes A6 -- the output of the network, the activations of the final layer
    Takes Y -- the set of training examples
    Takes parameters: a dictionary of the W and b parameters
    
    Returns: cost -- a summary of the error versus the training set.
    """
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = A6, labels = Y))+l2_weight*tf.nn.l2_loss(parameters['W1']) + \
        l2_weight*tf.nn.l2_loss(parameters['W2'])+l2_weight*tf.nn.l2_loss(parameters['W3']) + \
        l2_weight*tf.nn.l2_loss(parameters['W4'])+l2_weight*tf.nn.l2_loss(parameters['W5'])+l2_weight*tf.nn.l2_loss(parameters['W6'])
    return cost

#Bring it all together
def model(X_train,y_train,X_valid,y_valid, learning_rate=0.0003,num_epochs=iterations,print_cost = True):
    """
    Implements a six-layer tensorflow neural network linear->RELU->linear->RELU->linear->SIGMOID
    
    Arguments:
    X_train,Y_train -- training set
    X_test,Y_test -- test set
    learning_rate -- learning rate for gradient descent optimization
    num_epochs -- number of passes through the training set, iterations of gradient descent
    print_cost -- True to print the cost every 100 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph() #in order to rerun the model without overwriting the tf variables
    #Define some variables
    (n_x,m) = X_train.shape
    print('There are',m,'training examples')
    print('There are',n_x,'inputs')
    n_y = y_train.shape[0]
    print('There are',n_y,'outputs')
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    
    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    A6 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(A6, Y,parameters)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch 
            # Run the session to execute the "optimizer" and the "cost", the feedict is X_train,y_train for (X,Y).
            _ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: y_train})
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # Plot the cost function
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate accuracy on the training set
        accuracy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = A6, labels = Y))
        print ("Training Set Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
        
        # Calculate accuracy on the validation set
        (n_x,m) = X_valid.shape
        n_y = y_valid.shape[0]
        X, Y = create_placeholders(n_x, n_y)
        A6 = forward_propagation(X, parameters)
        accuracy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = A6, labels = Y))
        
        print ("Validation Set Accuracy:", accuracy.eval({X: X_valid, Y: y_valid}))
        
        return parameters

#Optimize the parameters
parameters = model(X_train, y_train, X_valid, y_valid)

#Predict mortalities of patients in the validation set
def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    W5 = tf.convert_to_tensor(parameters["W5"])
    b5 = tf.convert_to_tensor(parameters["b5"])
    W6 = tf.convert_to_tensor(parameters["W6"])
    b6 = tf.convert_to_tensor(parameters["b6"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4,
              "W5": W5,
              "b5": b5,
              "W6": W6,
              "b6": b6}
    
    x = tf.placeholder("float", [51, 1])
    
    a6 = forward_propagation(x, params)
    
    with tf.Session() as sess:
        prediction = sess.run(a6, feed_dict = {x: X})
        
    return prediction

#Predict the mortalities of validation set
sum_error=list()
y_predict=list()
for i in range(X_valid.shape[1]):
    fingerprint=np.array(X_valid)[:,i]
    fingerprint=fingerprint.reshape(fingerprint.shape[0],1)
    mortality_prediction = predict(fingerprint, parameters)
    y_predict.append(float(mortality_prediction))

#Print model metrics as dependent on decision cutoff
from sklearn.metrics import confusion_matrix
decision_cutoffs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
print("Calculating sensitivity (true pos rate), specificity/selectivity (true neg rate), false pos rate, false neg rate, and accuracy")
for cutoff in decision_cutoffs:
    tn, fp, fn, tp = confusion_matrix(np.array(y_valid).T, (np.array(y_predict).T>cutoff)*1).ravel()
    p=fn+tp
    n=fp+tn
    tpr=tp/p
    tnr=tn/n
    fpr=1-tnr
    fnr=1-tpr
    acc=(tp+tn)/(p+n)
    print("For cutoff of ",cutoff,': ',tpr,' ',tnr,' ',fpr,' ',fnr,' ',acc)

#Show the ROC curve    
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = logreg.score(X_test, y_test)
fpr, tpr, thresholds = roc_curve(np.array(y_valid).T, np.array(y_predict).T)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC-NN.png', format='png')
plt.show()

#Calculate AUROC model metric
auroc=roc_auc_score(np.array(y_valid).T, np.array(y_predict).T)
print("The area under the ROC curve for the test set is = ",auroc)
