import numpy as np
import pandas as pd
import scipy.io
from copy import *
 
#target variable of size (m,1) with K classes turns into a matrix (m,K)
def extend_target_variable(y):
    """(vector) -> matrix/vector
    
    If the number of classes is greater than 2 it returns a (m*K) matrix 
    in which each column  j is a binary vector denoting if the line i is 
    labeled j of classes is bigger than two if not returns the original 
    vector
    
    >>> y=np.array([[1],[2],[3]])
    >>> extend_target_variable(y)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    """
    #number of classes
    n_class=len(np.unique(y))
    if n_class>2:
        #ordered array (yes, ordered) with the different classes of y 
        classes=np.unique(y)
        #number of lines
        m=np.shape(y)[0]
        #initialize return matrix 
        y_new=np.zeros((m,n_class))
        for j in range(n_class):
            #column of y_new is a logical vector turned into an int vector
            y_new[:,j:j+1]=(y==classes[j]).astype(int)
    else:
        y_new=y
    return y_new
    
#derivative of the sigmoid function used in back_prop   
def sigmoid_grad(z):
    """(matrix/vector/number) -> (matrix/vector/number)
    
    Return the derivative of the sigmoid function applied to a number,vector
    or matrix
    
    >>> sigmoid_grad(0)
    0.25
    """
    return sigmoid(z)*(1-sigmoid(z))

             
def create_NN(X,y,layer_list,epsilon_init,seed):
    """(matrix,vector,list,int,int) -> (list of matrices,list of matrices)
    nodes,thetas=create_NN(X,y,layer_list,epsilon_init,seed)
    Builds neural network from the input matrix X with target variable y,
    with the number of nodes in the hidden layers specified in layer_list.
    epsilon_init and seed are used for the parameter random initialization.
    
    >>> y=np.array([[1],[2],[3]])
    >>> X=np.array([[1,2,3],[1,2,3],[1,2,3]])
    >>> layer_list=[3]
    >>> epsilon_init=0.12
    >>> seed=1234
    >>> nodes,thetas=create_NN(X,y,layer_list,epsilon_init,seed)    
    """
    cop_layer_list=copy(layer_list)
    
    #vector y with K classes extended into a (m,K) matrix 
    y_new=extend_target_variable(y)
    #number of classes in y
    K=np.shape(y_new)[1]
    cop_layer_list.append(K)
    # seeds the random number generator
    np.random.seed(seed)
    #number of lines
    m=np.shape(X)[0]
    #concatenate column of 1's with X or else no bias for u
    X=np.concatenate((np.ones((m,1)),X),axis=1)
    #number of columns of X 
    n=np.shape(X)[1]
    #create return nodes (list of matrices)
    nodes=[X]
    """append list of matrices according to layer_list adding an extra node 
    to account for the bias
    """
    nodes.extend(
        [np.ones((m,cop_layer_list[i]+1)) if i<len(cop_layer_list)-1 else np.ones((m,cop_layer_list[i])) for i in range(len(cop_layer_list))]
                )
    #updates layer_list with current size of X
    cop_layer_list.insert(0,n)
    
    """randomly initializes the thetas with the dimensions 
    (columns of nodes[i],columns of nodes[i+1]) 
    """
    thetas=[np.random.rand(np.shape(nodes[i])[1],
                           np.shape(nodes[i+1])[1]-1)
                           *2*epsilon_init-epsilon_init
                           if i<len(nodes)-2
                           else np.random.rand(np.shape(nodes[i])[1],
                           np.shape(nodes[i+1])[1])*2*epsilon_init-epsilon_init
                           for i in range(len(nodes)-1)
                           ]
    
    
    return nodes,thetas
    
def forward_prop(nod,thet):
    lin_output=copy(nod)
    for i in range(len(nod)-2):
        lin_output[i+1]=np.dot(nod[i],thet[i])
        nod[i+1][:,1:]=sigmoid(lin_output[i+1])
    i+=1
    lin_output[i+1]=np.dot(nod[i],thet[i])
    nod[i+1]=sigmoid(lin_output[i+1])
        
    return nod,lin_output

def back_prop(nodes,thetas,y,lambd):
    m=np.shape(nodes[0])[0]
    deltas=[0*node for node in nodes]
    grad=[0*theta for theta in thetas]
    nods,lin_output=forward_prop(nodes,thetas)
    deltas[-1]=nods[-1]-y
    for i in range(2,len(deltas)):
        if i==2:
            deltas[-i]=np.dot(deltas[-i+1],thetas[-i+1].T)*(nods[-i]*(1-nods[-i]))#sigmoid_grad(lin_output[-i])
        else:
            deltas[-i]=np.dot(deltas[-i+1][:,1:],thetas[-i+1].T)*(nodes[-i]*(1-nodes[-i]))
    for i in range(len(grad)-1):
        grad[i]+=np.dot(nods[i].T,deltas[i+1][:,1:])
    i+=1
    grad[i]+=np.dot(nods[i].T,deltas[i+1])
    grad=[(1./m)*big_delta for big_delta in grad]
    for i in range(len(grad)):
        grad[i][1:,:]+=(lambd/m) 
    
    return grad
    
def sigmoid(mtx):
    return 1./(1.+np.exp(-(mtx)))


    
def normalize_features(X):
    n=np.shape(X)[1]
    for j in range(n):
        X[:,j]=(X[:,j]-X.mean())/(X[:,j].std())
    return X

def grad_descent(nodes,thetas,y_new,alpha,n_iter,lambd):
    j=np.zeros((n_iter,1))
    for i in range(n_iter):
        grad=back_prop(nodes,thetas,y_new,lambd)
        thetas=[thetas[k]-(alpha*grad[k]) for k in range(len(thetas))]
        hi,_=forward_prop(nodes,thetas)
        hy=hi[-1]
        j[i]=cost(hy,y_new,thetas,lambd)
        print 'iteration '+str(i+1)+': '+str(j[i])
    return thetas
    
    
def cost(h,y_new,thetas,lambd):
    m=np.shape(h)[0]
    j_pre=-np.dot(y_new.T,np.log(h))-np.dot((1-y_new.T),(np.log(1-h)))
    j=(1./m)*np.sum(np.diag(j_pre))
    if lambd>0:
        reg_j=(float(lambd)/(2*m))*sum([np.sum(theta[1:,:]) for theta in np.power(thetas,2)])
        j=j+reg_j
    return j

def grad_check(nodes,y_new,thetas,lambd,back_prop_grad,epsilon):
    grad=copy.deepcopy(thetas)
    thetasplus=copy.deepcopy(thetas)
    thetasminus=copy.deepcopy(thetas)
    for k in range(len(thetas)):
        for i in range(np.shape(thetas[k])[0]):
            for j in range(np.shape(thetas[k])[1]):
                thetasplus[k][i,j]+=epsilon
                thetasminus[k][i,j]-=epsilon
                pluses,_=forward_prop(nodes,thetasplus)
                pluses2=copy(pluses)
                minuses,_=forward_prop(nodes,thetasminus)
                minuses2=copy(minuses)
                plus=pluses2[-1]
                minus=minuses2[-1]
                costplus=cost(plus,y_new,thetasplus,lambd)
                costminus=cost(minus,y_new,thetasminus,lambd)
                grad[k][i,j]=(costplus-costminus)/(2*epsilon)
                thetasplus[k][i,j]-=epsilon
                thetasminus[k][i,j]+=epsilon

    print [back_prop_grad[i]-grad[i] for i in range(len(back_prop_grad))]
    return grad
    
    
def accuracy(h,y,threshold):        
    print 'Training Accuracy: '+str(float(sum((h>=threshold)==y)))+'%'
    
def nnet(X,y,layer_list,alpha,epsilon_init,n_iter,lambd,seed):
    nodes,thetas=create_NN(X,y,layer_list,epsilon_init,seed)
    y_new=extend_target_variable(y)
    the=grad_descent(nodes,thetas,y_new,alpha,n_iter,lambd)

    return the

def predict_nnet(X,y,threshold,layer_list,theta,epsilon_init,seed):
    nodes,_=create_NN(X,y,layer_list,epsilon_init,seed)
    nods,_=forward_prop(nodes,theta)
    pred=nods[-1]
    return pred
