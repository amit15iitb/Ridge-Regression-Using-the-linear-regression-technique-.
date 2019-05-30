#Data preprocessing 
from pandas import *
from numpy import *
inp=read_csv('E:\ML\ML CLASSROOM\ML-ASSIGNMENT\Behavior of the urban traffic of the city of Sao Paulo in Brazil\Behavior of the urban traffic of the city of Sao Paulo in Brazil.csv')
Y=inp['Slowness in traffic (%)']
trainy=int(round(len(Y)*0.8))
Y_train=Y.iloc[:trainy]
Y_test=Y.iloc[trainy:]
print(Y_test.shape)
inp.drop(["Unnamed: 18","Slowness in traffic (%)"], axis = 1, inplace = True)
print(inp.shape)
X0=Series([1]*len(inp))
inp.insert(0, 'XO', 1)
trainx=int(round(len(inp)*0.8))
X_train=inp[:trainx]
print(X_train.shape)
X_test=inp[trainx:]
beta=zeros(len(inp.columns))
print(beta.shape)
h=dot(X_train,beta)
print(h.shape)
##### Gradient Descent algorithm
alpha=0.00001
lambd=0.001
iterate=1000
for i in range(iterate):
    h=dot(X_train,beta)
    beta[0]=beta[0]-(alpha*(sum((h)-Y_train)))
    for j in range (1,len(beta)):
        beta[j]=(beta[j]*(1-(alpha*lambd))-(alpha)*(sum(dot(((h)-Y_train),(X_train.iloc[:,j])))))
Y_est=dot(X_test,beta)
print(Y_est.shape)
error1=.5*(sum((Y_est)-(Y_test))**2)/(27*27)
print(error1)

##### Closed from algorithm
w1=(dot(X_train.T,X_train)+lambd*(eye(18)))
w2=dot(linalg.inv(w1),X_train.T)
w=dot(w2,Y_train)
print(w.shape)
Y1_est=dot(X_test,w)
error2=.5*(sum((Y1_est)-(Y_test))**2)/(27*27)
print(error2) 

#### MLE algorithm 
from math import *
sigma=.1

beta1=zeros(len(inp.columns))
for k in range(iterate):
    h1=dot(X_train,beta1)
    beta1[0]=beta1[0]-(sum((h1)-Y_train))
    for m in range(1,len(beta1)):
        beta1[m]=(beta1[m])-(sum(dot(((h1)-Y_train),X_train.iloc[:,m])))
Y2_est=dot(X_test,beta1)
print(Y2_est.shape)
error3=(27*(log(1/(sqrt(2*3.14)*sigma)))-((1/(2*sigma*sigma))*.5*(sum((Y2_est)-(Y_test))**2)))/(27*27)
print(error3)



