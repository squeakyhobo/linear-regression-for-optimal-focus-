import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 







def linearRegression(x_array,w,b):

    # x = 20,40
    y_predicted =[]
    
    
    
    for x in x_array:
        
        y= np.dot(x,w) + b
        y_predicted.append(y)
        
    
    return y_predicted



def cost_func(y_hatArray,y_array):
    
    # cost func = sum of (yhat - y)^2 * 2m
    sumCost= 0
    m = y_array.shape[0]

    cost = sum((y_hatArray- y_array)**2)
    
    total_cost =cost/(2*m)
    
    return total_cost

    

def w_gradient_descent(w,learning_rate,y_hatArray,y_array,x_array):
    #w= w- alpha(d/dw cost_function)
    #d/dw = 2u*x
    #d/dw =  (y_hat -wx+b)*x*1/m
# per feature 
    m = y_array.shape[0]
    n= w.shape[0]
     
    total_d_dw =np.array([0,0],dtype=float)
    for i in range(m):
        for  j in range(n) :
           
            d_dw_i = (y_hatArray[i] - y_array[i])*x_array[i][j]
            
            
            total_d_dw[j] = total_d_dw[j] +  d_dw_i
            
    d_dw = total_d_dw/m

    new_w = w - (learning_rate * d_dw)
    return new_w,d_dw


def b_gradient_descent(b,learning_rate,y_hatArray,y_array):
    #w= w- alpha(d/dw cost_function)
    # d/db = 2U*1
    #d/db =  (y_hat -wx+b)*1/m

    m = y_array.shape[0]
    total_d_db =0
    for i in range(m):
      d_db_i =  (y_hatArray[i] - y_array[i])
      total_d_db = total_d_db +d_db_i
    d_db = total_d_db/m

    new_b = b - (learning_rate * d_db)
    return new_b,d_db


        
   

    
    


def gradient_descent(w,b,learning_rate,y_hatArray,y_array,x_array):
   new_w = w_gradient_descent(w,learning_rate,y_hatArray,y_array,x_array)
   new_b = b_gradient_descent(b,learning_rate,y_hatArray,y_array)
   print(new_b)
   print(new_w)
   return new_w,new_b





#plt.scatter(x,y)
#plt.plot(x, y_predicted)
#plt.show()
