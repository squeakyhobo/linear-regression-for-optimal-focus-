import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import linearRegression
from functions import cost_func
from functions import w_gradient_descent
from functions import b_gradient_descent
# y is how many hours is actiually took. x1 is how many i think , and x2 is the diffculty i think 
def main():
    
    data =np.array([[1.1,1.5,5],[2.07,5,8],[1.93,4.0,7.0]])
    df= pd.read_csv("data.csv")
    #df =pd.DataFrame(data,columns=["0","1","2"])


    
       
    
    
        




        
       
    x1 =df["1"]
    x2 =df["2"]
    y= df["0"]

    x_max =max(x1)

    x2_max = max(x2)

    y_max = max(y)

    num_items = df.shape[0]

    df["0"] = df["0"].replace([df["0"]],[df["0"]/y_max])
    df["1"] = df["1"].replace([df["1"]],[df["1"]/x_max])
    df["2"] = df["2"].replace([df["2"]],[df["2"]/x2_max])
                        
    x1 =df["1"]
    x2 =df["2"]
    y= df["0"]    

    df2 = df[["1","2"]]
    
    x = np.array(df2.iloc[:])
   
    
   
    

   


        

    
    df_weights = pd.read_csv("weights.csv")
   
    
    

    w =np.array(df_weights.iloc[:2]).reshape((2,))
    
   
    
    b=np.array(df_weights.iloc[2])
    print(b)
   



    y= np.array(df["0"])

    alpha = 0.01
    epsilion = 0.0001
    #ask someone in discord about epsilon 





    j_hsitory = []
    w1_hisroty =[]
    w2_hisroty =[]
    b_history = []
    iteration_array =[]

    iterations = 0 

    
    
    while True:
        y_predicted = linearRegression(x,w,b) 
        
        error = cost_func(y_predicted,y)   
        new_w,d_dw =w_gradient_descent(w,alpha,y_predicted,y,x)
        new_b,d_db = b_gradient_descent(b,alpha,y_predicted,y)
        #stop when gradient is 0 as that is the minimum 
        if  d_db <epsilion and d_dw[0] < epsilion and d_dw[1] < epsilion:
            break 
        elif d_db >epsilion or d_dw[0] > epsilion or d_dw[1] >epsilion:
            b =new_b
            w = new_w
            
                
            j_hsitory.append(error)
            w1_hisroty.append(w[0])
            w2_hisroty.append(w[1])

            b_history.append(b)
            iteration_array.append(iterations)
            iterations += 1 
            if iterations% 10000 == 0:
                print(f"iteration{iterations} w is {w} b is {b}")
                
        
        
           
   
            
    print("done")
    new_weights= pd.DataFrame([w[0],w[1],b[0]])
    new_weights.to_csv("weights.csv",index=False)
    return w,b,y_max,x_max,x2_max,y_predicted,y,x1,x2
    
    


