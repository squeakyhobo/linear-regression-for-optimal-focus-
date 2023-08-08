import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import linearRegression
from functions import cost_func
from functions import w_gradient_descent
from functions import b_gradient_descent
from linear_regression import main as linreg
from mpl_toolkits.mplot3d import Axes3D




    
def main():
    
    st.title("task length predictor")
    with st.form("forM"):
        hrs =st.number_input("how many hours do you think you will take")

        diff =st.number_input("how hard do you think it is on a scale of 1-10")

        submitted = st.form_submit_button("Submit")

    if submitted:
        

        w,b,y_max,x_max,x2_max,y_predicted,y,x1,x2 =linreg()


        
       
        x_user = np.array([hrs/x_max,diff/x2_max])

        predicted = np.dot(x_user,w) + b
        value = predicted*y_max
        
        st.write(f"you think it will take {hrs} hours and it had a diffculty of {diff}") 
        st.write(f"currenly weight1 is {w[0]}, weight2 is {w[1]} and b is {b}")
        st.write(f"it will roughly take {value} hours ")
       
        fig =plt.figure()
        ax =fig.add_subplot(111,projection="3d")
        ax.plot([x1],[x2],y_predicted)
       
        
        
        

        st.pyplot(fig)
    
    
    
    with st.form("forrrm"):
       
       
       
        st.write("add how long the taks actually took when you complete it , along with how long you thought and it preceived difficulty")
        hrs2= st.number_input("how long you thought")
        diff2 =st.number_input("its diffculty")
        actual_hrs = st.number_input("how long it actually took")
        sub = st.form_submit_button()
        if sub:
            df= pd.read_csv("data.csv")
            data2 =[[actual_hrs,hrs2,diff2]]
        
            df2 = pd.DataFrame(data2,columns=df.columns)
            
            df3 =pd.concat([df,df2],axis=0,ignore_index=True)
            df3.to_csv("data.csv",index=False)
            st.write(df3)
            w,b,y_max,x_max,x2_max,y_predicted,y,x1,x2 =linreg()
            st.write(f"currenly weight1 is {w[0]}, weight2 is {w[1]} and b is {b}")
            fig,ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})
            
            ax.plot(sorted(x1),sorted(x2),sorted(y_predicted))
            

            st.pyplot(fig)

            
            
            
            
            


   
        
        





if __name__ == "__main__":
    main()









