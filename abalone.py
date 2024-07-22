import pandas as pd
import joblib
import streamlit as st
st.header('ABALONE')
st.subheader('WHAT ARE ABALONES')
st.sidebar.title('abalone creatures')
st.write('Abalones are sea creatures that are a found mostly in the carebean and Asia.') 
df='tec.pkl'
clf=joblib.load(df)
features_name=list(['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight',
       'Viscera_weight', 'Shell_weight','Sex_F' ,'Sex_I', 'Sex_M'])
def get_user_input():
    user_input=[]
    for feature in features_name:
        value=st.text_input(f'{feature}:','0')
        user_input.append(float(value))
    return pd.DataFrame([user_input],columns=features_name)    
def main():
    st.write('Enter the values for Abalone classification')
    user_input=get_user_input()
    if st.button('Predict'):
        prediction_proba=clf.predict_proba(user_input)[:,1]
        prediction=clf.predict(user_input)

        st.write(f'Predicted probability of positive class:{prediction_proba[0]:.4f}%')
        st.write(f"Predicted class: {'It is an Abalone' if prediction[0] == 1 else 'not an Abalone'}")

if __name__=="__main__":
    main()
    




    


      
