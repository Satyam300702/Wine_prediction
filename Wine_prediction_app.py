# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 21:49:28 2025

@author: HP
"""

import numpy as np
import pickle
import os
import streamlit as st

model_path = os.path.join(os.path.dirname(__file__),"Wine_prediction.sav")
try:
    wine_model = pickle.load(open(model_path,"rb"))
except FileNotFoundError:
    st.error("Model file not found! Make sure 'heart_disease1.sav' is in the same folder as this app.")
    st.stop()
    
def Wine_Prediction(input_data):
    input_data_as_np_array = np.asarray(input_data).reshape(1,-1)
    prediction = wine_model.predict(input_data_as_np_array)
    prediction
    if prediction[0] == 0:
        return "Wine is made from Cultivar A(grapes variety)"
    elif prediction[0] == 1:
        return "Wine is made from Cultiver B(grapes variety)"
    else:
        return "Wine is made from Cultiver C(grapes variety)"
    
def main():
    st.title("Grapes variety predictive system")
    
    alcohol = st.number_input("alcohol")
    malic_acid = st.number_input(" malic_acid ")
    ash  = st.number_input("ash")
    alcalinity_of_ash  = st.number_input("alcalinity_of_ash")
    magnesium = st.number_input("magnesium")
    total_phenols = st.number_input("total_phenols")
    flavanoids = st.number_input("flavanoids")
    nonflavanoid_phenols = st.number_input("nonflavanoid_phenols")
    proanthocyanins = st.number_input("proanthocyanins")
    color_intensity = st.number_input("color_intensity")
    hue = st.number_input("hue")
    od280_od315_of_diluted_wines = st.number_input("od280/od315_of_diluted_wines")
    proline = st.number_input("proline")
    
    Wine = ''
    if st.button("Check for grapes variety"):
        Wine = Wine_Prediction([alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280_od315_of_diluted_wines,proline])
    st.success(Wine)   
    
if __name__ == '__main__':
    main()
                        