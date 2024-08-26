import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from PIL import Image
import time

#welcoming
st.title("welcom to heat pipe solar collector predictor!")
photo_w=Image.open('welcome.jpg')
st.image(photo_w,use_column_width=True)
st.title("please read the user guidance before running the program!")
photo_g=Image.open('gd.jpg')
st.image(photo_g,use_column_width=True)
st.markdown("<h1 style='color: pink;'>user guidance:</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #ffe6f2; padding: 10px; border-radius: 5px;">
    <strong>This web application is designed to predict the outputs of a heat pipe collector, specifically the outlet temperature and collector efficiency. Based on the provided schematic of the solar collector below, please determine the requested design parameters and enter them into the boxes on the side of the page.!</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>"*10, unsafe_allow_html=True)


user_choice = st.sidebar.radio("please choose the function of prediction ",
                       ("feature-selection input datas", "all input datas"))

#########################guidence pic

#......................
if user_choice == "all input datas":
   # Title of the app
    st.title("Solar Thermal Collector Performance Prediction")

# Load and display image
    photo = Image.open('gui2.jpg')
    st.image(photo,use_column_width=True)
    
    
# Dictionary to map fluid names to saturation temperatures
    fluid_temps = {
    'Acetone': 0.000000169,
    'Methanol':9.23E-08,
    'Ethanol': 7.13E-08,
    'Water': 0.000000169
    }
# Sidebar inputs
    manifold_fluid = st.sidebar.radio('Manifold fluid', ['Water', 'Air'])
#using dct
    fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
    alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary
    st.sidebar.markdown("<h1 style='color: green;'>Environmental parameters</h1>", unsafe_allow_html=True)
    
    #m_dot=st.sidebar.slider('Mass flow rate of manifold fluid (m_dot)',min_value=0.001,max_value=1.0,step=0.0001)
    m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.5f",min_value=0.001,max_value=1.0)
    T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f",min_value=10.0,max_value=90.0)
    T_amb = st.sidebar.number_input('Ambient temperature (T_amb)',format="%.4f",min_value=10.0,max_value=50.0)
    U_amb = st.sidebar.number_input('Velocity of wind (U_amb)',format="%.4f",min_value=0,max_value=12)
    st.sidebar.markdown("<h1 style='color: red;'>Radiation parameters</h1>", unsafe_allow_html=True)
    #st.sidebar.write('Radiation specifications:')
    I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f",min_value=100.0,max_value=1200.0)
    tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)',format="%.4f",min_value=0.8,max_value=0.94)
    epsilon_ab = st.sidebar.number_input('Emissivity of absorber (epsilon_ab)',format="%.4f",min_value=0.08,max_value=0.1)
    epsilon_g = st.sidebar.number_input('Emissivity of glass (epsilon_g)',format="%.4f",min_value=0.02,max_value=0.07)
    alpha_ab = st.sidebar.number_input('Absorptivity of absorber (alpha_ab)',format="%.4f",min_value=0.9,max_value=0.98)
    st.sidebar.markdown("<h1 style='color: blue;'>Geometric parameters</h1>", unsafe_allow_html=True)
    #st.sidebar.write('Geometric specifications:')
    N = st.sidebar.number_input('Number of tubes (N)', min_value=1,max_value=25)
    D_o = st.sidebar.number_input('Outer diameter of tube (D_o)',format="%.4f",min_value=47.0,max_value=102.0)
    L_tube = st.sidebar.number_input('Length of the tube (L_tube)',format="%.4f",min_value=1000.0,max_value=2100.0)
    L_c = st.sidebar.number_input('Length of the condenser (L_c)',format="%.4f",min_value=70.0,max_value=400.0)
    L_e = st.sidebar.number_input('Length of the evaporator (L_e)',format="%.4f",min_value=750.0,max_value=2000.0)
    L_a = st.sidebar.number_input('Length of the adiabatic section (L_a)',format="%.4f",min_value=0.0,max_value=300.0)
    De_o = st.sidebar.number_input('Outer diameter of evaporator (De_o)',format="%.4f",min_value=8.0,max_value=36.0)
    t_e = st.sidebar.number_input('Thickness of evaporator (t_e)',format="%.4f",min_value=0.1,max_value=1.0)
    Dc_o = st.sidebar.number_input('Outer diameter of condenser (Dc_o)',format="%.4f",min_value=12.0,max_value=36.0)
    t_c = st.sidebar.number_input('Thickness of condenser (t_c)',format="%.4f",min_value=0.1,max_value=1.0)
    theta = st.sidebar.number_input('Angle of radiation (theta)',format="%.4f",min_value=0.0,max_value=60.0)
    t_g = st.sidebar.number_input('Thickness of glass (t_g)',format="%.4f",min_value=1.5,max_value=6.0)
    D_H = st.sidebar.number_input('Hydraulic diameter of tube (D_H)',format="%.4f",min_value=20.0,max_value=270.0)
    A_man = st.sidebar.number_input('Area of manifold (A_man)',format="%.4f",min_value=0.002,max_value=10.0)
   
   

# Prepare input data for prediction
    input_data = pd.DataFrame({
    'alpha': [alpha],
    'N': [N],
    'D_o': [D_o],
    'L_tube': [L_tube],
    'L_c': [L_c],
    'L_e': [L_e],
    'L_a': [L_a],
    'De_o': [De_o],
    't_e': [t_e],
    'Dc_o': [Dc_o],
    't_c': [t_c],
    'theta': [theta],
    't_g': [t_g],
    'D_H': [D_H],
    'A_man': [A_man],
    'alpha_ab': [alpha_ab],
    'epsilon_ab': [epsilon_ab],
    'epsilon_g': [epsilon_g],
    'tau_g': [tau_g],
    'I': [I],
    'T_amb': [T_amb],
    'U_amb': [U_amb],
    'T_in': [T_in],
    'm_dot': [m_dot]
    })
#.................create process bar
    progress_bar=st.progress(0)

# Load data based on fluid type selection
    if manifold_fluid == 'Water':
    # Load models and scalers for Water
        loaded_model_eta = xgb.XGBRegressor(n_estimators=13000)
        loaded_model_eta.load_model('xgbwatereta.json')
        loaded_model_T = xgb.XGBRegressor(n_estimators=13000)
        loaded_model_T.load_model('xgbwaterT.json')

#....................................................
        with open('Xwater_eta.pkl', 'rb') as f:
            x_scaler_eta = pickle.load(f)
        with open('Ywater_eta.pkl', 'rb') as f:
            scaler_y_eta = pickle.load(f)

        with open('Xwater_T.pkl', 'rb') as f:
            x_scaler_T = pickle.load(f)
        with open('Ywater_T.pkl', 'rb') as f:
            scaler_y_T = pickle.load(f)
        #............................
        progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models
        input_data_scaled_eta = x_scaler_eta.transform(input_data.values.reshape(1, -1))
        y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
        y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))

        input_data_scaled_T = x_scaler_T.transform(input_data.values.reshape(1, -1))
        y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
        y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
#..................
        progress_bar.progress(66)
    
    # Display prediction results
        st.subheader("Predicted Outputs:")
        st.write("Predicted Efficiency (eta):", y_pred_eta[0][0],format="%.4f")
        st.write("Predicted Exit Temperature (T):", y_pred_T[0][0],format="%.4f") 

        progress_bar.progress(100)
    else: # manifold_fluid == 'Air'
        # Load models and scalers for air
        loaded_model_eta = xgb.XGBRegressor(n_estimators=13000)
        loaded_model_eta.load_model('xgbaireta.json')
        loaded_model_T = xgb.XGBRegressor(n_estimators=13000)
        loaded_model_T.load_model('xgbairT.json')

#....................................................
        with open('Xair_eta.pkl', 'rb') as f:
            x_scaler_eta = pickle.load(f)
        with open('Yair_eta.pkl', 'rb') as f:
            scaler_y_eta = pickle.load(f)

        with open('Xair_T.pkl', 'rb') as f:
            x_scaler_T = pickle.load(f)
        with open('Yair_T.pkl', 'rb') as f:
            scaler_y_T = pickle.load(f)
        #..............
        progress_bar.progress(33) 

    # Make predictions for both models
        input_data_scaled_eta = x_scaler_eta.transform(input_data.values.reshape(1, -1))
        y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
        y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))

        input_data_scaled_T = x_scaler_T.transform(input_data.values.reshape(1, -1))
        y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
        y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
    #......
        progress_bar.progress(66) 

    # Display prediction results
        st.subheader("Predicted Outputs for Water:")
        st.write("Predicted Efficiency (eta):", y_pred_eta[0][0],format="%.4f")
        st.write("Predicted Exit Temperature (T_exit):", y_pred_T[0][0],format="%.4f")
        progress_bar.progress(100) 
        
        #########################################################################..................
        ##########################33
        ###########################33
        #.....................................feature selection
        #.................................................
        # Load data based on fluid type selection
else: #*******************second back end featuuureeeeeeeee
      # Title of the app
    st.title("Solar Thermal Collector Performance Prediction")

# Load and display image
    photo = Image.open('gui2.jpg')
    st.image(photo,use_column_width='auto')
    manifold_fluid = st.sidebar.radio('Manifold fluid', ['Water', 'Air'])
    predicted_parameter = st.sidebar.radio('predicted_parameter', ['Temperature', 'efficiency'])
    
    if manifold_fluid == 'Water':
        if predicted_parameter=='Temperature':
            
            fluid_temps = {
            'Acetone': 0.000000169,
            'Methanol':9.23E-08,
            'Ethanol': 7.13E-08,
            'Water': 0.000000169
            }
            fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
            alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary
            st.sidebar.markdown("<h1 style='color: green;'>Environmental parameters</h1>", unsafe_allow_html=True)
            m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.5f",min_value=0.01,max_value=1.0)      
            T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f",min_value=10.0,max_value=90.0)
            T_amb = st.sidebar.number_input('Ambient temperature (T_amb)',format="%.4f",min_value=10.0,max_value=50.0)
            st.sidebar.markdown("<h1 style='color: red;'>Radiation parameters</h1>", unsafe_allow_html=True)
            I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f",min_value=100.0,max_value=1200.0)
            st.sidebar.markdown("<h1 style='color: blue;'>Geometric parameters</h1>", unsafe_allow_html=True)
            N = st.sidebar.number_input('Number of tubes (N)',  min_value=1,max_value=25)
             # Dictionary to map fluid names to saturation temperatures
            
            input_data_T_water = pd.DataFrame({
            'alpha': [alpha],
            'T_in': [T_in],
            'm_dot': [m_dot],
            'N': [N],
            'I': [I],
            'T_amb': [T_amb],})
            progress_bar=st.progress(0)
    
            loaded_model_T = xgb.XGBRegressor(n_estimators=13000)
            loaded_model_T.load_model('xgbFwaterT.json')

#....................................................
            with open('XwaterF_T.pkl', 'rb') as f:
                x_scaler_T = pickle.load(f)
            with open('YwaterF_T.pkl', 'rb') as f:
                scaler_y_T = pickle.load(f)
        #............................
            progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models

            input_data_scaled_T = x_scaler_T.transform(input_data_T_water.values.reshape(1, -1))
            y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
            y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
#..................
            progress_bar.progress(66)
    
    # Display prediction results
            st.subheader("Predicted Outputs:")
            st.write("Predicted outlet Temperature (T_exit):", y_pred_T[0],format="%.4f")  
            progress_bar.progress(100)
   #.........................

        else:#target is water efficiency
    # Dictionary to map fluid names to saturation temperatures

            fluid_temps = {
            'Acetone': 0.000000169,
            'Methanol':9.23E-08,
            'Ethanol': 7.13E-08,
            'Water': 0.000000169
            }
            fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
            alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary 
            st.sidebar.markdown("<h1 style='color: green;'>Environmental parameters</h1>", unsafe_allow_html=True)
            m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.5f",min_value=0.01,max_value=1.0)
            T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f",min_value=10.0,max_value=90.0)
            T_amb = st.sidebar.number_input('Ambient temperature (T_amb)',format="%.4f",min_value=10.0,max_value=50.0)
            st.sidebar.markdown("<h1 style='color: red;'>Radiation parameters</h1>", unsafe_allow_html=True)
            I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f",min_value=100.0,max_value=1200.0)
            tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)',format="%.4f",min_value=0.8,max_value=0.94)
            alpha_ab = st.sidebar.number_input('Absorptivity of absorber (alpha_ab)',format="%.4f",min_value=0.9,max_value=0.98)
            st.sidebar.markdown("<h1 style='color: blue;'>Geometric parameters</h1>", unsafe_allow_html=True)
            N = st.sidebar.number_input('Number of tubes (N)', min_value=1,max_value=25)
            theta = st.sidebar.number_input('Angle of radiation (theta)',format="%.4f",min_value=0.0,max_value=60.0)
            D_o = st.sidebar.number_input('Outer diameter of tube (D_o)',format="%.4f",min_value=47.0,max_value=102.0)
            L_c = st.sidebar.number_input('Length of the condenser (L_c)',format="%.4f",min_value=70.0,max_value=400.0)
              
            input_data_ETA_water = pd.DataFrame({
            'alpha': [alpha],   
            'tau_g': [tau_g],
            'I': [I],
            'T_in': [T_in],
            'alpha_ab': [alpha_ab],
            'm_dot': [m_dot],
            'T_amb': [T_amb],
            'L_c': [L_c],  
            'theta': [theta],
            'N': [N],
            'D_o': [D_o]
            })
            progress_bar=st.progress(0)
          
            loaded_model_eta = xgb.XGBRegressor(n_estimators=13000)
            loaded_model_eta.load_model('xgbFwatereta.json')

            with open('XwaterF_eta.pkl', 'rb') as f:
                x_scaler_eta = pickle.load(f)
            with open('YwaterF_eta.pkl', 'rb') as f:
                scaler_y_eta = pickle.load(f)
        #............................
            progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models

            input_data_scaled_eta = x_scaler_eta.transform(input_data_ETA_water.values.reshape(1, -1))
            y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
            y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))
#..................
            progress_bar.progress(66)
    
    # Display prediction results
            st.subheader("Predicted Outputs:")
            st.write("Predicted efficiency (Efficiency):", y_pred_eta[0],format="%.4f")  
            progress_bar.progress(100)
            
    else:  #manifold fluid is air&&&&&&&&&&&&&&&&&&&&&&&&&&&7%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        if predicted_parameter=='Temperature': 
            
            fluid_temps = {
            'Acetone': 0.000000169,
            'Methanol':9.23E-08,
            'Ethanol': 7.13E-08,
            'Water': 0.000000169
            }
            fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
            alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary
            st.sidebar.markdown("<h1 style='color: green;'>Environmental parameters</h1>", unsafe_allow_html=True)
            m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.5f",min_value=0.001,max_value=0.05)
            T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f",min_value=10.0,max_value=90.0)
            st.sidebar.markdown("<h1 style='color: red;'>Radiation parameters</h1>", unsafe_allow_html=True)
            I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f",min_value=100.0,max_value=1200.0)
            tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)',format="%.4f",min_value=0.8,max_value=0.94)
            epsilon_g = st.sidebar.number_input('Emissivity of glass (epsilon_g)',format="%.4f",min_value=0.02,max_value=0.07)
            st.sidebar.markdown("<h1 style='color: blue;'>Geometric parameters</h1>", unsafe_allow_html=True)
            N = st.sidebar.number_input('Number of tubes (N)',min_value=1,max_value=25)
            D_o = st.sidebar.number_input('Outer diameter of tube (D_o)',format="%.4f",min_value=47.0,max_value=102.0)
            L_c = st.sidebar.number_input('Length of the condenser (L_c)',format="%.4f",min_value=70.0,max_value=400.0)
            theta = st.sidebar.number_input('Angle of radiation (theta)',format="%.4f",min_value=0.0,max_value=60.0)
            L_tube = st.sidebar.number_input('Length of the tube (L_tube)',format="%.4f",min_value=1000.0,max_value=2100.0)
            
            input_data_T_air = pd.DataFrame({
            'alpha': [alpha],  
            'm_dot': [m_dot],
            'N': [N],
            'I': [I],
            'T_in': [T_in],
            'D_o': [D_o],
            'theta': [theta],
            'L_c': [L_c],
            'L_tube': [L_tube],  
            'epsilon_g': [epsilon_g],
            'tau_g': [tau_g],
            })
            progress_bar=st.progress(0)
    
            loaded_model_T = xgb.XGBRegressor(n_estimators=13000)
            loaded_model_T.load_model('xgbFairT.json')

            with open('XairF_T.pkl', 'rb') as f:
                x_scaler_T = pickle.load(f)
            with open('YairF_T.pkl', 'rb') as f:
                scaler_y_T = pickle.load(f)
        #............................
            progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models

            input_data_scaled_T = x_scaler_T.transform(input_data_T_air.values.reshape(1, -1))
            y_pred_scaled_T = loaded_model_T.predict(input_data_scaled_T)
            y_pred_T = scaler_y_T.inverse_transform(y_pred_scaled_T.reshape(-1, 1))
#..................
            progress_bar.progress(66)
    
    # Display prediction results
            st.subheader("Predicted Outputs:")
            st.write("Predicted outlet Temperature (T_exit):", y_pred_T[0],format="%.4f")  
            progress_bar.progress(100)
   #.........................

        else:#target is air efficiency*********************
  
            fluid_temps = {
            'Acetone': 0.000000169,
            'Methanol':9.23E-08,
            'Ethanol': 7.13E-08,
            'Water': 0.000000169
            }
            fluid_name = st.sidebar.radio('Heatpipe Fluid', list(fluid_temps.keys()))
            alpha = fluid_temps[fluid_name] # Get saturation temperature from dictionary
            st.sidebar.markdown("<h1 style='color: green;'>Environmental parameters</h1>", unsafe_allow_html=True)
            m_dot = st.sidebar.number_input('Mass flow rate of manifold fluid (m_dot)',format="%.5f",min_value=0.001,max_value=0.05)
            T_in = st.sidebar.number_input('Inlet temperature (T_in)',format="%.4f",min_value=10.0,max_value=90.0)
            st.sidebar.markdown("<h1 style='color: red;'>Radiation parameters</h1>", unsafe_allow_html=True)
            I = st.sidebar.number_input('Solar irradiance (I)',format="%.4f",min_value=100.0,max_value=1200.0)
            tau_g = st.sidebar.number_input('Transmissivity of glass (tau_g)',format="%.4f",min_value=0.8,max_value=0.94)
            epsilon_g = st.sidebar.number_input('Emissivity of glass (epsilon_g)',format="%.4f",min_value=0.02,max_value=0.07)
            st.sidebar.markdown("<h1 style='color: blue;'>Geometric parameters</h1>", unsafe_allow_html=True)
            N = st.sidebar.number_input('Number of tubes (N)', min_value=1,max_value=25)
            L_tube = st.sidebar.number_input('Length of the tube (L_tube)',format="%.4f",min_value=1000.0,max_value=2100.0)
            L_c = st.sidebar.number_input('Length of the condenser (L_c)',format="%.4f",min_value=70.0,max_value=400.0)
            D_o = st.sidebar.number_input('Outer diameter of tube (D_o)',format="%.4f",min_value=47.0,max_value=102.0)
            D_H = st.sidebar.number_input('Hydraulic diameter of tube (D_H)',format="%.4f",min_value=20.0,max_value=270.0)
           
            
            
            input_data_ETA_air = pd.DataFrame({
            'alpha': [alpha],  
            'm_dot': [m_dot],
            'L_c': [L_c],
            'tau_g': [tau_g],
            'D_o': [D_o],
            'L_tube': [L_tube],
            'N': [N],
            'T_in': [T_in],
            'D_H': [D_H],  
            'epsilon_g': [epsilon_g],
            'I': [I],
            })
            progress_bar=st.progress(0)
           
            loaded_model_eta = xgb.XGBRegressor(n_estimators=13000)
            loaded_model_eta.load_model('xgbFaireta.json')
            
            with open('XairF_eta.pkl', 'rb') as f:
                x_scaler_eta = pickle.load(f)
            with open('YairF_eta.pkl', 'rb') as f:
                scaler_y_eta = pickle.load(f)
        #............................
            progress_bar.progress(33)        #1/3 of complication

    # Make predictions for both models

            input_data_scaled_eta = x_scaler_eta.transform(input_data_ETA_air.values.reshape(1, -1))
            y_pred_scaled_eta = loaded_model_eta.predict(input_data_scaled_eta)
            y_pred_eta = scaler_y_eta.inverse_transform(y_pred_scaled_eta.reshape(-1, 1))
#..................
            progress_bar.progress(66)
    
    # Display prediction results
            st.subheader("Predicted Outputs:")
            st.write("Predicted efficiency (Efficiency):", y_pred_eta[0],format="%.4f")  
            progress_bar.progress(100)
        
        