import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load the pre-trained models and scaler
svm = load('milk_svm_classifier.joblib')
rf = load('milk_random_forest_classifier.joblib')
scaler = load('milk_scaler.joblib')  # Assuming you have saved the scaler object as 'milk_scaler.joblib'

# Create a Streamlit app
def main():
    st.title('Milk Quality Prediction App')

    # User input parameters form
    st.sidebar.header('Input Milk Quality Parameters')
    st.sidebar.write('Enter parameters to predict milk quality:')

    def user_input_features():
        pH = st.sidebar.number_input('pH Level', min_value=0.0, max_value=14.0, value=6.7)
        Temperature = st.sidebar.number_input('Temperature (°C)', min_value=0.0, value=25.0)
        Taste = st.sidebar.radio('Taste', options=[0, 1])
        Odor = st.sidebar.radio('Odor', options=[0, 1])
        Fat = st.sidebar.radio('Fat (%)', options=[0, 1])
        Turbidity = st.sidebar.radio('Turbidity (NTU)', options=[0, 1])
        Colour = st.sidebar.radio('Colour', options=['Normal', 'Abnormal'])
        
        # Convert categorical input to numerical data
        Colour_mapping = {'Normal': 0, 'Abnormal': 1}
        
        data = {'pH': pH,
                'Temperature': Temperature,
                'Taste': Taste,
                'Odor': Odor,
                'Fat': Fat,
                'Turbidity': Turbidity,
                'Colour': Colour_mapping[Colour]}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Scale numerical features
    numerical_features = input_df[['pH', 'Temperature', 'Fat', 'Turbidity']]
    input_df_scaled = scaler.transform(numerical_features)
    input_df[['pH', 'Temperature', 'Fat', 'Turbidity']] = input_df_scaled

    # Prediction on user input
    if st.sidebar.button('Predict Milk Quality'):
        prediction_svm = svm.predict(input_df)
        prediction_rf = rf.predict(input_df)
        
        st.subheader('Prediction Results')
        result_svm = 'Good Quality' if prediction_svm[0] == 1 else 'Poor Quality'
        result_rf = 'Good Quality' if prediction_rf[0] == 1 else 'Poor Quality'
        
        st.write('SVM Classifier Prediction: **{}**'.format(result_svm))
        st.write('Random Forest Classifier Prediction: **{}**'.format(result_rf))

if __name__ == '__main__':
    main()