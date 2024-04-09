import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained models and scaler for milk
svm = load('milk_svm_classifier.joblib')
rf = load('milk_random_forest_classifier.joblib')
milk_scaler = load('milk_scaler.joblib')

# Load the pre-trained models for water
Rfc = load('random_forest_classifier.joblib')
xgb = load('xgb_classifier.joblib')

# Create a Streamlit app
def main():
    st.title('Liquid Quality Prediction App')

    # Option to select liquid type
    liquid_type = st.sidebar.radio('Select Liquid Type', options=['Milk', 'Water'])

    if liquid_type == 'Milk':
        # User input parameters form for milk
        st.sidebar.header('Input Milk Quality Parameters')
        st.sidebar.write('Enter parameters to predict milk quality:')
        input_df = user_input_features_milk()

        # Prediction on user input for milk
        if st.sidebar.button('Predict Milk Quality'):
            predict_milk_quality(input_df)

    elif liquid_type == 'Water':
        # User input parameters form for water
        st.sidebar.header('Input Water Quality Parameters')
        st.sidebar.write('Enter parameters to predict water potability:')
        input_df = user_input_features_water()

        # Prediction on user input for water
        if st.sidebar.button('Predict Water Potability'):
            predict_water_potability(input_df)

        # File upload prompts for model testing
        model_testing_section()

def user_input_features_milk():
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

def predict_milk_quality(input_df):
    # Scale numerical features
    numerical_features = input_df[['pH', 'Temperature', 'Fat', 'Turbidity']]
    input_df_scaled = milk_scaler.transform(numerical_features)
    input_df[['pH', 'Temperature', 'Fat', 'Turbidity']] = input_df_scaled

    # Prediction on user input
    prediction_svm = svm.predict(input_df)
    prediction_rf = rf.predict(input_df)
    
    st.subheader('Milk Quality Prediction Results')
    result_svm = 'Good Quality' if prediction_svm[0] == 1 else 'Poor Quality'
    result_rf = 'Good Quality' if prediction_rf[0] == 1 else 'Poor Quality'
    
    st.write('SVM Classifier Prediction: **{}**'.format(result_svm))
    st.write('Random Forest Classifier Prediction: **{}**'.format(result_rf))

def user_input_features_water():
    ph = st.sidebar.number_input('pH Level', min_value=0.0, max_value=14.0, value=7.0)
    Hardness = st.sidebar.number_input('Hardness', min_value=0.0, value=100.0)
    Solids = st.sidebar.number_input('Solids', min_value=0.0, value=10000.0)
    Chloramines = st.sidebar.number_input('Chloramines', min_value=0.0, value=7.0)
    Sulfate = st.sidebar.number_input('Sulfate', min_value=0.0, value=300.0)
    Conductivity = st.sidebar.number_input('Conductivity', min_value=0.0, value=400.0)
    Organic_carbon = st.sidebar.number_input('Organic Carbon', min_value=0.0, value=10.0)
    Trihalomethanes = st.sidebar.number_input('Trihalomethanes', min_value=0.0, value=80.0)
    Turbidity = st.sidebar.number_input('Turbidity', min_value=0.0, value=4.0)
    
    data = {'ph': ph,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity}
    features = pd.DataFrame(data, index=[0])
    return features

def predict_water_potability(input_df):
    prediction_rfc = Rfc.predict(input_df)
    prediction_xgb = xgb.predict(input_df)
    
    st.subheader('Water Potability Prediction Results')
    result_rfc = 'Potable' if prediction_rfc[0] == 1 else 'Not Potable'
    result_xgb = 'Potable' if prediction_xgb[0] == 1 else 'Not Potable'
    
    st.write('Random Forest Classifier Prediction: **{}**'.format(result_rfc))
    st.write('XGB Classifier Prediction: **{}**'.format(result_xgb))

def model_testing_section():
    st.header('Model Testing with Uploaded Data')
    X_test_file = st.file_uploader("Upload your X_test CSV", type=["csv"])
    y_test_file = st.file_uploader("Upload your y_test CSV", type=["csv"])

    if X_test_file is not None and y_test_file is not None:
        # Read the uploaded CSV files
        X_test = pd.read_csv(X_test_file)
        y_test = pd.read_csv(y_test_file)

        # Ensure y_test is a 1D array
        y_test = y_test.iloc[:, 0]

        # Random Forest Classifier
        st.write('Random Forest Classifier')
        y_Rfc = Rfc.predict(X_test)
        st.text(classification_report(y_test, y_Rfc))
        st.write('Accuracy:', accuracy_score(y_test, y_Rfc))

        # Plotting heatmap for Random Forest Classifier
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_Rfc), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

        # XGB Classifier
        st.write('XGB Classifier')
        y_xgb = xgb.predict(X_test)
        st.text(classification_report(y_test, y_xgb))
        st.write('Accuracy:', accuracy_score(y_test, y_xgb))

        # Plotting heatmap for XGB Classifier
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_xgb), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

if __name__ == '__main__':
    main()