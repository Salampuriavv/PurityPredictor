import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

# Load the pre-trained models (make sure the .joblib files are in the same directory or provide the correct path)
Rfc = load('random_forest_classifier.joblib')
xgb = load('xgb_classifier.joblib')

# Create a Streamlit app
def main():
    st.title('Water Potability Prediction App')

    # User input parameters form
    st.sidebar.header('Input Water Quality Parameters')
    st.sidebar.write('Enter parameters to predict potability:')

    def user_input_features():
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

    input_df = user_input_features()

    # Prediction on user input
    if st.sidebar.button('Predict Potability'):
        prediction_rfc = Rfc.predict(input_df)
        prediction_xgb = xgb.predict(input_df)
        
        st.subheader('Prediction Results')
        result_rfc = 'Potable' if prediction_rfc[0] == 1 else 'Not Potable'
        result_xgb = 'Potable' if prediction_xgb[0] == 1 else 'Not Potable'
        
        st.write('Random Forest Classifier Prediction: **{}**'.format(result_rfc))
        st.write('XGB Classifier Prediction: **{}**'.format(result_xgb))

    # File upload prompts for model testing
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