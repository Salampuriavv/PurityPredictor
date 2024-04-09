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
    st.title('Model Testing with Uploaded Data')

    # File upload prompts
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