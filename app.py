import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache_data
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    df = load_data()

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df)
        st.write(df.columns)

    @st.cache_data
    def split_data(df):
        X = df.drop(columns=['class'])
        y = df['class']
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def plot_metrics(metrics_list, model, X_test, y_test):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names, ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

    X_train, X_test, y_train, y_test = split_data(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01, key='C_svm')
        kernel = st.sidebar.radio("Kernel", ("linear", "rbf"), key='kernel_svm')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma_svm')

        metrics = st.sidebar.multiselect("Which metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify_svm'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test)

    elif classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Inverse Regularization)", 0.01, 10.0, step=0.01, key='C_lr')
        max_iter = st.sidebar.slider("Max Iterations", 100, 1000, step=100, key='max_iter_lr')

        metrics = st.sidebar.multiselect("Which metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify_lr'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test)

    elif classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of Trees", 10, 100, step=10, key='n_estimators_rf')
        max_depth = st.sidebar.slider("Max Depth", 1, 20, step=1, key='max_depth_rf')

        metrics = st.sidebar.multiselect("Which metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify_rf'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, X_test, y_test)

if __name__ == '__main__':
    main()
