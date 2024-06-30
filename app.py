import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from io import BytesIO
import pickle
import base64

# Navigation bar
st.sidebar.title(":violet[AutoML Wizard]")
st.sidebar.markdown('''(experimental)''')
st.sidebar.image("wizard_icon.png")
st.toast("Open Side bar on Left to Begin)
sections = ["Overview","Upload Data", "Data Overview", "Model Building and Training", "Model Evaluation", "Exploratory Data Analysis (EDA)", "Download Model", "Make Predictions"]
selection = st.sidebar.radio("Contents", sections)
# Global variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'label_encoder' not in st.session_state:
    st.session_state['label_encoder'] = {}

#Information
if selection == "Overview":
    st.title(":violet[Welcome to AutoML Wizard]")
    st.markdown('''AutoML Wizard is a comprehensive tool designed to streamline the machine learning workflow for :violet[CSV | Excel] Data.
             It allows users to effortlessly upload datasets, perform exploratory data analysis,
             build and train models, and evaluate their performance. Additionally, users can download
             trained models and make predictions, all within an intuitive, user-friendly interface.''')
    st.title(":violet[EDA Used]")
    code_eda = '''Plots:
                  Histogram (histplot)
                  Count Plot (count plot)
                  Correlation Heatmap (correlation heatmap) 
                  Box Plot (box plot)
                  Pairplot (pairplot)'''
    st.code(code_eda, language='python')
    st.title(":violet[Models Used]")
    code_models = '''models = {
                    'Random Forest': RandomForestClassifier(),
                    'Gradient Boosting': GradientBoostingClassifier(),
                    'Logistic Regression': LogisticRegression(max_iter=1000),
                    'Ridge Classifier': RidgeClassifier(),
                    'Support Vector Machine': SVC(probability=True),
                    'Linear SVM': LinearSVC(),
                    'K-Nearest Neighbors': KNeighborsClassifier(),
                    'Decision Tree': DecisionTreeClassifier(),
                    'Naive Bayes (Gaussian)': GaussianNB(),
                    'Naive Bayes (Multinomial)': MultinomialNB(),
                    'MLP Classifier': MLPClassifier(max_iter=1000),
                    'XGBoost': XGBClassifier()  
                }'''
    st.code(code_models, language='python')



# Upload Data
if selection == "Upload Data":
    st.title(":violet[Upload Data]")
    st.markdown('''Upload Data: This section allows you to upload your dataset in either :violet[CSV or Excel] 
                format for analysis and modeling. For a quick start,
                 you can also upload a :violet[pre-loaded sample dataset]. View and verify your data instantly 
                after uploading.''')
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

    if st.button("Upload Sample Heart Disease Data"):
        try:
            data = pd.read_csv("heart.csv")
            st.session_state['data'] = data
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error: {e}")

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state['data'] = data
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error: {e}")

# Data Overview
elif selection == "Data Overview":
    st.title(":violet[Data Overview]")
    st.markdown('''Data Overview: This section provides a summary of your dataset,
                including :violet[basic statistics], data types, and any missing values.
                It helps you understand the structure and quality of your data.''')
    if st.session_state['data'] is not None:
        data = st.session_state['data']
        st.subheader(":violet[Basic Statistics]")
        st.write(data.describe())
        st.subheader(":violet[Data Types and Missing Values]")
        st.write(data.dtypes)
        st.write(data.isnull().sum())
    else:
        st.error("Please upload a dataset first.")

# Model Building and Training
elif selection == "Model Building and Training":
    st.title(":violet[Model Building and Training]")
    st.markdown('''Model Building and Training: This section guides you through selecting
                the :violet[target variable] and features for your model. It offers a range of
                machine learning algorithms to choose from, and trains models on your data.''')
    if st.session_state['data'] is not None:
        data = st.session_state['data']
        
        st.subheader(":violet[Select Target Variable]")
        target = st.selectbox("Select target variable", data.columns)
        
        st.subheader(":violet[Select Features]")
        features = st.multiselect("Select features", data.columns, default=[col for col in data.columns if col != target])
        
        if st.button("Train Models"):
            if target and features:
                X = data[features]
                y = data[target]
                
                if y.dtype == 'object':
                    st.session_state['label_encoder'][target] = LabelEncoder()
                    y = st.session_state['label_encoder'][target].fit_transform(y)
                
                for col in X.select_dtypes(include=['object']).columns:
                    st.session_state['label_encoder'][col] = LabelEncoder()
                    X[col] = st.session_state['label_encoder'][col].fit_transform(X[col])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                models = {
                    'Random Forest': RandomForestClassifier(),
                    'Gradient Boosting': GradientBoostingClassifier(),
                    'Logistic Regression': LogisticRegression(max_iter=1000),
                    'Ridge Classifier': RidgeClassifier(),
                    'Support Vector Machine': SVC(probability=True),
                    'Linear SVM': LinearSVC(),
                    'K-Nearest Neighbors': KNeighborsClassifier(),
                    'Decision Tree': DecisionTreeClassifier(),
                    'Naive Bayes (Gaussian)': GaussianNB(),
                    'Naive Bayes (Multinomial)': MultinomialNB(),
                    'MLP Classifier': MLPClassifier(max_iter=1000),
                    'XGBoost': XGBClassifier()  
                }
                
                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    st.session_state['models'][model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'roc_curve': None  # Removing ROC curve for now to avoid multi-class issue
                    }
                
                st.write("Models trained successfully.")
    else:
        st.error("Please upload a dataset first.")

# Model Evaluation
elif selection == "Model Evaluation":
    st.title(":violet[Model Evaluation]")
    st.markdown('''Model Evaluation: This section presents the performance metrics of
                the trained models, including :violet[accuracy, precision, recall, and F1 score].
                It also provides confusion matrices to evaluate classification performance.''')
    if st.session_state['models']:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        results = {metric: [] for metric in metrics}
        results['model'] = []

        for model_name, model_info in st.session_state['models'].items():
            results['model'].append(model_name)
            for metric in metrics:
                results[metric].append(model_info[metric])

        results_df = pd.DataFrame(results)
        st.write(results_df)

        for model_name, model_info in st.session_state['models'].items():
            st.subheader(f"Confusion Matrix for {model_name}")
            fig, ax = plt.subplots()
            sns.heatmap(model_info['confusion_matrix'], annot=True, fmt='d', cmap='Purples', ax=ax)
            st.pyplot(fig)
    else:
        st.error("Please build and train a model first.")

# Exploratory Data Analysis (EDA)
elif selection == "Exploratory Data Analysis (EDA)":
    st.title(":violet[Exploratory Data Analysis (EDA)]")
    st.markdown('''Exploratory Data Analysis: This section allows you to visualize and explore your data
                through various plots like :violet[histograms, count plots, correlation heatmaps, box plots, and pair plots].
                Gain insights into the distribution and relationships within your data.''')
    
    if st.session_state['data'] is not None:
        data = st.session_state['data']
        
        st.subheader(":violet[Select EDA Plot Type]")
        eda_options = ["Histogram (histplot)", "Count Plot (count plot)", "Correlation Heatmap (correlation heatmap)", 
                       "Box Plot (box plot)", "Pairplot (pairplot)"]
        selected_option = st.selectbox("Select plot type", eda_options)
        
        if selected_option == "Histogram (histplot)":
            st.write("Select Features for Histogram")
            feature_x = st.selectbox("Select feature for x-axis", data.columns)
            feature_y = st.selectbox("Select feature for y-axis", data.columns)
            
            fig, ax = plt.subplots()
            sns.histplot(data=data, x=feature_x, y=feature_y, kde=True, ax=ax,palette='Purples')
            plt.title(f'Joint Histogram of {feature_x} and {feature_y}')
            st.pyplot(fig)

        
        elif selected_option == "Count Plot (count plot)":
            st.write("Select Feature for Count Plot")
            feature_x = st.selectbox("Select feature", data.columns)
            
            if data[feature_x].dtype == 'object':
                # For categorical variables, ensure they are properly encoded for counting
                fig, ax = plt.subplots()
                sns.countplot(data[feature_x], ax=ax,palette="Purples")
                plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility if needed
                st.pyplot(fig)
            else:
                # For numerical variables, convert to categorical or adjust as needed
                st.error("Selected feature is not categorical. Please choose a categorical feature.")

        
        elif selected_option == "Correlation Heatmap (correlation heatmap)":
            st.write("Select Features for Correlation Heatmap")
            selected_features = st.multiselect("Select features", data.columns)
            
            if len(selected_features) > 0:
                # Encode categorical columns using one-hot encoding
                categorical_cols = [col for col in selected_features if data[col].dtype == 'object']
                if categorical_cols:
                    encoded_data = pd.get_dummies(data[selected_features], columns=categorical_cols)
                else:
                    encoded_data = data[selected_features].copy()  # No categorical columns to encode
                
                # Calculate correlation matrix
                correlation_matrix = encoded_data.corr()
                
                # Plot correlation heatmap
                fig, ax = plt.subplots()
                sns.heatmap(correlation_matrix, annot=True, cmap='Purples', ax=ax)
                plt.title('Correlation Heatmap')
                st.pyplot(fig)
            else:
                st.info("Select at least one feature.")

        
        elif selected_option == "Box Plot (box plot)":
            st.write("Select Feature for Box Plot")
            feature_x = st.selectbox("Select feature", data.columns)
            
            fig, ax = plt.subplots()
            sns.boxplot(x=data[feature_x], ax=ax, palette="Purples")
            st.pyplot(fig)
        
        elif selected_option == "Pairplot (pairplot)":
            st.write("Select Features for Pairplot")
            feature_x = st.selectbox("Select feature for x-axis", data.columns)
            feature_y = st.selectbox("Select feature for y-axis", data.columns)
            
            fig = sns.pairplot(data[[feature_x, feature_y]], palette="Purples")
            st.pyplot(fig)
    
    else:
        st.error("Please upload a dataset first.")

# Download Model
elif selection == "Download Model":
    st.title(":violet[Download Model]")
    st.markdown('''Download Model: This section allows you to download the trained models
                for deployment or further analysis. Select the desired :violet[model] and
                click the download button to get the model in a :violet[pickle] file.''')
    if st.session_state['models']:
        model_name = st.selectbox("Select model to download", list(st.session_state['models'].keys()))
        model = st.session_state['models'][model_name]['model']
        
        if st.button("Download Model"):
            buffer = BytesIO()
            pickle.dump(model, buffer)
            buffer.seek(0)
            
            b64 = base64.b64encode(buffer.read()).decode()
            href = f'<a href="data:file/zip;base64,{b64}" download="{model_name}.pkl">Download {model_name} Model</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("Please build and train a model first.")

# Make Predictions
elif selection == "Make Predictions":
    st.title(":violet[Make Predictions]")
    st.markdown('''Make Predictions: This section allows you to make predictions using the
                trained models. Enter the input data and select a :violet[trained model] to
                see the predictions based on the provided inputs.''')
    if st.session_state['models']:
        model_name = st.selectbox("Select model for prediction", list(st.session_state['models'].keys()))
        model_info = st.session_state['models'][model_name]

        st.write("Enter Data for Prediction")
        input_data = {}
        for col in st.session_state['data'].columns:
            input_data[col] = st.text_input(f"Enter {col}")
        
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    if col in st.session_state['label_encoder']:
                        input_df[col] = st.session_state['label_encoder'][col].transform(input_df[col])
            
            prediction = model_info['model'].predict(input_df)
            
            st.write("Prediction Result")
            st.write(prediction)
    else:
        st.error("Please build and train a model first.")
