import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# Function to load CSV data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function for data validation: Missing values and outlier detection
def validate_data(df):
    # Impute missing values (Simple mean imputation)
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Detecting outliers using Isolation Forest
    model = IsolationForest(contamination=0.05)
    df['outliers'] = model.fit_predict(df_imputed)
    return df

# Streamlit app
st.title('CSV Data Validation with Machine Learning')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.write("Data Preview:")
    st.dataframe(df)
    
    if st.button('Validate Data'):
        validated_df = validate_data(df)
        
        st.write("Validated Data:")
        st.dataframe(validated_df)
        
        # Show outliers
        st.write("Outliers Detected:")
        st.dataframe(validated_df[validated_df['outliers'] == -1])

    if st.checkbox("Show Descriptive Statistics"):
        st.write(df.describe())

# Download button for the validated data
if 'validated_df' in locals():
    st.download_button(label="Download Validated Data",
                       data=validated_df.to_csv(index=False),
                       file_name='validated_data.csv')
