# ImmoEliza - Streamlit App

ImmoEliza is a machine learning project that provides accurate real estate price estimations. This `README` focuses on the Streamlit app, which allows users to input property details and receive a price estimation. The app builds on the previous stages of the project, including data cleaning, preprocessing, and model training.

## Overview

The Streamlit app is the final stage of the ImmoEliza project. It provides a user-friendly interface for inputting property details and receiving real-time price estimations. The app leverages pre-trained machine learning models to predict property prices based on user input.

## Structure

The Streamlit app is constructed as follows:

- **`deployment.py`**: The main script that runs the Streamlit application. It includes:
  - **User Input Form**: Collects property details through a sidebar. The form dynamically adjusts based on whether the user selects "House" or "Apartment", displaying the relevant input fields for each type.
  - **Data Processing**: Processes and transforms user input into a format suitable for prediction. Missing values are handled by using the mode values from the datasets.
  - **Model Prediction**: Loads the appropriate trained model based on the property type selected by the user:
    - `xgb_model_house.pkl` for houses
    - `xgb_model_apt.pkl` for apartments
  - **Results Display**: Shows a summary of the property details and the estimated price.

## Code Explanation

- **Styling**: Custom CSS is used to adjust the appearance of the app, including setting the maximum width and adjusting the sidebar width.
- **Data Loading**: The app reads datasets for houses, apartments, and localities to facilitate user input processing.
- **Conversion Dictionaries**: Mappings for converting categorical values to numerical ones, used for encoding user input.
- **User Input**: Collects various property details from the user. The input form adjusts to include fields relevant to the selected property type. For example, fields for the surface of the plot and number of facades are shown only if "House" is selected.
- **Model Loading and Prediction**: Depending on the property type, the app loads the corresponding XGBoost model and makes a prediction.
- **Results Presentation**: Displays the property details and the estimated price in a formatted manner.
## Deployment

To deploy the Streamlit app:

1. **Install Dependencies**: Ensure you have the necessary Python packages. You can install them using:

    ```bash
    pip install numpy pandas streamlit xgboost
    ```

2. **Run the App**: Use the following command to start the Streamlit server:

    ```bash
    streamlit run streamlit_app.py
    ```

3. **Access the App**: Once the server is running, you can access the app in your web browser at:

    [ImmoEliza - Real Estate Price Estimator](https://immo-eliza-belgium.streamlit.app/)

## Context

The Streamlit app is built on the results from previous stages of the ImmoEliza project:

- **Data Cleaning**: Preparing the raw data for analysis.
- **Preprocessing and Modeling**: Transforming the data and training machine learning models for price prediction.

These stages ensure that the data used by the Streamlit app is clean and the models are accurate.

______________________________

Thank you for using ImmoEliza! 
