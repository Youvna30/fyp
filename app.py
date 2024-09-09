import streamlit as st
import joblib
import pandas as pd

# Function to load the model
def load_model(model_path):
    """Load the trained model from a file."""
    return joblib.load(model_path)

# Load the pre-trained model
model = load_model('random_forest_model.pkl')

# Streamlit app
st.title('Traffic Density Prediction')

# Create two columns
col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        'Select City',
        ['SolarisVille', 'AquaCity', 'Neuroburg', 'Ecoopolis']
    )

    vehicle_type = st.selectbox(
        'Select Vehicle Type',
        ['Autonomous Vehicle', 'Drone', 'Flying Car', 'Car']
    )

    weather = st.selectbox(
        'Select Weather',
        ['Solar Flare', 'Snowy', 'Electromagnetic Storm', 'Clear', 'Rainy']
    )

    economic_condition = st.selectbox(
        'Select Economic Condition',
        ['Booming', 'Recession', 'Stable']
    )

    hour_of_day = st.slider(
        'Select Hour Of Day',
        min_value=0,
        max_value=23
    )

with col2:
    day_of_week = st.selectbox(
        'Select Day Of Week',
        ['Wednesday', 'Thursday', 'Tuesday', 'Saturday', 'Monday', 'Sunday', 'Friday']
    )

    speed = st.number_input(
        'Enter Speed',
        min_value=6.6934,  # Based on min and max values from your data
        max_value=163.0886,
        step=0.1
    )

    is_peak_hour = st.selectbox(
        'Is it Peak Hour?',
        [0, 1]
    )

    random_event_occurred = st.selectbox(
        'Did a Random Event Occur?',
        [0, 1]
    )

    energy_consumption = st.number_input(
        'Enter Energy Consumption',
        min_value=4.9296,  # Based on min and max values from your data
        max_value=189.9489,
        step=0.1
    )

# Button to make prediction
if st.button('Predict Traffic Density'):
    # Prepare input data
    input_data = {
        'City': city,
        'Vehicle Type': vehicle_type,
        'Weather': weather,
        'Economic Condition': economic_condition,
        'Day Of Week': day_of_week,
        'Hour Of Day': hour_of_day,
        'Speed': speed,
        'Is Peak Hour': is_peak_hour,
        'Random Event Occurred': random_event_occurred,
        'Energy Consumption': energy_consumption
    }

    # Make prediction
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    
    # Display prediction
    st.success(f'Predicted Traffic Density: {prediction[0]:.4f}')
