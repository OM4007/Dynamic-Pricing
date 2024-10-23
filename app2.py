import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Title of the Streamlit app
st.title("Dynamic Pricing Model")

# Load the dynamic pricing data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("dynamic_pricing.csv")
        return data
    except FileNotFoundError:
        st.error("dynamic_pricing.csv file not found. Please ensure the file is in the correct directory.")
        return None

data = load_data()
if data is not None:
    # Data preprocessing and calculation of multipliers
    def preprocess_data(data):
        # Calculate demand and supply multipliers based on percentiles
        high_demand_percentile = 75
        low_demand_percentile = 25

        data['demand_multiplier'] = np.where(
            data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
            data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
            data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile)
        )

        high_supply_percentile = 75
        low_supply_percentile = 25

        data['supply_multiplier'] = np.where(
            data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'], low_supply_percentile),
            np.percentile(data['Number_of_Drivers'], high_supply_percentile) / data['Number_of_Drivers'],
            np.percentile(data['Number_of_Drivers'], low_supply_percentile) / data['Number_of_Drivers']
        )

        # Define thresholds and calculate adjusted ride cost
        demand_threshold_low = 0.8
        supply_threshold_high = 0.8

        data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride'] * (
            np.maximum(data['demand_multiplier'], demand_threshold_low) *
            np.maximum(data['supply_multiplier'], supply_threshold_high)
        )

        # Handle missing values for model input
        numeric_features = data.select_dtypes(include=['float', 'int']).columns
        data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

        categorical_features = data.select_dtypes(include=['object']).columns
        data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

        # Convert 'Vehicle_Type' to one-hot encoding for model training
        data = pd.get_dummies(data, columns=['Vehicle_Type'], drop_first=True)
        return data

    # Preprocess data
    data = preprocess_data(data)
    st.dataframe(data.head())

    # Function to train the Random Forest model
    def train_model(data):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Select relevant features for training
        if 'adjusted_ride_cost' in data.columns:
            X = data[['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration', 'Vehicle_Type_Premium']]
            y = data['adjusted_ride_cost']
            model.fit(X, y)
            return model
        else:
            st.error("Column 'adjusted_ride_cost' not found in the dataset.")
            return None

    # Model training
    model = train_model(data)

    # Function to map user input to the numeric Vehicle_Type
    def get_vehicle_type_numeric(vehicle_type):
        vehicle_type_mapping = {"Premium": 1, "Economy": 0}
        return vehicle_type_mapping.get(vehicle_type)

    # Predict function for user input
    def predict_price(model, number_of_riders, number_of_drivers, vehicle_type, expected_ride_duration):
        vehicle_type_numeric = get_vehicle_type_numeric(vehicle_type)
        input_data = np.array([[number_of_riders, number_of_drivers, expected_ride_duration, vehicle_type_numeric]])
        predicted_price = model.predict(input_data)
        return predicted_price[0]

    # Streamlit UI for user input
    st.sidebar.header("User Input Parameters")
    user_number_of_riders = st.sidebar.slider("Number of Riders", min_value=1, max_value=100, value=50)
    user_number_of_drivers = st.sidebar.slider("Number of Drivers", min_value=1, max_value=100, value=25)
    user_vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Economy", "Premium"])
    expected_ride_duration = st.sidebar.slider("Expected Ride Duration (minutes)", min_value=5, max_value=60, value=30)

    # Predict ride cost based on user input
    if model:
        predicted_price = predict_price(model, user_number_of_riders, user_number_of_drivers, user_vehicle_type, expected_ride_duration)
        st.write(f"Predicted ride cost: ${predicted_price:.2f}")

    # Visualization for actual vs predicted values (test data)
    st.header("Actual vs Predicted Values")

    @st.cache_data
    def load_test_data():
        try:
            test_data = pd.read_csv("test_data.csv")
            return test_data
        except FileNotFoundError:
            st.error("test_data.csv file not found. Please ensure the file is in the correct directory.")
            return None

    test_data = load_test_data()
    if test_data is not None:
        # Preprocess the test data similarly
        test_data = preprocess_data(test_data)
        X_test = test_data[['Number_of_Riders', 'Number_of_Drivers', 'Expected_Ride_Duration', 'Vehicle_Type_Premium']]
        y_test = test_data['adjusted_ride_cost']

        # Predict using the model
        y_pred = model.predict(X_test)

        # Plot actual vs predicted values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs Predicted'))
        fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))

        fig.update_layout(title='Actual vs Predicted Values', xaxis_title='Actual Values', yaxis_title='Predicted Values', showlegend=True)
        st.plotly_chart(fig)

'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

st.title("Dynamic Pricing Model")

# Load data
data = pd.read_csv("dynamic_pricing.csv")

# Calculate demand_multiplier based on percentile for high and low demand
high_demand_percentile = 75
low_demand_percentile = 25

data['demand_multiplier'] = np.where(data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                     data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                     data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile))

# Calculate supply_multiplier based on percentile for high and low supply
high_supply_percentile = 75
low_supply_percentile = 25

data['supply_multiplier'] = np.where(data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'], 
                                                                               low_supply_percentile),
                                     np.percentile(data['Number_of_Drivers'],
                                                   high_supply_percentile) / data['Number_of_Drivers'],
                                     np.percentile(data['Number_of_Drivers'], 
                                                   low_supply_percentile) / data['Number_of_Drivers'])

# Define price adjustment factors for high and low demand/supply
demand_threshold_high = 1.2  # Higher demand threshold
demand_threshold_low = 0.8  # Lower demand threshold
supply_threshold_high = 0.8  # Higher supply threshold
supply_threshold_low = 1.2  # Lower supply threshold

# Calculate adjusted_ride_cost for dynamic pricing
data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride'] * (
    np.maximum(data['demand_multiplier'], 
               demand_threshold_low) *
    np.maximum(data['supply_multiplier'], 
               supply_threshold_high)
    )


# Calculate the profit percentage for each ride
data['profit_percentage'] = ((data['adjusted_ride_cost'] - data['Historical_Cost_of_Ride']) / data['Historical_Cost_of_Ride']) * 100
# Identify profitable rides where profit percentage is positive
profitable_rides = data[data['profit_percentage'] > 0]

# Identify loss rides where profit percentage is negative
loss_rides = data[data['profit_percentage'] < 0]

# Calculate the count of profitable and loss rides
profitable_count = len(profitable_rides)
loss_count = len(loss_rides)

# Create a donut chart to show the distribution of profitable and loss rides
labels = ['Profitable Rides', 'Loss Rides']
values = [profitable_count, loss_count]


# Preprocessing function
def preprocess_data(data):
    # Handling missing values
    numeric_features = data.select_dtypes(include=['float', 'int']).columns
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())
    
    categorical_features = data.select_dtypes(include=['object']).columns
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])
    
    # Convert categorical feature "Vehicle_Type" to one-hot encoded columns
    data = pd.get_dummies(data, columns=["Vehicle_Type"], drop_first=True)
    
    return data
new_data = preprocess_data(data)
st.dataframe(new_data)
# Train a Random Forest model
def train_model(data):
    model = RandomForestRegressor()
    x = data[["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type", "Expected_Ride_Duration"]]
    # Check if the column exists before using it
    if "adjusted_ride_cost" in data.columns:
        y = data["adjusted_ride_cost"]
        model.fit(x, y)
        return model
    else:
        st.write("Column 'adjusted_ride_cost' not found in the dataset.")
        return None
    
def get_vehicle_type_numeric(Vehicle_Type):
    vehicle_type_mapping = {
        "Premium": 1,
        "Economy": 0
    }
    vehicle_type_numeric = vehicle_type_mapping.get(Vehicle_Type)
    return vehicle_type_numeric


# User input and prediction function
def predict_price(model, number_of_riders, number_of_drivers, Vehicle_Type, expected_ride_duration):
    vehicle_type_numeric = get_vehicle_type_numeric(Vehicle_Type)
    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, expected_ride_duration]])
    predicted_price = model.predict(input_data)
    return predicted_price

# Streamlit UI
user_number_of_riders = st.slider("Number of Riders", min_value=1, max_value=100, value=50)
user_number_of_drivers = st.slider("Number of Drivers", min_value=1, max_value=100, value=25)
user_vehicle_type = st.selectbox("Vehicle Type", ["Economy", "Premium"])
expected_ride_duration = st.slider("Expected Ride Duration (minutes)", min_value=5, max_value=60, value=30)

# Preprocess data and train the model
data = preprocess_data(data)
model = train_model(data)

# Predict using user inputs
predicted_price = predict_price(model, user_number_of_riders, user_number_of_drivers, user_vehicle_type, expected_ride_duration)
st.write(f"Predicted price: ${predicted_price:.2f}")

# Visualization
st.header("Actual vs Predicted Values")

# Load test set for visualization
test_data = pd.read_csv("test_data.csv")
x_test = test_data[["Number_of_Riders", "Number_of_Drivers", "Vehicle_Type", "Expected_Ride_Duration"]]
y_test = test_data["adjusted_ride_cost"]

# Predict on the test set
y_pred = model.predict(x_test)

# Create a scatter plot with actual vs predicted values
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_test,
    y=y_pred,
    mode='markers',
    name='Actual vs Predicted'
))

# Add a line representing the ideal case
fig.add_trace(go.Scatter(
    x=[min(y_test), max(y_test)],
    y=[min(y_test), max(y_test)],
    mode='lines',
    name='Ideal',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title='Actual vs Predicted Values',
    xaxis_title='Actual Values',
    yaxis_title='Predicted Values',
    showlegend=True,
)

st.plotly_chart(fig)
'''