import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the Excel data
data = pd.read_excel('love.xlsx', engine='openpyxl')

# Convert Date column to datetime format and create year and month features
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Define target-influential state pairs with the exact case
state_pairs = {
    'Arunachal Pradesh': 'Bihar',
    'Orissa': 'Andhra Pradesh',
    'Jharkhand': 'Bihar',
    'Bihar': 'Jharkhand',
    'West Bengal': 'Jharkhand',
    'Nagaland': 'Arunachal Pradesh',
    'Sikkim': 'Arunachal Pradesh',
    'Assam & Meghalaya': 'Nagaland',
    'Andaman & Nicobar Islands': 'Tamil Nadu',
    'Uttar Pradesh': 'Haryana Delhi & Chandigarh',
    'Uttarakhand': 'Himachal Pradesh',
    'Haryana Delhi & Chandigarh': 'Himachal Pradesh',
    'Punjab': 'Rajasthan',
    'Himachal Pradesh': 'Jammu & Kashmir',
    'Jammu & Kashmir': 'Himachal Pradesh',
    'Rajasthan': 'Punjab',
    'Madhya Pradesh': 'Gujarat',
    'Gujarat': 'Rajasthan',
    'Goa': 'Karnataka',
    'Lakshadweep': 'Kerala',
    'Chhattisgarh': 'Telangana',
    'Andhra Pradesh': 'Tamil Nadu',
    'Telangana': 'Andhra Pradesh',
    'Tamil Nadu': 'Kerala',
    'Karnataka': 'Andhra Pradesh',
    'Kerala': 'Lakshadweep',
    'Maharashtra': 'Gujarat',
}

# Streamlit App
st.title("Praveen's Rainfall Prediction Tool")

# Select target state
target_state = st.selectbox("Select the target state you want to predict rainfall for:", options=list(state_pairs.keys()))
influential_state = state_pairs.get(target_state)

if influential_state:
    st.write(f"Using {influential_state} as the influential state for {target_state}.")

    # Create pairs of consecutive months across years
    data['Next_Month_Value'] = data[target_state].shift(-1)
    data['Next_Month'] = data['Date'].shift(-1).dt.month
    data['Next_Year'] = data['Date'].shift(-1).dt.year

    # Filter only consecutive month pairs
    valid_pairs = data.copy()
    valid_pairs['Is_Consecutive'] = (
        ((valid_pairs['Month'] + 1) % 12 == valid_pairs['Next_Month']) & 
        ((valid_pairs['Month'] == 12) & (valid_pairs['Next_Year'] == valid_pairs['Year'] + 1) |
         (valid_pairs['Month'] != 12) & (valid_pairs['Next_Year'] == valid_pairs['Year']))
    )
    valid_pairs = valid_pairs[valid_pairs['Is_Consecutive']]

    # Prepare data for Random Forest
    X = valid_pairs[['Month', influential_state]]
    y = valid_pairs['Next_Month_Value']

    # Fit the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Month selection and input for influential state
    month_mapping = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    input_month = st.selectbox(f"Select the current month of {influential_state}:", list(month_mapping.keys()))
    month_num = month_mapping[input_month]

    random_value = st.number_input(f"Enter rainfall value for {influential_state} in {input_month}:", min_value=0.0)

    # Predict the next month's value for the target state
    if st.button("Predict Next Month's Rainfall"):
        predicted_value = max(model.predict([[month_num, random_value]])[0], 0)
        next_month_num = (month_num % 12) + 1
        reverse_month_mapping = {v: k for k, v in month_mapping.items()}
        next_month = reverse_month_mapping[next_month_num]

        # Display the prediction
        st.write(f"The predicted rainfall for {target_state} in {next_month} is: {predicted_value:.2f} mm")
else:
    st.write("No influential state found for the selected target state.")

