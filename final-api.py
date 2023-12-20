import streamlit as st
import requests

# Create a Streamlit app
st.title("Flight Information Search")

# Create a text input for the user to enter a flight number
flight_number = st.text_input("Enter Flight Number:")

# Create a button to trigger the search
search_button = st.button("Search")

if search_button:
    # Get the user's search query
    user_query = flight_number

    # Define parameters for the API request
    params = {
        'access_key': '8767529ae88b310c5e0648b5fc974f26',
        'flight_number': user_query
    }

    # Define the API endpoint URL
    api_url = "http://api.aviationstack.com/v1/flights"

    # Make the API request
    req = requests.get(api_url, params=params)

    if req.status_code == 200:
        api_response = req.json()
        flight_data = api_response.get("data")[0]

        st.write("Flight Information:")
        st.write(f'Time-Zone: {flight_data["departure"]["timezone"]}')
        st.write(f'Airline: {flight_data["airline"]["name"]}')
        st.write(f'Status: {flight_data["flight_status"]}')
        st.write(f'Departure Airport: {flight_data["departure"]["airport"]}')
        st.write(f'Arrival Airport: {flight_data["arrival"]["airport"]}')
    else:
        st.error(f"Failed to fetch data for flight number: {user_query}")
