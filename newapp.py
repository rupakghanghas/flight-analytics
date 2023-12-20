import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
price=pd.read_csv('Cleaned_dataset.csv')
delay1=pd.read_csv('Final.csv')
delay2=pd.read_csv('Delay.csv')
st.sidebar.title('Flights Analytics')

user_option = st.sidebar.selectbox('Menu',['Select One','Check Flights','Flight Delay analysis','Price analysis','Airport analysis','Prediction','Real time info'])

if user_option == 'Check Flights':
    st.title('Check Flights')

    col1,col2 = st.columns(2)
    # adding source and destination 
    source=[]
    dest=[]

    for i in price['Destination'].drop_duplicates():
        dest.append(i)
    # print(dest)
    for i in price['Source'].drop_duplicates():
        source.append(i)
    # print(source)

    with col1:
        source_city = st.selectbox('Source',source)
    with col2:
        destination_city = st.selectbox('Destination', dest)
    # displaying information
    results=price[(price['Source']==source_city) & (price['Destination']==destination_city)].groupby('Flight_code').agg(
    Airline=pd.NamedAgg(column='Airline', aggfunc='first'),
    Average_Fare=pd.NamedAgg(column='Fare', aggfunc=lambda x: round(x.mean(), 2)),
    mean_duration=pd.NamedAgg(column='Duration_in_hours', aggfunc=lambda x: round(x.mean(), 2)),
    Arrival=pd.NamedAgg(column='Arrival', aggfunc='first'),
    total_stops=pd.NamedAgg(column='Total_stops', aggfunc='first')).reset_index()
    if st.button('Search'):
        st.dataframe(results)

elif user_option == 'Flight Delay analysis':
    def convert_to_datetime(row):
        parts = row.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return minutes+hours+(seconds/60)
    
    def is_delay(x):
        if x.startswith('-'):
            return x[1:]
        else:
            return x
    delay1['Arrival Time Delay']=delay1['Arrival Time Delay'].apply(is_delay)
    delay1['Arrival Time Delay']=delay1['Arrival Time Delay'].apply(convert_to_datetime)
    df1=pd.DataFrame(delay1.groupby('Carrier')['Arrival Time Delay'].mean().round(2))
    df1=df1.reset_index()
    fig1=px.bar(x='Carrier',y='Arrival Time Delay',data_frame=df1)
    # bar chart for arrival time delay 
    st.title("Arrivel delay of different airlines")
    st.plotly_chart(fig1,theme="streamlit", use_container_width=True)
     # bar chart for arrival time delay 
    st.title("Departure delay of different airlines")
    delay1['Departure Delay']=delay1['Departure Delay'].apply(is_delay)
    delay1['Departure Delay']=delay1['Departure Delay'].apply(convert_to_datetime)
    df2=pd.DataFrame(delay1.groupby('Carrier')['Departure Delay'].mean().round(2))
    df2=df2.reset_index()
    fig2=px.bar(x='Carrier',y='Departure Delay',data_frame=df2)
    st.plotly_chart(fig2,theme="streamlit", use_container_width=True)
    # route delay
    st.title("average delay exprienced by different routes")
    df3=pd.DataFrame(delay1.groupby('route')['Arrival Time Delay'].mean())
    df3=df3.reset_index()
    fig3=px.bar(x='route',y='Arrival Time Delay',data_frame=df3)
    st.plotly_chart(fig3,theme="streamlit", use_container_width=True)
    # scatterplot
    from plotly.subplots import make_subplots
    trace1 = go.Scatter(x=delay1['Arrival Time Delay'].head(1000), y=delay1['A_precipMM'].head(1000), mode='markers', name='air precipitation VS arrival delay',opacity=0.5)
    trace2 = go.Scatter(x=delay1['Arrival Time Delay'].head(1000), y=delay1['A_windspeedKmph'].head(1000), mode='markers', name='wind speed VS arrival delay',opacity=0.5)
    trace3 = go.Scatter(x=delay1['Arrival Time Delay'].head(1000), y=delay1['A_visibility'].head(1000), mode='markers', name='air visibility VS arrival delay',opacity=0.5)
    trace4 = go.Scatter(x=delay1['Arrival Time Delay'].head(500), y=delay1['D_cloudcover'].head(500), mode='markers', name='cloud cover VS arrival delay',opacity=0.5)
    fig= make_subplots(rows=2, cols=2)
    fig.update_xaxes(title_text='flight delay(in minutes)', row=1, col=1)
    fig.update_yaxes(title_text='air precipitation in mm', row=1, col=1)

    fig.update_xaxes(title_text='flight delay(in minutes)', row=1, col=2)
    fig.update_yaxes(title_text='wind speed in km/h', row=1, col=2)
    fig.update_xaxes(title_text='flight delay(in minutes)', row=2, col=1)
    fig.update_yaxes(title_text='air visibilty', row=2, col=1)
    # Add the trace to the first subplot
    fig.add_trace(trace1, row=1, col=1)  
    fig.add_trace(trace2, row=1, col=2)
    fig.add_trace(trace3, row=2, col=1)  
    fig.add_trace(trace4, row=2, col=2)
    st.plotly_chart(fig)
elif user_option == 'Price analysis':
    g1=price.groupby('Airline').agg(fare=pd.NamedAgg(column='Fare',aggfunc=lambda x: round(x.mean(), 1))).reset_index()
    g2=price.groupby('Airline').agg(fare=pd.NamedAgg(column='Fare',aggfunc='max')).reset_index()
    g3=price.groupby('Airline').agg(fare=pd.NamedAgg(column='Fare',aggfunc='min')).reset_index()
    b1=px.bar(data_frame=g1,x='Airline',y='fare')
    b2=px.bar(data_frame=g2,x='Airline',y='fare')
    b3=px.bar(data_frame=g3,x='Airline',y='fare')
    st.header("Average fare price of different airlines")
    st.plotly_chart(b1,theme='streamlit')
    st.header("highest fare price of different airlines")
    st.plotly_chart(b2,theme='streamlit')
    st.header("cheapest fare price of different airlines")
    st.plotly_chart(b3,theme='streamlit')
elif user_option=='Airport analysis':
    # airport rating
    st.header("Airport rating")
    delay1['arrival_airport'] = delay1['route'].str.split('-').str[1]
    delay1['departure_airport'] = delay1['route'].str.split('-').str[0]
    air_df=pd.DataFrame(delay1.groupby('departure_airport')['Departure Airport Rating (out of 10)'].mean()).reset_index()
    air_bar1=px.bar(x='departure_airport',y='Departure Airport Rating (out of 10)',data_frame=air_df)
    st.plotly_chart(air_bar1)
    # airport on time rating
    st.header("Airport on time rating")
    delay1['arrival_airport'] = delay1['route'].str.split('-').str[1]
    delay1['departure_airport'] = delay1['route'].str.split('-').str[0]
    air_df3=pd.DataFrame(delay1.groupby('departure_airport')['Departure Airport On Time Rating (out of 10)'].mean()).reset_index()
    air_bar3=px.bar(x='departure_airport',y='Departure Airport On Time Rating (out of 10)',data_frame=air_df3)
    st.plotly_chart(air_bar3)
    # service rating
    st.header("Airport service rating")
    delay1['arrival_airport'] = delay1['route'].str.split('-').str[1]
    delay1['departure_airport'] = delay1['route'].str.split('-').str[0]
    air_df4=pd.DataFrame(delay1.groupby('departure_airport')['Departure Airport Service Rating (out of 10)'].mean()).reset_index()
    air_bar4=px.bar(x='departure_airport',y='Departure Airport Service Rating (out of 10)',data_frame=air_df4)
    st.plotly_chart(air_bar4)
    # carrier rating
    air_df2=pd.DataFrame(delay1.groupby('Carrier')['Carrier Rating (out of 10)'].mean()).reset_index()
    air_bar2=px.bar(x='Carrier',y='Carrier Rating (out of 10)',data_frame=air_df2)
    st.header("Carrier Rating")
    st.plotly_chart(air_bar2,theme='streamlit')
elif user_option == 'Prediction':
    st.title('Make Prediction')

    col1,col2 = st.columns(2)
    # prediction and input
    pred=['A_DewPointC','A_WindGustKmph','A_cloudcover','A_humidity','A_precipMM','A_pressure','A_tempC','A_visibility','A_winddirDegree','A_windspeedKmph','D_DewPointC','D_WindGustKmph','D_cloudcover','D_humidity','D_precipMM','D_pressure','D_tempC','D_visibility','D_winddirDegree','D_windspeedKmph']
    with col1:
        selected_pred = st.selectbox('Select a parameter for prediction',pred)
    with col2:
        given_input = st.number_input('Enter parameter value for time delay prediction')
    def convert_to_datetime(row):
        parts = row.split(':')
        
        
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return minutes+hours+(seconds/60)
        
    def is_delay(x):
        if x.startswith('-'):
            return x[1:]
        else:
            return x
    delay1['Arrival Time Delay']=delay1['Arrival Time Delay'].apply(is_delay)
    delay1['Arrival Time Delay']=delay1['Arrival Time Delay'].apply(convert_to_datetime)
    # dew_df=delay1[['Arrival Time Delay','A_DewPointC']]


    # l2=['A_DewPointC','A_WindGustKmph','A_cloudcover','A_humidity','A_precipMM','A_pressure','A_tempC','A_visibility','A_winddirDegree','A_windspeedKmph']
    # l1=['D_DewPointC','D_WindGustKmph','D_cloudcover','D_humidity','D_precipMM','D_pressure','D_tempC','D_visibility','D_winddirDegree','D_windspeedKmph']
    # Create a linear regression model
    model = LinearRegression()
    x=np.array(delay1[selected_pred])
    x=x.reshape((len(x), 1))
    y=np.array(delay1['Arrival Time Delay']) 
    y=y.reshape((len(y), 1))

    # X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
    # Fit the model to your data
    model.fit(X=x,y=y)
    given_input=np.array(given_input)
    given_input=given_input.reshape(1,1)
    # # Make predictions
    y_pred = model.predict(given_input)
    # print(y_pred)
    # Calculate Mean Absolute Error (MAE)
    if st.button("click for expected time delay"):
        st.text(f"The predicted time delay is {y_pred[0]} minutes")
    # mae = mean_absolute_error(y_test, y_pred)
elif user_option == 'Real time info':
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
else:
    st.title('FLIGHT INFO HUB')
