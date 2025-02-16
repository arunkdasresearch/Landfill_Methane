import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the merged data
file = 'data/merged_output.xlsx'
df = pd.read_excel(file)

# Strip column names of any trailing spaces
df.columns = df.columns.str.strip()

# Title of the app
st.title("GHGRP Landfill Emissions Dashboard")

# Dropdown to select the type of plot
plot_type = st.selectbox("Select plot type:", ["Bar Plot", "Scatter Plot with Regression"])

# Slider to control the number of data points displayed
num_points = st.slider("Select number of data points to display:", min_value=10, max_value=len(df), value=len(df), step=10)

# Subset the data based on the slider selection
df_subset = df.head(num_points)

# If scatter plot is selected, allow showing the regression line
show_regression = False
if plot_type == "Scatter Plot with Regression":
    show_regression = st.checkbox("Show Regression Line")

# Generate the selected plot
if plot_type == "Bar Plot":
    fig = px.bar(
        df_subset, 
        x='Waste in Place (tons)', 
        y='Average reported GHGRP emissions (METRIC TONS CO2e)',
        text='Average reported GHGRP emissions (METRIC TONS CO2e)',  # Display values on bars
        labels={"Waste in Place (tons)": "Waste in Place (tons)", "Average reported GHGRP emissions (METRIC TONS CO2e)": "Emissions (CO2e)"},
        title="Reported GHGRP Emissions vs Waste in Place",
    )

    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')  # Keep labels readable
    fig.update_layout(
        xaxis_tickangle=-45,  
        yaxis_title="Reported Emissions (Metric Tons CO2e)",
        xaxis_title="Waste in Place (tons)",
        margin=dict(l=50, r=50, t=50, b=50)
    )

elif plot_type == "Scatter Plot with Regression":
    fig = px.scatter(
        df_subset, 
        x='Waste in Place (tons)', 
        y='Average reported GHGRP emissions (METRIC TONS CO2e)',
        hover_name="Landfill Name",  # Show landfill name on hover
        labels={"Waste in Place (tons)": "Waste in Place (tons)", "Average reported GHGRP emissions (METRIC TONS CO2e)": "Emissions (CO2e)"},
        title="Reported GHGRP Emissions vs Waste in Place (Scatter)",
    )

    if show_regression:
        # Fit a linear regression model
        X = df_subset[['Waste in Place (tons)']].values
        y = df_subset['Average reported GHGRP emissions (METRIC TONS CO2e)'].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        # Add regression line
        fig.add_trace(
            go.Scatter(
                x=df_subset['Waste in Place (tons)'], 
                y=y_pred, 
                mode='lines', 
                name=f"Regression Line (Slope: {model.coef_[0]:.2f})",
                line=dict(color="red")
            )
        )

# Show plot in Streamlit
st.plotly_chart(fig)
