import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Beijing Air Quality Analysis",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('preprocessed_air_quality.csv', index_col='datetime', parse_dates=True)
    return df

@st.cache_resource
def load_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    data = load_data().copy()
    data['station_type_encoded'] = (data['station_type'] == 'Urban').astype(int)
    
    features = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 
                'WSPM', 'hour', 'month', 'station_type_encoded']
    
    
    model_df = data[features + ['PM2.5']].dropna()
    
    X = model_df[features]
    y = model_df['PM2.5']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20,
        min_samples_split=2, random_state=42
    )
    model.fit(X_scaled, y)
    return model, scaler
    
df = load_data()
model, scaler = load_model()

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
met_vars = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a section:", 
                         ["Home", "Dataset Explorer", "Visualisation", "Model Outputs"])

st.sidebar.markdown("---")
st.sidebar.markdown("**CMP7005 PRAC1**")
st.sidebar.markdown("Beijing Air Quality Analysis")
st.sidebar.markdown("Student ID: ST20349610")

# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.title("Beijing Air Quality Analysis")
    st.markdown("### From Data to Application Development")
    
    st.markdown("""
    This application provides an interactive platform for exploring air quality 
    data from four monitoring stations in Beijing (March 2013 – February 2017). 
    It enables users to examine the dataset, visualise pollution patterns, and 
    generate PM2.5 predictions using a trained machine learning model.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Stations", df['station'].nunique())
    col3.metric("Date Range", "2013-2017")
    col4.metric("Mean PM2.5", f"{df['PM2.5'].mean():.1f} µg/m³")
    
    st.markdown("---")
    st.markdown("#### How to Use")
    st.markdown("""
    Use the sidebar to navigate between sections:
    
    - **Dataset Explorer** — Browse, filter, and examine the raw and processed data
    - **Visualisation** — Interactive charts exploring pollutant distributions, 
      station comparisons, correlations, and temporal patterns
    - **Model Outputs** — View model performance metrics, feature importance, 
      and generate PM2.5 predictions with custom inputs
    """)

# ============================================================
# DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    
    # Filters
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stations = st.multiselect(
            "Select Stations:", 
            options=df['station'].unique().tolist(),
            default=df['station'].unique().tolist()
        )
    
    with col2:
        selected_type = st.multiselect(
            "Station Type:",
            options=df['station_type'].unique().tolist(),
            default=df['station_type'].unique().tolist()
        )
    
    with col3:
        date_range = st.date_input(
            "Date Range:",
            value=(df.index.min().date(), df.index.max().date()),
            min_value=df.index.min().date(),
            max_value=df.index.max().date()
        )
    
    # Apply filters
    filtered_df = df[
        (df['station'].isin(selected_stations)) &
        (df['station_type'].isin(selected_type))
    ]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df.index.date >= date_range[0]) &
            (filtered_df.index.date <= date_range[1])
        ]
    
    st.markdown(f"### Filtered Dataset ({len(filtered_df):,} records)")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    st.dataframe(filtered_df[pollutants + met_vars].describe().round(2), 
                 use_container_width=True)
    
    # Missing values (from original data before imputation)
    st.markdown("### Station Summary")
    station_summary = filtered_df.groupby('station').agg({
        'PM2.5': ['mean', 'std', 'min', 'max'],
        'AQI': ['mean', 'std']
    }).round(2)
    station_summary.columns = ['PM2.5 Mean', 'PM2.5 Std', 'PM2.5 Min', 'PM2.5 Max', 
                                'AQI Mean', 'AQI Std']
    st.dataframe(station_summary, use_container_width=True)
    
    # AQI distribution
    st.markdown("### AQI Category Distribution")
    aqi_counts = filtered_df['AQI_level'].value_counts()
    aqi_order = ['Excellent', 'Good', 'Lightly Polluted', 'Moderately Polluted',
                 'Heavily Polluted', 'Severely Polluted']
    aqi_counts = aqi_counts.reindex(aqi_order).dropna()
    
    fig = px.pie(values=aqi_counts.values, names=aqi_counts.index,
             title='AQI Category Distribution',
             color_discrete_sequence=['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023'])
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# VISUALISATION
# ============================================================
elif page == "Visualisation":
    st.title("Visualisation")
    
    viz_type = st.selectbox("Select Visualisation Type:", [
        "Pollutant Distributions",
        "Urban vs Suburban Comparison",
        "Scatter Plot Explorer",
        "Correlation Heatmap",
        "Temporal Trends",
        "Seasonal Comparison",
        "AQI Distribution by Station"
    ])
    
    if viz_type == "Pollutant Distributions":
        st.markdown("### Pollutant Distributions")
        selected_pol = st.selectbox("Select Pollutant:", pollutants)
        
        fig = px.histogram(df, x=selected_pol, color='station_type',
                          nbins=50, barmode='overlay', opacity=0.7,
                          title=f'Distribution of {selected_pol}',
                          color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{df[selected_pol].mean():.2f}")
        col2.metric("Median", f"{df[selected_pol].median():.2f}")
        col3.metric("Std Dev", f"{df[selected_pol].std():.2f}")
        col4.metric("Max", f"{df[selected_pol].max():.2f}")
    
    elif viz_type == "Urban vs Suburban Comparison":
        st.markdown("### Urban vs Suburban Comparison")
        selected_pol = st.selectbox("Select Pollutant:", pollutants)
        
        fig = px.box(df, x='station', y=selected_pol, color='station_type',
                     title=f'{selected_pol} by Station',
                     color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        st.plotly_chart(fig, use_container_width=True)
        
        comparison = df.groupby('station_type')[selected_pol].agg(['mean', 'median', 'std']).round(2)
        st.dataframe(comparison, use_container_width=True)
    
    elif viz_type == "Scatter Plot Explorer":
        st.markdown("### Scatter Plot Explorer")
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X-axis:", pollutants + met_vars, index=6)
        with col2:
            y_var = st.selectbox("Y-axis:", pollutants + met_vars, index=0)
        
        sample = df.sample(min(10000, len(df)), random_state=42)
        fig = px.scatter(sample, x=x_var, y=y_var, color='station_type',
                        opacity=0.4, trendline='ols',
                        title=f'{y_var} vs {x_var}',
                        color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        st.markdown("### Correlation Heatmap")
        
        station_filter = st.selectbox("Filter by:", ["All Stations", "Urban Only", "Suburban Only"])
        
        if station_filter == "Urban Only":
            corr_data = df[df['station_type'] == 'Urban']
        elif station_filter == "Suburban Only":
            corr_data = df[df['station_type'] == 'Suburban']
        else:
            corr_data = df
        
        corr = corr_data[pollutants + met_vars].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        ax.set_title(f'Correlation Heatmap ({station_filter})')
        st.pyplot(fig)
    
    elif viz_type == "Temporal Trends":
        st.markdown("### Temporal Trends")
        selected_pol = st.selectbox("Select Pollutant:", pollutants)
        
        monthly = df.groupby([df.index.to_period('M'), 'station_type'])[selected_pol].mean().reset_index()
        monthly.columns = ['Month', 'station_type', selected_pol]
        monthly['Month'] = monthly['Month'].astype(str)
        
        fig = px.line(monthly, x='Month', y=selected_pol, color='station_type',
                      title=f'Monthly Average {selected_pol} Trend',
                      color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        fig.update_xaxes(tickangle=45, dtick=3)
        st.plotly_chart(fig, use_container_width=True)
        
        # Diurnal pattern
        hourly = df.groupby(['hour', 'station_type'])[selected_pol].mean().reset_index()
        fig2 = px.line(hourly, x='hour', y=selected_pol, color='station_type',
                       title=f'Diurnal {selected_pol} Pattern',
                       color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        fig2.update_xaxes(dtick=1)
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Seasonal Comparison":
        st.markdown("### Seasonal Comparison")
        selected_pol = st.selectbox("Select Pollutant:", pollutants)
        
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        seasonal = df.groupby(['season', 'station_type'])[selected_pol].mean().reset_index()
        seasonal['season'] = pd.Categorical(seasonal['season'], categories=season_order, ordered=True)
        seasonal = seasonal.sort_values('season')
        
        fig = px.bar(seasonal, x='season', y=selected_pol, color='station_type',
                     barmode='group', title=f'Seasonal {selected_pol} by Station Type',
                     color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "AQI Distribution by Station":
        st.markdown("### AQI Distribution by Station")
        
        aqi_order = ['Excellent', 'Good', 'Lightly Polluted', 'Moderately Polluted',
                     'Heavily Polluted', 'Severely Polluted']
        aqi_ct = pd.crosstab(df['station'], df['AQI_level'], normalize='index') * 100
        aqi_ct = aqi_ct.reindex(columns=aqi_order)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        aqi_ct.plot(kind='bar', stacked=True, colormap='RdYlGn_r', 
                    edgecolor='black', linewidth=0.5, ax=ax)
        ax.set_title('AQI Category Distribution by Station')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Station')
        ax.legend(title='AQI Level', bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# MODEL OUTPUTS
# ============================================================
elif page == "Model Outputs":
    st.title("Model Outputs")
    
    model_tab = st.selectbox("Select View:", [
        "Performance Summary",
        "Feature Importance",
        "PM2.5 Prediction Tool"
    ])
    
    if model_tab == "Performance Summary":
        st.markdown("### Model Performance Comparison")
        
        results = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest (Default)', 'Random Forest (Optimised)'],
            'RMSE (µg/m³)': [44.07, 28.61, 26.29],
            'MAE (µg/m³)': [29.18, 17.48, 15.65],
            'R²': [0.6881, 0.8686, 0.8891]
        })
        st.dataframe(results, use_container_width=True)
        
        st.markdown("### Key Findings")
        st.markdown("""
        - The optimised Random Forest achieves **R² = 0.8891**, explaining 88.9% 
          of PM2.5 variability.
        - RMSE of **26.29 µg/m³** represents a 40.4% improvement over Linear 
          Regression (44.07 µg/m³).
        - Hyperparameter optimisation (200 trees, max depth 20) improved R² 
          from 0.8686 to 0.8891.
        """)
        
        st.markdown("### Best Model Parameters")
        params = {
            'Parameter': ['n_estimators', 'max_depth', 'min_samples_split', 'random_state'],
            'Value': [200, 20, 2, 42]
        }
        st.dataframe(pd.DataFrame(params), use_container_width=True)
    
    elif model_tab == "Feature Importance":
        st.markdown("### Feature Importance — Optimised Random Forest")
        
        importance_data = pd.DataFrame({
            'Feature': ['CO', 'DEWP', 'NO2', 'SO2', 'month', 'TEMP', 'O3', 'PRES', 
                        'hour', 'WSPM', 'station_type', 'RAIN'],
            'Importance': [0.6858, 0.0777, 0.0572, 0.0352, 0.0311, 0.0293, 0.0288, 
                          0.0271, 0.0104, 0.0077, 0.0073, 0.0024]
        })
        
        fig = px.bar(importance_data, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance Ranking',
                     color='Importance', color_continuous_scale='viridis')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **CO dominates** with 0.6858 importance, confirming that shared combustion 
        sources are the primary driver of PM2.5 variability. DEWP (0.0777) and 
        NO2 (0.0572) rank second and third.
        """)
    
    elif model_tab == "PM2.5 Prediction Tool":
        st.markdown("### PM2.5 Prediction Tool")
        st.markdown("Adjust the input values below to generate a PM2.5 prediction:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            so2 = st.slider("SO2 (µg/m³)", 0.0, 200.0, 14.0)
            no2 = st.slider("NO2 (µg/m³)", 0.0, 200.0, 42.0)
            co = st.slider("CO (µg/m³)", 100.0, 10000.0, 1150.0)
            o3 = st.slider("O3 (µg/m³)", 0.0, 500.0, 62.0)
        
        with col2:
            temp = st.slider("Temperature (°C)", -20.0, 42.0, 13.0)
            pres = st.slider("Pressure (hPa)", 982.0, 1042.0, 1010.0)
            dewp = st.slider("Dew Point (°C)", -44.0, 30.0, 2.0)
            rain = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
        
        with col3:
            wspm = st.slider("Wind Speed (m/s)", 0.0, 13.0, 1.8)
            hour = st.slider("Hour of Day", 0, 23, 12)
            month = st.slider("Month", 1, 12, 6)
            station_type = st.selectbox("Station Type", ["Urban", "Suburban"])
        
        station_encoded = 1 if station_type == "Urban" else 0
        
        # Prepare input
        features = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 
                    'WSPM', 'hour', 'month', 'station_type_encoded']
        input_data = np.array([[so2, no2, co, o3, temp, pres, dewp, rain, 
                                wspm, hour, month, station_encoded]])
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Determine AQI level
        if prediction <= 35:
            aqi_level = "Excellent"
            color = "green"
        elif prediction <= 75:
            aqi_level = "Good"
            color = "green"
        elif prediction <= 115:
            aqi_level = "Lightly Polluted"
            color = "orange"
        elif prediction <= 150:
            aqi_level = "Moderately Polluted"
            color = "orange"
        elif prediction <= 250:
            aqi_level = "Heavily Polluted"
            color = "red"
        else:
            aqi_level = "Severely Polluted"
            color = "red"
        
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        col1, col2 = st.columns(2)
        col1.metric("Predicted PM2.5", f"{prediction:.1f} µg/m³")
        col2.metric("AQI Category", aqi_level)
        
        if prediction > 75:
            st.warning(f"PM2.5 level is {aqi_level}. Health precautions may be advisable.")
        else:
            st.success(f"PM2.5 level is {aqi_level}. Air quality is acceptable.")
