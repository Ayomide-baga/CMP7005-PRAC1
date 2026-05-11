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
with st.spinner("🤖 Training model on Beijing air quality data... this may take a minute on first load."):
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
    # ── Hero Banner ──────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%); 
                padding: 40px; border-radius: 15px; margin-bottom: 25px;
                border-left: 5px solid #63b3ed;">
        <h1 style="color: #63b3ed; margin:0; font-size: 2.5em;">🌍 Beijing Air Quality Analysis</h1>
        <p style="color: #a0aec0; font-size: 1.1em; margin-top: 10px;">
           Exploratory Analysis, Predictive Modelling & Interactive Visualisation · CMP7005 PRAC1
        </p>
        <p style="color: #cbd5e0; margin-top: 10px;">
            Interactive exploration of hourly air quality data from four monitoring 
            stations across Beijing's urban-suburban gradient (March 2013 – February 2017).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Better Metric Cards ──────────────────────────────────
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    mean_pm25 = df['PM2.5'].mean()
    if mean_pm25 <= 35:
        pm_color = "#00e400"
        pm_status = "Good"
    elif mean_pm25 <= 75:
        pm_color = "#ffff00"
        pm_status = "Moderate"
    elif mean_pm25 <= 115:
        pm_color = "#ff7e00"
        pm_status = "Unhealthy"
    else:
        pm_color = "#ff0000"
        pm_status = "Very Unhealthy"

    col1.metric("📁 Total Records", f"{len(df):,}")
    col2.metric("📍 Stations", df['station'].nunique())
    col3.metric("📅 Date Range", "2013–2017")
    col4.metric("💨 Mean PM2.5", f"{mean_pm25:.1f} µg/m³", delta=pm_status)

    # ── Quick Stats Row ──────────────────────────────────────
    st.markdown("### ⚡ Quick Insights")
    col1, col2, col3, col4 = st.columns(4)

    worst_station = df.groupby('station')['PM2.5'].mean().idxmax()
    best_station = df.groupby('station')['PM2.5'].mean().idxmin()
    worst_month = df.groupby(df.index.month)['PM2.5'].mean().idxmax()
    best_month = df.groupby(df.index.month)['PM2.5'].mean().idxmin()
    month_names = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
                   7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

    col1.metric("🏭 Most Polluted Station", worst_station)
    col2.metric("🌿 Cleanest Station", best_station)
    col3.metric("❄️ Worst Month", month_names[worst_month])
    col4.metric("☀️ Best Month", month_names[best_month])

    # ── Pollution Level Legend ───────────────────────────────
    st.markdown("### 🎨 PM2.5 Air Quality Index")
    cols = st.columns(6)
    levels = [
        ("Excellent", "≤35", "#00e400"),
        ("Good", "≤75", "#ffff00"),
        ("Lightly Polluted", "≤115", "#ff7e00"),
        ("Moderately Polluted", "≤150", "#ff0000"),
        ("Heavily Polluted", "≤250", "#8f3f97"),
        ("Severely Polluted", ">250", "#7e0023"),
    ]
    for col, (label, threshold, color) in zip(cols, levels):
        col.markdown(f"""
        <div style="background-color: {color}22; border-left: 4px solid {color}; 
                    padding: 10px; border-radius: 5px; text-align:center;">
            <div style="color: {color}; font-weight: bold; font-size:0.85em;">{label}</div>
            <div style="color: #a0aec0; font-size:0.8em;">{threshold} µg/m³</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── About the Data ───────────────────────────────────────
    with st.expander("📂 About the Data"):
        st.markdown("""
        **Source:** Beijing Multi-Site Air Quality Dataset (UCI Machine Learning Repository)
        
        | Station | Type | Location |
        |---------|------|----------|
        | Dongsi | 🏙️ Urban | Inner Beijing |
        | Tiantan | 🏙️ Urban | Inner Beijing |
        | Dingling | 🌿 Suburban | Northern outskirts |
        | Huairou | 🌿 Suburban | Northern outskirts |
        
        **Pollutants measured:** PM2.5, PM10, SO2, NO2, CO, O3  
        **Meteorological variables:** Temperature, Pressure, Dew Point, Rainfall, Wind Speed  
        **Period:** March 2013 – February 2017 (hourly records)
        """)

    # ── How to Use ───────────────────────────────────────────
    st.markdown("### 🧭 How to Use")
    col1, col2, col3 = st.columns(3)
    col1.info("**📊 Dataset Explorer**\nBrowse, filter, and examine the raw and processed data by station, type, and date range.")
    col2.info("**📈 Visualisation**\nInteractive charts exploring pollutant distributions, correlations, and temporal patterns.")
    col3.info("**🤖 Model Outputs**\nView model performance, feature importance, and generate live PM2.5 predictions.")

# ============================================================
# DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.info("💡 Use the filters below to explore the dataset by station, type, and date range. The table shows the first 100 rows of your filtered selection.")

    # Filters
    st.markdown("### 🔍 Filters")
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
        min_date = pd.Timestamp('2013-03-01').date()
        max_date = pd.Timestamp('2017-02-28').date()
        date_range = st.date_input(
            "Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="YYYY/MM/DD"
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
    
    st.markdown(f"### 📋 Filtered Dataset ({len(filtered_df):,} records)")
    st.info("💡 Showing the first 100 rows of the filtered dataset. Use the download button below to export the full filtered dataset.")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    csv = filtered_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="beijing_air_quality_filtered.csv",
        mime="text/csv"
    )
    
    st.markdown("### 📈 Summary Statistics")
    st.info("💡 Descriptive statistics for pollutants and meteorological variables in the filtered dataset. Includes mean, standard deviation, min, max and quartile values.")
    st.dataframe(filtered_df[pollutants + met_vars].describe().round(2), 
                 use_container_width=True)
    
    st.markdown("### 🏭 Station Summary")
    st.info("💡 Average PM2.5 and AQI statistics grouped by station, allowing comparison of pollution levels across the four monitoring sites.")
    station_summary = filtered_df.groupby('station').agg({
        'PM2.5': ['mean', 'std', 'min', 'max'],
        'AQI': ['mean', 'std']
    }).round(2)
    station_summary.columns = ['PM2.5 Mean', 'PM2.5 Std', 'PM2.5 Min', 'PM2.5 Max', 
                                'AQI Mean', 'AQI Std']
    st.dataframe(station_summary, use_container_width=True)
    
    st.markdown("### 🥧 AQI Category Distribution")
    st.info("💡 Proportion of hourly readings falling into each AQI category for the filtered selection. A larger green slice indicates better overall air quality.")
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
    st.title("📈 Visualisation")
    st.info("💡 Select a visualisation type from the dropdown below to explore different aspects of the Beijing air quality dataset.")
    
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
        st.markdown("### 🌫️ Pollutant Distributions")
        st.info("💡 Histogram showing the frequency distribution of the selected pollutant across Urban and Suburban stations. Overlapping bars reveal differences in pollution exposure between station types.")
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
        st.markdown("### 🏙️ Urban vs Suburban Comparison")
        st.info("💡 Box plots showing the spread and median of the selected pollutant across all four stations, grouped by station type. The box represents the interquartile range, and whiskers show the data range excluding outliers.")
        selected_pol = st.selectbox("Select Pollutant:", pollutants)
        
        fig = px.box(df, x='station', y=selected_pol, color='station_type',
                     title=f'{selected_pol} by Station',
                     color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        st.plotly_chart(fig, use_container_width=True)
        
        comparison = df.groupby('station_type')[selected_pol].agg(['mean', 'median', 'std']).round(2)
        st.dataframe(comparison, use_container_width=True)
    
    elif viz_type == "Scatter Plot Explorer":
        st.markdown("### 🔵 Scatter Plot Explorer")
        st.info("💡 Explore relationships between any two variables. Each point represents one hourly reading. The OLS trendline shows the overall linear relationship — a steeper line indicates a stronger correlation.")
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
        st.markdown("### 🔥 Correlation Heatmap")
        st.info("💡 Pearson correlation coefficients between all pollutants and meteorological variables. Values close to +1 (dark red) indicate strong positive correlation, values close to -1 (dark blue) indicate strong negative correlation, and values near 0 indicate no linear relationship.")
        
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
        st.markdown("### 📅 Temporal Trends")
        st.info("💡 Monthly average concentrations over the full 4-year period (top) and average hourly patterns within a day (bottom). Use these to identify seasonal peaks and daily pollution cycles.")
        selected_pol = st.selectbox("Select Pollutant:", pollutants)
        
        monthly = df.groupby([df.index.to_period('M'), 'station_type'])[selected_pol].mean().reset_index()
        monthly.columns = ['Month', 'station_type', selected_pol]
        monthly['Month'] = monthly['Month'].astype(str)
        
        fig = px.line(monthly, x='Month', y=selected_pol, color='station_type',
                      title=f'Monthly Average {selected_pol} Trend',
                      color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        fig.update_xaxes(tickangle=45, dtick=3)
        st.plotly_chart(fig, use_container_width=True)
        
        hourly = df.groupby(['hour', 'station_type'])[selected_pol].mean().reset_index()
        fig2 = px.line(hourly, x='hour', y=selected_pol, color='station_type',
                       title=f'Diurnal {selected_pol} Pattern',
                       color_discrete_map={'Urban': '#e74c3c', 'Suburban': '#2ecc71'})
        fig2.update_xaxes(dtick=1)
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Seasonal Comparison":
        st.markdown("### 🌸 Seasonal Comparison")
        st.info("💡 Average pollutant concentrations grouped by season and station type. Winter typically shows higher pollution due to heating emissions and stable atmospheric conditions trapping pollutants near the surface.")
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
        st.markdown("### 🏭 AQI Distribution by Station")
        st.info("💡 Stacked bar chart showing the percentage of hourly readings in each AQI category per station. Stations with more green indicate better overall air quality, while more red/purple indicates worse pollution levels.")
        
        aqi_order = ['Excellent', 'Good', 'Lightly Polluted', 'Moderately Polluted',
                     'Heavily Polluted', 'Severely Polluted']
        
        aqi_counts = df.groupby(['station', 'AQI_level']).size().reset_index(name='count')
        aqi_totals = aqi_counts.groupby('station')['count'].transform('sum')
        aqi_counts['percentage'] = aqi_counts['count'] / aqi_totals * 100
        aqi_counts['AQI_level'] = pd.Categorical(aqi_counts['AQI_level'], 
                                                  categories=aqi_order, ordered=True)
        aqi_counts = aqi_counts.sort_values('AQI_level')
        
        fig = px.bar(aqi_counts, x='station', y='percentage', color='AQI_level',
                     title='AQI Category Distribution by Station (%)',
                     labels={'percentage': 'Percentage (%)', 'station': 'Station'},
                     color_discrete_map={
                         'Excellent': '#00e400',
                         'Good': '#ffff00',
                         'Lightly Polluted': '#ff7e00',
                         'Moderately Polluted': '#ff0000',
                         'Heavily Polluted': '#8f3f97',
                         'Severely Polluted': '#7e0023'
                     },
                     category_orders={'AQI_level': aqi_order})
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODEL OUTPUTS
# ============================================================
elif page == "Model Outputs":
    st.title("🤖 Model Outputs")
    st.info("💡 This section presents the machine learning model built to predict PM2.5 concentrations. Navigate between the tabs below to explore model performance, feature importance, and generate predictions.")
    
    model_tab = st.selectbox("Select View:", [
        "Performance Summary",
        "Feature Importance",
        "PM2.5 Prediction Tool",
        "Actual vs Predicted",
        "Residual Analysis",
        "Batch Prediction"
    ])
    
    if model_tab == "Performance Summary":
        st.markdown("### 📊 Model Performance Comparison")
        st.info("💡 Comparison of three models evaluated on the test set. RMSE and MAE measure average prediction error in µg/m³ — lower is better. R² measures how much variance the model explains — closer to 1.0 is better.")
        results = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest (Default)', 'Random Forest (Optimised)'],
            'RMSE (µg/m³)': [44.07, 28.61, 26.29],
            'MAE (µg/m³)': [29.18, 17.48, 15.65],
            'R²': [0.6881, 0.8686, 0.8891]
        })
        st.dataframe(results, use_container_width=True)
        
        st.markdown("### 🔍 Key Findings")
        st.markdown("""
        - The optimised Random Forest achieves **R² = 0.8891**, explaining 88.9% of PM2.5 variability.
        - RMSE of **26.29 µg/m³** represents a 40.4% improvement over Linear Regression (44.07 µg/m³).
        - Hyperparameter optimisation (200 trees, max depth 20) improved R² from 0.8686 to 0.8891.
        """)
        
        st.markdown("### ⚙️ Best Model Parameters")
        params = {
            'Parameter': ['n_estimators', 'max_depth', 'min_samples_split', 'random_state'],
            'Value': [200, 20, 2, 42]
        }
        st.dataframe(pd.DataFrame(params), use_container_width=True)

    elif model_tab == "Feature Importance":
        st.markdown("### 🎯 Feature Importance — Optimised Random Forest")
        st.info("💡 Feature importance scores show how much each variable contributes to the model's predictions. A higher score means the feature has greater influence on PM2.5 predictions. Scores sum to 1.0.")
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
        st.markdown("### 🎛️ PM2.5 Prediction Tool")
        st.info("💡 Adjust the input sliders to simulate different environmental conditions and generate a real-time PM2.5 prediction using the trained Random Forest model. The AQI category is determined based on Chinese air quality standards.")
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
        features = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN',
                    'WSPM', 'hour', 'month', 'station_type_encoded']
        input_data = np.array([[so2, no2, co, o3, temp, pres, dewp, rain,
                                wspm, hour, month, station_encoded]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        if prediction <= 35:
            aqi_level = "Excellent"
        elif prediction <= 75:
            aqi_level = "Good"
        elif prediction <= 115:
            aqi_level = "Lightly Polluted"
        elif prediction <= 150:
            aqi_level = "Moderately Polluted"
        elif prediction <= 250:
            aqi_level = "Heavily Polluted"
        else:
            aqi_level = "Severely Polluted"
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Predicted PM2.5", f"{prediction:.1f} µg/m³")
        col2.metric("AQI Category", aqi_level)
        if prediction > 75:
            st.warning(f"PM2.5 level is {aqi_level}. Health precautions may be advisable.")
        else:
            st.success(f"PM2.5 level is {aqi_level}. Air quality is acceptable.")

    elif model_tab == "Actual vs Predicted":
        st.markdown("### 📉 Actual vs Predicted PM2.5")
        st.info("💡 Scatter plot comparing actual PM2.5 values against model predictions on the test set (20% holdout). Points lying close to the red dashed line indicate accurate predictions. Spread away from the line represents prediction error.")
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        data = df.copy()
        data['station_type_encoded'] = (data['station_type'] == 'Urban').astype(int)
        features = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN',
                    'WSPM', 'hour', 'month', 'station_type_encoded']
        model_df = data[features + ['PM2.5']].dropna()
        X = model_df[features]
        y = model_df['PM2.5']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
        sample = results_df.sample(2000, random_state=42)
        fig = px.scatter(sample, x='Actual', y='Predicted', opacity=0.4,
                         title='Actual vs Predicted PM2.5 (Test Set Sample)',
                         labels={'Actual': 'Actual PM2.5 (µg/m³)',
                                 'Predicted': 'Predicted PM2.5 (µg/m³)'})
        max_val = max(sample['Actual'].max(), sample['Predicted'].max())
        fig.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color='red', dash='dash'))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red dashed line = perfect prediction. Points closer to the line = better accuracy.")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f} µg/m³")
        col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f} µg/m³")
        col3.metric("R²", f"{r2_score(y_test, y_pred):.4f}")

    elif model_tab == "Residual Analysis":
        st.markdown("### 🔬 Residual Analysis")
        st.info("💡 Residuals are the differences between actual and predicted values. A good model should have residuals randomly scattered around zero (top chart) and follow a bell-curve distribution centred near zero (bottom chart).")
        from sklearn.model_selection import train_test_split
        data = df.copy()
        data['station_type_encoded'] = (data['station_type'] == 'Urban').astype(int)
        features = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN',
                    'WSPM', 'hour', 'month', 'station_type_encoded']
        model_df = data[features + ['PM2.5']].dropna()
        X = model_df[features]
        y = model_df['PM2.5']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        residuals = y_test.values - y_pred
        fig1 = px.scatter(x=y_pred, y=residuals, opacity=0.4,
                          title='Residuals vs Predicted Values',
                          labels={'x': 'Predicted PM2.5 (µg/m³)',
                                  'y': 'Residual (Actual − Predicted)'})
        fig1.add_hline(y=0, line_dash='dash', line_color='red')
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Points scattered randomly around 0 = good model. Patterns suggest bias.")
        fig2 = px.histogram(x=residuals, nbins=80,
                            title='Residual Distribution',
                            labels={'x': 'Residual (µg/m³)', 'y': 'Count'})
        fig2.add_vline(x=0, line_dash='dash', line_color='red')
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("A bell curve centred near 0 indicates unbiased predictions.")

    elif model_tab == "Batch Prediction":
        st.markdown("### 📂 Batch Prediction from Uploaded CSV")
        st.info("💡 Upload a CSV file containing multiple air quality readings to generate PM2.5 predictions for all rows at once. Download the template below to see the required format.")
        st.warning("""
        ⚠️ **Important Notice:** This model was trained on Beijing air quality data (2013–2017). 
        Predictions are most accurate for similar urban/suburban environments. 
        Results may be unreliable for significantly different geographical locations or climates.
        """)
        st.info("""
        📋 **Expected value ranges for best results:**
        - SO2: 0–200 µg/m³ | NO2: 0–200 µg/m³ | CO: 100–10,000 µg/m³ | O3: 0–500 µg/m³
        - Temperature: -20°C to 42°C | Pressure: 982–1042 hPa | Wind Speed: 0–13 m/s
        - Hour: 0–23 | Month: 1–12 | Station Type: Urban or Suburban
        """)
        template_data = pd.DataFrame({
            'SO2': [14.0], 'NO2': [42.0], 'CO': [1150.0], 'O3': [62.0],
            'TEMP': [13.0], 'PRES': [1010.0], 'DEWP': [2.0], 'RAIN': [0.0],
            'WSPM': [1.8], 'hour': [12], 'month': [6], 'station_type': ['Urban']
        })
        template_csv = template_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Template",
            data=template_csv,
            file_name="prediction_template.csv",
            mime="text/csv"
        )
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload your CSV file:", type="csv")
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.markdown("#### 👀 Uploaded Data Preview")
                st.dataframe(input_df.head(), use_container_width=True)
                required_cols = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES',
                               'DEWP', 'RAIN', 'WSPM', 'hour', 'month', 'station_type']
                missing_cols = [c for c in required_cols if c not in input_df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}. Please use the template.")
                else:
                    input_df['station_type_encoded'] = (input_df['station_type'] == 'Urban').astype(int)
                    features = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP',
                               'RAIN', 'WSPM', 'hour', 'month', 'station_type_encoded']
                    X_new = input_df[features].dropna()

                    # Validate input ranges
                    validation_ranges = {
                        'SO2': (0, 200), 'NO2': (0, 200), 'CO': (100, 10000),
                        'O3': (0, 500), 'TEMP': (-20, 42), 'PRES': (982, 1042),
                        'WSPM': (0, 13), 'hour': (0, 23), 'month': (1, 12)
                    }
                    warnings_list = []
                    for col, (low, high) in validation_ranges.items():
                        if col in X_new.columns:
                            out_of_range = ((X_new[col] < low) | (X_new[col] > high)).sum()
                            if out_of_range > 0:
                                warnings_list.append(f"**{col}**: {out_of_range} row(s) outside expected range ({low}–{high})")
                    if warnings_list:
                        st.warning("⚠️ Some values are outside the training data range — predictions may be less accurate:\n\n" +
                                   "\n".join(f"- {w}" for w in warnings_list))

                    X_scaled = scaler.transform(X_new)
                    predictions = model.predict(X_scaled)
                    input_df.loc[X_new.index, 'Predicted_PM2.5'] = predictions.round(2)

                    def get_aqi_level(pm):
                        if pm <= 35: return 'Excellent'
                        elif pm <= 75: return 'Good'
                        elif pm <= 115: return 'Lightly Polluted'
                        elif pm <= 150: return 'Moderately Polluted'
                        elif pm <= 250: return 'Heavily Polluted'
                        else: return 'Severely Polluted'

                    input_df['AQI_Category'] = input_df['Predicted_PM2.5'].apply(get_aqi_level)

                    st.markdown("#### 📊 Prediction Results")
                    st.dataframe(input_df[['SO2', 'NO2', 'CO', 'O3', 'TEMP',
                                           'station_type', 'Predicted_PM2.5',
                                           'AQI_Category']],
                                use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Predictions", len(predictions))
                    col2.metric("Mean Predicted PM2.5", f"{predictions.mean():.1f} µg/m³")
                    col3.metric("Max Predicted PM2.5", f"{predictions.max():.1f} µg/m³")

                    fig = px.histogram(x=predictions, nbins=30,
                                      title='Distribution of Predicted PM2.5 Values',
                                      labels={'x': 'Predicted PM2.5 (µg/m³)', 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)

                    result_csv = input_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=result_csv,
                        file_name="pm25_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error("Error processing file. Please check your CSV format and try again.")
        else:
            st.info("👆 Upload a CSV file above to get started, or download the template first.")


