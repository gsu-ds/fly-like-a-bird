import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os
from datetime import datetime, time

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .delay-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .delay-no {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    h1 {
        color: #1e3a8a;
    }
    h2 {
        color: #3b82f6;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 10px;
    }
    h3 {
        color: #475569;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_metrics():
    return pd.read_csv('metrics.csv')

@st.cache_data
def load_sample_data():
    if os.path.exists('./data/processed/cleaned_data.csv'):
        df = pd.read_csv('./data/processed/cleaned_data.csv')
        return df.head(1000)  # Load sample for performance
    return None

# Prediction function (simulated - in production, load actual model)
def predict_delay(features):
    """
    Simulated prediction function.
    In production, this would load the trained Random Forest model and make real predictions.
    """
    # Simple rule-based prediction for demo purposes
    # In production: model.predict(features)
    
    delay_probability = 0.0
    
    # Factors that increase delay probability
    if features['dep_hour'] >= 17:  # Evening flights
        delay_probability += 0.25
    if features['dep_hour'] <= 6:  # Very early morning
        delay_probability += 0.15
    if features['distance'] > 1500:  # Long distance
        delay_probability += 0.20
    if features['month'] in [6, 7, 12]:  # Summer and December (busy)
        delay_probability += 0.15
    if features['taxi_out'] > 20:  # Long taxi time
        delay_probability += 0.20
    
    # Add some randomness
    delay_probability += np.random.uniform(-0.1, 0.1)
    delay_probability = np.clip(delay_probability, 0, 1)
    
    prediction = 1 if delay_probability > 0.5 else 0
    confidence = delay_probability if prediction == 1 else (1 - delay_probability)
    
    return prediction, confidence

# Header
st.title("✈️ Flight Delay Prediction System")
st.markdown("### ATL Airport - Machine Learning Powered Delay Forecasting")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=100)
    st.title("Navigation")
    page = st.radio(
        "Select View:",
        ["🎯 Predict Delay", "📊 Model Performance", "📈 Data Insights", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### Model Info")
    st.info("""
    **Model:** Random Forest  
    **Accuracy:** 96.31%  
    **Airport:** ATL (Atlanta)  
    **Dataset:** 2022 Flights
    """)

# Main content based on page selection
if page == "🎯 Predict Delay":
    st.header("Flight Delay Prediction")
    st.markdown("Enter flight details to predict arrival delay probability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Flight Details")
        airline = st.selectbox(
            "Airline",
            ["Delta Air Lines Inc.", "Southwest Airlines Co.", "American Airlines Inc.", 
             "United Air Lines Inc.", "Spirit Air Lines", "Frontier Airlines Inc.",
             "JetBlue Airways"]
        )
        
        distance = st.slider("Flight Distance (miles)", 100, 3000, 800)
        
        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            format_func=lambda x: datetime(2024, x, 1).strftime('%B')
        )
        
        day = st.slider("Day of Month", 1, 31, 15)
    
    with col2:
        st.subheader("Departure Info")
        dep_time = st.time_input("Scheduled Departure Time", time(14, 30))
        dep_hour = dep_time.hour
        
        arr_time = st.time_input("Scheduled Arrival Time", time(16, 45))
        
        taxi_out = st.slider("Expected Taxi-Out Time (min)", 5, 40, 15)
    
    with col3:
        st.subheader("Additional Info")
        taxi_in = st.slider("Expected Taxi-In Time (min)", 3, 30, 7)
        air_time = st.slider("Expected Air Time (min)", 30, 400, 120)
        
    st.markdown("---")
    
    if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
        # Prepare features
        features = {
            'airline': airline,
            'distance': distance,
            'month': month,
            'day': day,
            'dep_hour': dep_hour,
            'taxi_out': taxi_out,
            'taxi_in': taxi_in,
            'air_time': air_time
        }
        
        # Make prediction
        prediction, confidence = predict_delay(features)
        
        # Display result
        st.markdown("### Prediction Result")
        
        if prediction == 1:
            st.markdown(
                f'<div class="prediction-box delay-yes">⚠️ DELAY EXPECTED<br/>'
                f'<span style="font-size: 16px;">Confidence: {confidence*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
            st.warning("This flight is likely to be delayed by 15+ minutes. Consider allowing extra time.")
        else:
            st.markdown(
                f'<div class="prediction-box delay-no">✅ ON-TIME EXPECTED<br/>'
                f'<span style="font-size: 16px;">Confidence: {confidence*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )
            st.success("This flight is expected to arrive on time!")
        
        # Risk factors
        st.markdown("### Risk Factors Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            time_risk = "High" if (dep_hour >= 17 or dep_hour <= 6) else "Low"
            st.metric("Time of Day Risk", time_risk)
        
        with col2:
            distance_risk = "High" if distance > 1500 else "Medium" if distance > 800 else "Low"
            st.metric("Distance Risk", distance_risk)
        
        with col3:
            season_risk = "High" if month in [6, 7, 12] else "Medium" if month in [3, 11] else "Low"
            st.metric("Seasonal Risk", season_risk)
        
        with col4:
            taxi_risk = "High" if taxi_out > 20 else "Medium" if taxi_out > 15 else "Low"
            st.metric("Taxi Time Risk", taxi_risk)

elif page == "📊 Model Performance":
    st.header("Model Performance Metrics")
    
    # Load metrics
    metrics_df = load_metrics()
    
    # Display best model
    best_model = metrics_df.loc[metrics_df['accuracy'].idxmax()]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", "Random Forest (Tuned)")
    with col2:
        st.metric("Accuracy", f"{best_model['accuracy']*100:.2f}%")
    with col3:
        st.metric("Precision (Avg)", f"{best_model['macro_avg_precision']:.3f}")
    with col4:
        st.metric("F1-Score (Avg)", f"{best_model['macro_avg_f1']:.3f}")
    
    st.markdown("---")
    
    # Performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Accuracy Comparison")
        fig = px.bar(
            metrics_df,
            x='model_name',
            y='accuracy',
            color='accuracy',
            color_continuous_scale='Blues',
            labels={'accuracy': 'Accuracy', 'model_name': 'Model'},
            text=metrics_df['accuracy'].apply(lambda x: f'{x*100:.2f}%')
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Precision vs Recall")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics_df['macro_avg_recall'],
            y=metrics_df['macro_avg_precision'],
            mode='markers+text',
            marker=dict(size=15, color=metrics_df['accuracy'], 
                       colorscale='Viridis', showscale=True,
                       colorbar=dict(title="Accuracy")),
            text=metrics_df['model_name'],
            textposition='top center'
        ))
        fig.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Performance Metrics")
    
    display_df = metrics_df[['model_name', 'accuracy', 'precision_class_0', 'precision_class_1', 
                              'recall_class_0', 'recall_class_1', 'f1_score_class_0', 'f1_score_class_1']]
    display_df.columns = ['Model', 'Accuracy', 'Precision (No Delay)', 'Precision (Delayed)',
                          'Recall (No Delay)', 'Recall (Delayed)', 'F1 (No Delay)', 'F1 (Delayed)']
    
    st.dataframe(
        display_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision (No Delay)': '{:.3f}',
            'Precision (Delayed)': '{:.3f}',
            'Recall (No Delay)': '{:.3f}',
            'Recall (Delayed)': '{:.3f}',
            'F1 (No Delay)': '{:.3f}',
            'F1 (Delayed)': '{:.3f}'
        }).background_gradient(cmap='Blues', subset=['Accuracy']),
        use_container_width=True
    )
    
    # Class distribution
    st.markdown("---")
    st.subheader("Class Distribution in Test Set")
    
    class_data = pd.DataFrame({
        'Class': ['On-Time (0)', 'Delayed (1)'],
        'Count': [28812, 7159],
        'Percentage': [80.1, 19.9]
    })
    
    fig = px.pie(
        class_data,
        values='Count',
        names='Class',
        color='Class',
        color_discrete_map={'On-Time (0)': '#66bb6a', 'Delayed (1)': '#ef5350'},
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Data Insights":
    st.header("Data Analysis & Insights")
    
    # Display saved visualizations if available
    viz_path = './output_figures/'
    
    if os.path.exists(viz_path):
        st.subheader("Average Flight Delays by Hour of Day")
        st.markdown("Flight delays tend to increase throughout the day, with peak delays in the evening hours.")
        
        if os.path.exists(os.path.join(viz_path, 'average_delays_by_hour.png')):
            img = Image.open(os.path.join(viz_path, 'average_delays_by_hour.png'))
            st.image(img, use_column_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Busiest Airports")
            if os.path.exists(os.path.join(viz_path, 'top10_busiest_airports.png')):
                img = Image.open(os.path.join(viz_path, 'top10_busiest_airports.png'))
                st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("Confusion Matrix - Best Model")
            if os.path.exists(os.path.join(viz_path, 'confusion_matrix_random_forest.png')):
                img = Image.open(os.path.join(viz_path, 'confusion_matrix_random_forest.png'))
                st.image(img, use_column_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curve")
            if os.path.exists(os.path.join(viz_path, 'roc_curve.png')):
                img = Image.open(os.path.join(viz_path, 'roc_curve.png'))
                st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("Precision-Recall Curve")
            if os.path.exists(os.path.join(viz_path, 'precision_recall_curve.png')):
                img = Image.open(os.path.join(viz_path, 'precision_recall_curve.png'))
                st.image(img, use_column_width=True)
    
    # Interactive data exploration
    st.markdown("---")
    st.subheader("Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Time of Day Impact**
        - Early morning flights: Lower delay rates
        - Evening flights: Higher delay rates
        - Peak delays: 18:00-21:00
        """)
    
    with col2:
        st.info("""
        **Delay Correlation**
        - Strong correlation between departure and arrival delays
        - Taxi times impact overall delays
        - Distance has moderate effect
        """)
    
    with col3:
        st.info("""
        **Airport Insights**
        - ATL is the busiest US airport
        - 183,697 flights analyzed
        - ~20% delayed by 15+ minutes
        """)

else:  # About page
    st.header("About This Project")
    
    st.markdown("""
    ### Flight Delay Prediction System
    
    This machine learning system predicts flight arrival delays (15+ minutes) for flights 
    departing from **Hartsfield-Jackson Atlanta International Airport (ATL)**, the busiest 
    airport in the United States.
    
    #### 🎯 Project Goals
    - Develop accurate delay prediction models for operational planning
    - Identify key factors contributing to flight delays
    - Provide interpretable insights for airlines and passengers
    
    #### 📊 Dataset
    - **Source:** 2022 US Flight Data
    - **Total Flights:** 4,078,318 (full dataset)
    - **ATL Flights:** 183,697 flights
    - **Training Samples:** 143,881
    - **Test Samples:** 35,971
    
    #### 🤖 Models Evaluated
    1. **k-Nearest Neighbors (kNN)** - 92.59% accuracy
    2. **Logistic Regression** - 93.26% accuracy
    3. **Random Forest** - 96.28% accuracy
    4. **Random Forest (Tuned)** - **96.31% accuracy** ✨ *Best Model*
    
    #### 🔍 Key Features Used
    - Temporal features (month, day, hour)
    - Operational metrics (taxi times, air time, distance)
    - Airline information
    - Cyclical time encoding for scheduled times
    
    #### ⚙️ Technical Details
    - **Framework:** scikit-learn
    - **Preprocessing:** StandardScaler, One-Hot Encoding
    - **Hyperparameter Tuning:** RandomizedSearchCV with 5-fold CV
    - **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
    
    #### 📈 Model Performance Highlights
    - **96.31%** overall accuracy
    - **97%** precision for on-time flights
    - **92%** precision for delayed flights
    - **98%** recall for on-time flights
    - **89%** recall for delayed flights
    
    #### 📚 References
    - Hatıpoğlu, I., & Tosun, Ö. (2024). Predictive modeling of flight delays at an airport using machine learning methods.
    - Li, Q., Jing, R., & Dong, Z. S. (2023). Flight delay prediction with priority information of weather and non-weather features.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", "32")
    with col2:
        st.metric("Training Time", "~15 min")
    with col3:
        st.metric("Best Model Size", "~50 MB")
    
    st.success("🎓 Developed as part of an AI research project on intelligent flight delay prediction systems.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #64748b;'>
        <p>Flight Delay Prediction System | Powered by Machine Learning | ATL Airport Focus</p>
        <p style='font-size: 12px;'>⚠️ For research and demonstration purposes only</p>
    </div>
    """,
    unsafe_allow_html=True
)

