# app.py

# استيراد المكتبات اللازمة
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import re
import matplotlib.pyplot as plt 

# استيراد المكونات من الملفات الأخرى
from config import CITIES
from data_fetcher import get_nasa_weather, create_weather_dataframe, NASA_DATA_START_YEAR
from ai_planner import generate_schedule
from utils import extract_time_from_activity

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Smart Activity Planner", layout="wide")

# --- واجهة المستخدم ---
st.title("🗓️ Smart Activity Planner")
st.markdown("---")

# --- إعدادات النموذج (Model Settings) ---
# هذا الجزء يوضح للمستخدم أن التطبيق يستخدم نموذجًا محليًا (Ollama)
with st.sidebar:
    st.header("🤖 Model Settings")
    st.info("This app uses a local Ollama model (`llama2`) for generating schedules. Please ensure Ollama is running on your machine.")

# --- باقي الكود يبقى كما هو بدون أي تغيير ---
# (من هنا إلى نهاية الملف، الكود هو نفسه الذي أرسلته)
# لقد قمت فقط بإزالة الجزء المتعلق بـ OpenAI.

selected_city = st.selectbox("Select your city:", list(CITIES.keys()))
plan_type = st.radio("Plan type:", ["Daily Plan", "Weekly Plan"])

if plan_type == "Daily Plan":
    selected_date = st.date_input("Select date:", datetime.now().date())
    activities = st.text_area("Enter your daily activities (one per line):", height=200, placeholder="Morning jog\nGrocery shopping\nPicnic in the park\nGardening\nEvening walk")
else:
    start_date = st.date_input("Start date:", datetime.now().date())
    end_date = start_date + timedelta(days=6)
    activities = st.text_area("Enter your weekly activities (one per line with day):", height=200, placeholder="Monday: Team meeting\nTuesday: Outdoor photoshoot\n...")

# --- منطق التطبيق عند الضغط على الزر ---
# لقد قمت بإزالة شرط التحقق من مفتاح API
if st.button("🧠 Create Smart Schedule"):
    if not activities:
        st.warning("Please enter your activities!")
    else:
        with st.spinner("📈 Analyzing decades of historical data to predict weather patterns..."):
            target_date = selected_date if plan_type == "Daily Plan" else start_date
            st.info(f"Analyzing historical data for {target_date.strftime('%Y-%m-%d')} based on trends from {NASA_DATA_START_YEAR} onwards.")
            
            weather_data = {}
            historical_data_for_plot = {}
            trend_data_for_plot = {}
            predicted_hourly_data_for_plot = {}
            city_coords = CITIES[selected_city]

            if plan_type == "Daily Plan":
                pred, hist, trend, pred_hourly = get_nasa_weather(city_coords, selected_date)
                if pred:
                    weather_data[selected_date] = pred
                    historical_data_for_plot[selected_date] = hist
                    trend_data_for_plot[selected_date] = trend
                    predicted_hourly_data_for_plot[selected_date] = pred_hourly
                else:
                    st.error("Could not retrieve enough historical data to make a prediction.")
                    st.stop()
            else:
                current_date = start_date
                while current_date <= end_date:
                    pred, hist, trend, pred_hourly = get_nasa_weather(city_coords, current_date)
                    if pred:
                        weather_data[current_date] = pred
                        historical_data_for_plot[current_date] = hist
                        trend_data_for_plot[current_date] = trend
                        predicted_hourly_data_for_plot[current_date] = pred_hourly
                    current_date += timedelta(days=1)
                if not weather_data:
                    st.error("Could not retrieve weather forecast for any of the selected days.")
                    st.stop()
            
            st.session_state['weather_data'] = weather_data
            st.session_state['historical_data'] = historical_data_for_plot
            st.session_state['trend_data'] = trend_data_for_plot
            st.session_state['predicted_hourly_data'] = predicted_hourly_data_for_plot
            st.session_state['activities'] = activities
            st.session_state['plan_type'] = plan_type
            st.session_state['selected_city'] = selected_city
            if plan_type == "Weekly Plan":
                st.session_state['start_date'] = start_date
                st.session_state['end_date'] = end_date

            try:
                ai_schedule = generate_schedule(
                    weather_data,
                    st.session_state.get('predicted_hourly_data', {}),
                    activities,
                    plan_type,
                    selected_city,
                    selected_date if plan_type == "Daily Plan" else None
                )
                st.session_state['ai_schedule'] = ai_schedule
                st.success("Smart schedule created successfully!")
            except Exception as e:
                st.error(str(e))

if 'weather_data' in st.session_state:
    st.subheader("🌤️ Predicted Weather Data (Based on Historical Trends)")
    weather_df = create_weather_dataframe(st.session_state['weather_data'])
    st.dataframe(weather_df, use_container_width=True)

    if 'ai_schedule' in st.session_state:
        st.subheader(f"📅 Smart Schedule for {st.session_state['selected_city']}")
        st.markdown(st.session_state['ai_schedule'])

if 'weather_data' in st.session_state:
    st.subheader("🌤️ Detailed Weather Information & Trend Analysis")
    
    weather_data = st.session_state['weather_data']
    historical_data = st.session_state['historical_data']
    trend_data = st.session_state['trend_data']
    
    for date, data in weather_data.items():
        if data:
            with st.expander(f"Weather Details for {date.strftime('%A, %B %d')}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Temperature", f"{data['temperature']:.1f}°C")
                    st.metric("Humidity", f"{data['humidity']:.1f}%")
                
                with col2:
                    st.metric("Wind Speed", f"{data['wind_speed']:.1f} m/s")
                    st.metric("Precipitation", f"{data['precipitation']:.1f} mm")
                
                with col3:
                    st.metric("Pressure", f"{data['pressure']:.1f} hPa")
                    st.metric("Solar Radiation", f"{data['solar_radiation']:.1f} W/m²")

                st.subheader("📈 Temperature Trend Analysis")
                
                hist = historical_data.get(date)
                trend = trend_data.get(date)
                
                if hist and trend and not np.isnan(trend['temperature']['slope']):
                    years = list(hist.keys())
                    temps = [hist[y]['temperature'] for y in years]
                    slope = trend['temperature']['slope']
                    intercept = trend['temperature']['intercept']
                    predicted_temp = data['temperature']

                    fig, ax = plt.subplots()
                    ax.scatter(years, temps, color='royalblue', label='Historical Data Points')
                    
                    trend_line_x = np.array([min(years), max(years), date.year])
                    trend_line_y = slope * trend_line_x + intercept
                    ax.plot(trend_line_x, trend_line_y, color='red', linestyle='--', linewidth=2, label='Trend Line')
                    
                    ax.scatter(date.year, predicted_temp, color='green', s=100, zorder=5, label=f'Predicted ({date.year})')
                    
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Temperature (°C)")
                    ax.set_title(f"Temperature Trend for {date.strftime('%B %d')} in {st.session_state['selected_city']}")
                    ax.legend()
                    ax.grid(True, linestyle=':', alpha=0.6)
                    
                    st.pyplot(fig)
                    
                    if slope > 0.05:
                        st.success("📈 The trend shows a clear **increase** in temperature over the years.")
                    elif slope < -0.05:
                        st.warning("📉 The trend shows a clear **decrease** in temperature over the years.")
                    else:
                        st.info("➡️ The temperature appears **stable** over the years with no significant trend.")
                else:
                    st.info("Not enough historical data to generate a reliable trend analysis.")

                st.subheader("🕐 Predicted Hourly Temperature")
                predicted_hourly = st.session_state.get('predicted_hourly_data', {}).get(date)
                
                if predicted_hourly:
                    hourly_df = pd.DataFrame(predicted_hourly)
                    hourly_df = hourly_df.set_index('hour')
                    st.dataframe(hourly_df[['temperature']])
                    
                    st.line_chart(hourly_df['temperature'])
                else:
                    st.info("Could not generate hourly predictions.")

if 'weather_data' in st.session_state:
    st.subheader("💡 Smart Recommendations")
    
    activities_list = [act.strip() for act in activities.split('\n') if act.strip()]
    
    for activity in activities_list:
        with st.expander(f"Recommendations for: {activity}"):
            activity_type = "unknown"
            if any(word in activity.lower() for word in ["jog", "run", "walk", "cycle", "hike"]):
                activity_type = "outdoor_exercise"
            elif any(word in activity.lower() for word in ["picnic", "park", "beach", "garden"]):
                activity_type = "outdoor_leisure"
            elif any(word in activity.lower() for word in ["photo", "shoot", "camera"]):
                activity_type = "photography"
            elif any(word in activity.lower() for word in ["meeting", "work", "office", "lecture", "virtual"]):
                activity_type = "indoor_work"
            elif any(word in activity.lower() for word in ["shop", "grocery", "mall", "buying"]):
                activity_type = "shopping"
            
            recommendations = []
            
            for date, weather in weather_data.items():
                if weather:
                    if activity_type == "outdoor_exercise":
                        if weather['temperature'] > 30:
                            recommendations.append(f"⚠️ {date.strftime('%A')}: Too hot for exercise ({weather['temperature']:.1f}°C). Try early morning or evening.")
                        elif weather['precipitation'] > 2:
                            recommendations.append(f"🌧️ {date.strftime('%A')}: Rain expected ({weather['precipitation']:.1f}mm). Consider indoor exercise.")
                        elif 18 <= weather['temperature'] <= 25 and weather['precipitation'] < 1:
                            recommendations.append(f"✅ {date.strftime('%A')}: Perfect conditions for exercise!")
                    
                    elif activity_type == "photography":
                        if weather['solar_radiation'] and weather['solar_radiation'] > 200:
                            recommendations.append(f"☀️ {date.strftime('%A')}: Great lighting for photography ({weather['solar_radiation']:.0f} W/m²)")
                        elif weather['precipitation'] > 1:
                            recommendations.append(f"🌧️ {date.strftime('%A')}: Rain may affect outdoor photography")
                    
                    elif activity_type == "outdoor_leisure":
                        if weather['temperature'] > 32:
                            recommendations.append(f"🥵 {date.strftime('%A')}: Very hot ({weather['temperature']:.1f}°C). Seek shade or indoor alternatives")
                        elif weather['wind_speed'] > 15:
                            recommendations.append(f"💨 {date.strftime('%A')}: Strong winds ({weather['wind_speed']:.1f} m/s). May affect outdoor activities")
            
            if recommendations:
                for rec in recommendations:
                    if "⚠️" in rec or "🌧️" in rec or "🥵" in rec:
                        st.warning(rec)
                    else:
                        st.success(rec)
            else:
                st.info("No specific weather concerns for this activity.")
            
            st.markdown("**General Tips:**")
            if activity_type == "outdoor_exercise":
                st.write("- Always check air quality before outdoor exercise")
                st.write("- Stay hydrated during hot weather")
                st.write("- Wear appropriate clothing for the conditions")
            elif activity_type == "photography":
                st.write("- Golden hour (sunrise/sunset) offers best lighting")
                st.write("- Protect equipment from rain and dust")
                st.write("- Use polarizing filters on bright days")
            elif activity_type == "indoor_work":
                st.write("- Ensure good lighting and ventilation")
                st.write("- Take regular breaks to stretch")
                st.write("- Minimize distractions for better focus")
            elif activity_type == "shopping":
                st.write("- Check store hours before going")
                st.write("- Avoid peak hours for less crowded experience")
                st.write("- Bring reusable bags for your purchases")

if 'ai_schedule' in st.session_state:
    st.subheader("💾 Save & Share Your Schedule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        schedule_text = st.session_state['ai_schedule']
        st.download_button(
            label="Download as Text",
            data=schedule_text,
            file_name=f"smart_schedule_{selected_city}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    with col2:
        st.info("Share your schedule:")
        share_text = f"Check out my smart schedule for {selected_city}:\n\n{schedule_text}"
        st.text_area("Copy to share:", share_text, height=100)

st.subheader("📝 Feedback & Improvement")

feedback = st.text_area("How was your experience with the smart schedule?", height=100)

if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")
    
    st.subheader("📊 Usage Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Activities Planned", len(activities_list))
    
    with col2:
        st.metric("Weather Data Points", len(weather_data))
    
    with col3:
        st.metric("City", selected_city)