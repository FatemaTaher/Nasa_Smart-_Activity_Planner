import streamlit as st

st.set_page_config(page_title="Smart Activity Planner", layout="wide")

CITIES = {
    "Cairo": {"lat": 30.0444, "lon": 31.2357},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Tokyo": {"lat": 35.6762, "lon": 139.6503},
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "Sydney": {"lat": -33.8688, "lon": 151.2093}
}