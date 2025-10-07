# data_fetcher.py
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st # <-- أضف هذا السطر


# --- تعريف عام البداية للأرشيف ---
NASA_DATA_START_YEAR = 1981

def clean_nasa_value(value):
    """دالة لتحويل رمز ناسا -999 إلى قيمة فارغة (NaN)"""
    return np.nan if value == -999 else value

def get_nasa_weather_for_single_year(city_coords, date_str):
    """دالة مساعدة لجلب بيانات سنة واحدة"""
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?start={date_str}&end={date_str}"
        f"&latitude={city_coords['lat']}&longitude={city_coords['lon']}"
        f"&community=SB&parameters=T2M,RH2M,WS2M,PRECTOTCORR,PS,ALLSKY_SFC_SW_DWN"
        f"&format=JSON"
    )
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            params = data.get("properties", {}).get("parameter", {})
            return {
                "temperature": clean_nasa_value(params.get("T2M", {}).get(str(date_str), -999)),
                "humidity": clean_nasa_value(params.get("RH2M", {}).get(str(date_str), -999)),
                "wind_speed": clean_nasa_value(params.get("WS2M", {}).get(str(date_str), -999)),
                "precipitation": clean_nasa_value(params.get("PRECTOTCORR", {}).get(str(date_str), -999)),
                "pressure": clean_nasa_value(params.get("PS", {}).get(str(date_str), -999)),
                "solar_radiation": clean_nasa_value(params.get("ALLSKY_SFC_SW_DWN", {}).get(str(date_str), -999))
            }
    except:
        pass
    return None

def get_multi_year_weather_data(city_coords, target_date):
    """جلب بيانات لنفس اليوم من كل السنوات المتاحة في الأرشيف"""
    historical_data = {}
    for year in range(NASA_DATA_START_YEAR, target_date.year):
        historical_date = target_date.replace(year=year)
        date_str = historical_date.strftime("%Y%m%d")
        data = get_nasa_weather_for_single_year(city_coords, date_str)
        if data:
            historical_data[year] = data
        else:
            if not historical_data:
                st.warning(f"Could not find data for year {year}. The archive for this location might start later.")
            break
    return historical_data

def predict_weather_and_get_trend(historical_data, target_year):
    """تحليل البيانات التاريخية للتنبؤ ببيانات السنة المستهدفة وإعادة معاملات الاتجاه"""
    if not historical_data or len(historical_data) < 2:
        return None, None

    years_list = list(historical_data.keys())
    years_for_fit = np.array(years_list).reshape(-1, 1)
    
    prediction = {}
    trend_parameters = {}
    
    for param in ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure', 'solar_radiation']:
        values = np.array([historical_data[y][param] for y in years_list])
        
        valid_indices = ~np.isnan(values)
        if np.sum(valid_indices) < 2:
            prediction[param] = np.nan
            trend_parameters[param] = {'slope': np.nan, 'intercept': np.nan}
            continue
            
        clean_years = years_for_fit[valid_indices]
        clean_values = values[valid_indices]
        
        try:
            slope, intercept = np.polyfit(clean_years.flatten(), clean_values, 1)
            predicted_value = slope * target_year + intercept
            
            prediction[param] = predicted_value
            trend_parameters[param] = {'slope': slope, 'intercept': intercept}
        except:
            prediction[param] = np.nan
            trend_parameters[param] = {'slope': np.nan, 'intercept': np.nan}
            
    return prediction, trend_parameters

def get_hourly_nasa_weather(city_coords, date):
    """جلب بيانات الطقس بالساعة من ناسا لنفس اليوم من العام الماضي"""
    historical_date = date.replace(year=date.year - 1)
    date_str = historical_date.strftime("%Y%m%d")
    
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?start={date_str}&end={date_str}"
        f"&latitude={city_coords['lat']}&longitude={city_coords['lon']}"
        f"&community=SB&parameters=T2M,RH2M,WS2M,PRECTOTCORR"
        f"&format=JSON"
    )
    try:
        response = requests.get(url)
        # --- التحقق من حالة الاستجابة ---
        response.raise_for_status() # هذا السطر سيطلق خطأ إذا كانت الحالة غير 200
        
        data = response.json()
        params = data["properties"]["parameter"]
        hourly_data = []
        for hour in range(24):
            hour_str = f"{date_str}{hour:02d}00"
            hourly_data.append({
                "hour": hour,
                "temperature": clean_nasa_value(params.get("T2M", {}).get(hour_str, -999)),
                "humidity": clean_nasa_value(params.get("RH2M", {}).get(hour_str, -999)),
                "wind_speed": clean_nasa_value(params.get("WS2M", {}).get(hour_str, -999)),
                "precipitation": clean_nasa_value(params.get("PRECTOTCORR", {}).get(hour_str, -999))
            })
        
        # --- التحقق مما إذا كانت البيانات فارغة ---
        if not hourly_data:
            st.warning(f"API returned success but no hourly data was found for {date_str}.")
            return None
            
        return hourly_data
        
    # --- جعل الخطأ ظاهراً ---
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred while fetching hourly data: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as err:
        st.error(f"An unexpected error occurred: {err}")
    except KeyError:
        st.error(f"Could not parse the JSON response from NASA API for {date_str}. The structure might be different.")
        
    return None

def adjust_hourly_weather_with_trend(hourly_data_last_year, predicted_daily_temp):
    """
    تعديل بيانات الساعة بناءً على الفرق بين متوسط اليوم المتنبأ به ومتوسط العام الماضي
    """
    if not hourly_data_last_year or np.isnan(predicted_daily_temp):
        return []

    # حساب متوسط درجة حرارة العام الماضي من البيانات الساعية
    last_year_temps = [h['temperature'] for h in hourly_data_last_year if not np.isnan(h['temperature'])]
    if not last_year_temps:
        return []
        
    last_year_avg_temp = np.mean(last_year_temps)
    
    # حساب الفرق (التعديل)
    adjustment = predicted_daily_temp - last_year_avg_temp
    
    # تطبيق التعديل على كل ساعة
    adjusted_hourly_data = []
    for hour_data in hourly_data_last_year:
        adjusted_temp = hour_data['temperature'] + adjustment
        adjusted_hourly_data.append({
            "hour": hour_data['hour'],
            "temperature": adjusted_temp,
            "humidity": hour_data['humidity'], # نترك بقية القيم كما هي
            "wind_speed": hour_data['wind_speed'],
            "precipitation": hour_data['precipitation']
        })
        
    return adjusted_hourly_data

def get_nasa_weather(city_coords, date):
    """
    الدالة الرئيسية التي تجلب كل البيانات: التنبؤ اليومي، التاريخي، والساعات المعدلة
    """
    # 1. الحصول على التنبؤ اليومي والبيانات التاريخية
    historical_data = get_multi_year_weather_data(city_coords, date)
    if not historical_data:
        return None, None, None, None
        
    predicted_weather, trend_params = predict_weather_and_get_trend(historical_data, date.year)
    
    # 2. الحصول على بيانات الساعات من العام الماضي
    hourly_data_last_year = get_hourly_nasa_weather(city_coords, date)
    
    # 3. تعديل بيانات الساعات بناءً على التنبؤ
    predicted_hourly_weather = adjust_hourly_weather_with_trend(hourly_data_last_year, predicted_weather.get('temperature', np.nan))
    
    # --- هذا هو السطر الأهم: إرجاع 4 قيم ---
    return predicted_weather, historical_data, trend_params, predicted_hourly_weather

def create_weather_dataframe(weather_data):
    """تحويل بيانات الطقس إلى DataFrame للعرض"""
    if not weather_data:
        return pd.DataFrame()
    
    weather_df = pd.DataFrame.from_dict(weather_data, orient='index')
    weather_df.index.name = 'Date'
    weather_df.reset_index(inplace=True)
    weather_df['Date'] = pd.to_datetime(weather_df['Date']).dt.strftime('%Y-%m-%d')
    weather_df = weather_df.rename(columns={
        'temperature': 'Temperature (°C)',
        'humidity': 'Humidity (%)',
        'wind_speed': 'Wind Speed (m/s)',
        'precipitation': 'Precipitation (mm)',
        'pressure': 'Pressure (hPa)',
        'solar_radiation': 'Solar Radiation (W/m²)'
    })
    return weather_df