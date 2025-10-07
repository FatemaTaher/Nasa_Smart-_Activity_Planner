# ai_planner.py
import requests
import json

def generate_schedule(weather_data, hourly_weather_data, activities, plan_type, city, selected_date=None):
    """بناء الـ prompt وإرساله إلى Ollama لإنشاء الجدول"""

    # تجهيز نص الطقس
    weather_text = "Weather Forecast:\n"
    hourly_weather_text = "Hourly Weather Details:\n"
    
    for date, data in weather_data.items():
        if data:
            weather_text += f"{date.strftime('%A, %B %d')}:\n"
            weather_text += f"  Temperature: {data['temperature']:.1f}°C\n"
            weather_text += f"  Humidity: {data['humidity']:.1f}%\n"
            weather_text += f"  Wind Speed: {data['wind_speed']:.1f} m/s\n"
            weather_text += f"  Precipitation: {data['precipitation']:.1f} mm\n"
            weather_text += f"  Solar Radiation: {data['solar_radiation']:.1f} W/m²\n\n"
            
            hourly_weather_text += f"{date.strftime('%A, %B %d')} Hourly Details:\n"
            if date in hourly_weather_data and hourly_weather_data[date]:
                for hour_data in hourly_weather_data[date]:
                    hour = hour_data['hour']
                    temp = hour_data['temperature']
                    humidity = hour_data['humidity']
                    wind = hour_data['wind_speed']
                    precip = hour_data['precipitation']
                    hourly_weather_text += f"  {hour:02d}:00 - Temp: {temp:.1f}°C, Humidity: {humidity:.1f}%, Wind: {wind:.1f} m/s, Precip: {precip:.1f} mm\n"
            hourly_weather_text += "\n"

    # بناء الـ prompt حسب نوع الخطة
    if plan_type == "Daily Plan":
        user_prompt = f"""
        Create an optimized schedule for a SINGLE DAY ({selected_date.strftime('%A, %B %d')}) in {city} based on the weather forecast and the user's activities.
        
        {weather_text}
        {hourly_weather_text}
        
        User's Activities for {selected_date.strftime('%A, %B %d')}:
        {activities}
        
        CRITICAL INSTRUCTIONS:
        1. This is a DAILY PLAN - schedule ALL activities on the SAME DAY ({selected_date.strftime('%A, %B %d')})
        2. Do NOT spread activities across multiple days
        3. Assign specific times to each activity based on weather conditions
        4. Consider these factors:
           - Avoid outdoor activities during extreme heat (>30°C) or heavy rain (>5mm)
           - Prefer moderate temperatures (18-25°C) for physical activities
           - Consider wind speed for cycling or outdoor events
           - Utilize solar radiation for photography or solar-related activities
           - Plan indoor activities during poor weather conditions
        5. If an activity already has a time specified, try to honor it if weather permits
        6. Provide detailed weather information for each time slot
        
        Format your response as:
        ## Optimized Schedule for {selected_date.strftime('%A, %B %d')}
        [Time]: [Activity] - [Weather conditions at that time and reason for scheduling]
        
        ## Weather Conditions Summary for {selected_date.strftime('%A, %B %d')}
        [Detailed summary of weather conditions for the day]
        
        ## Detailed Hourly Schedule
        [Time]: [Activity] - [Weather conditions at that time]
        
        ## Weather-Based Recommendations
        [Specific tips for each activity]
        
        ## Alternative Plans
        [Backup suggestions for poor weather]
        
        ## Explanation of Schedule Logic
        [Detailed explanation of why activities were scheduled at specific times based on weather conditions]
        """
    else:
        user_prompt = f"""
        Based on the weather forecast for {city} and the user's activities, create an optimized weekly schedule.
        
        {weather_text}
        {hourly_weather_text}
        
        User's Activities:
        {activities}
        
        Instructions:
        1. Analyze each activity and determine the best day based on weather conditions.
        2. Consider these factors:
           - Avoid outdoor activities during extreme heat (>30°C) or heavy rain (>5mm)
           - Prefer moderate temperatures (18-25°C) for physical activities
           - Consider wind speed for cycling or outdoor events
           - Utilize solar radiation for photography or solar-related activities
           - Plan indoor activities during poor weather conditions
           - If an activity requires leaving home NEVER assign it to days with temperature >30°C or precipitation >3mm.
           - If an activity is clearly indoors , you can assign it on any day.
           - Always prefer days with moderate weather for outdoor tasks.

        3. Assign specific times to each activity based on hourly weather conditions
        4. Provide detailed weather information for each time slot
        
        Format your response as:
        ## Optimized Weekly Schedule
        [Day, Time]: [Activity] - [Weather conditions at that time and reason for scheduling]
        
        ## Weather Conditions Summary
        [Detailed summary of weather conditions for each day]
        
        ## Detailed Daily Schedules
        ### [Day]
        [Time]: [Activity] - [Weather conditions at that time]
        
        ## Weather-Based Recommendations
        [Specific tips for each activity]
        
        ## Alternative Plans
        [Backup suggestions for poor weather]
        
        ## Explanation of Schedule Logic
        [Detailed explanation of why activities were scheduled on specific days and times based on weather conditions]
        """
    
    # دمج رسالة النظام مع طلب المستخدم
    system_message = "You are a smart activity planner that creates optimized schedules based on weather conditions."
    full_prompt = f"{system_message}\n\n{user_prompt}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": full_prompt, "stream": True},
            stream=True
        )
        
        output = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    output += data["response"]
        
        return output
            
    except Exception as e:
        raise Exception(f"Error connecting to Ollama: {e}")
