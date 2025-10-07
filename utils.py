import re

def extract_time_from_activity(activity):
    time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)'
    match = re.search(time_pattern, activity)
    
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3).upper()
        
        if period == "PM" and hour != 12:
            hour += 12
        elif period == "AM" and hour == 12:
            hour = 0
        return hour * 60 + minute
    return None