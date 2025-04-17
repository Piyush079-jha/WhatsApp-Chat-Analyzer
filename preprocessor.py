import re
import pandas as pd

def preprocess(data):
    """
    Parses WhatsApp chat data from Android or iOS formats.
    Returns a DataFrame with date, user, and message columns.
    """
    # Android: 12/04/2023, 10:34 AM - John: Message
    pattern_android = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?[APMapm]{2})\s?-\s(.*?):\s(.*)'

    # iOS: [12/04/2023, 10:34:15] John: Message
    pattern_ios = r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\]\s(.*?):\s(.*)'

    messages = []

    # Check Android format first
    matches = re.findall(pattern_android, data)
    if matches:
        for date, time, author, msg in matches:
            messages.append([f"{date}, {time}", author.strip(), msg.strip()])
        df = pd.DataFrame(messages, columns=["date", "user", "message"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        return df.dropna()

    # Check iOS format if Android not found
    matches = re.findall(pattern_ios, data)
    if matches:
        for date, time, author, msg in matches:
            messages.append([f"{date}, {time}", author.strip(), msg.strip()])
        df = pd.DataFrame(messages, columns=["date", "user", "message"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        return df.dropna()

    # Return empty DataFrame with expected columns if nothing matched
    return pd.DataFrame(columns=["date", "user", "message"])
