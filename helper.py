import re
import pandas as pd
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
from textblob import TextBlob

extract = URLExtract()

# -------------------------
# Preprocess Function (Android + iOS Support)
# -------------------------
def preprocess(data):
    pattern_android = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?[APMapm]{2})\s?-\s(.*)'
    pattern_ios = r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\]\s(.*)'

    android = re.findall(pattern_android, data)
    ios = re.findall(pattern_ios, data)

    messages = []

    if android:
        for date, time, message in android:
            full_msg = f"{date}, {time}"
            msg_parts = re.split(r'([\w\W]+?):\s', message)
            if len(msg_parts) > 2:
                user = msg_parts[1]
                msg = msg_parts[2]
            else:
                user = "group_notification"
                msg = msg_parts[0]
            messages.append([f"{date}, {time}", user, msg])
    elif ios:
        for date, time, message in ios:
            msg_parts = re.split(r'([\w\W]+?):\s', message)
            if len(msg_parts) > 2:
                user = msg_parts[1]
                msg = msg_parts[2]
            else:
                user = "group_notification"
                msg = msg_parts[0]
            messages.append([f"{date}, {time}", user, msg])

    df = pd.DataFrame(messages, columns=["date", "user", "message"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df.dropna(subset=["date"], inplace=True)

    # Add extra columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month_name()
    df["month_num"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["period"] = df["hour"].apply(lambda x: f"{x}-{x+1}")
    df["only_date"] = df["date"].dt.date

    return df


# -------------------------
# Stats & Analysis Functions
# -------------------------
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(str(message).split())

    num_media_messages = df[df['message'].astype(str).str.contains("<Media omitted>")].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(str(message)))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    busy_users = df['user'].value_counts().head()
    user_percent_df = (df['user'].value_counts() / df.shape[0]) * 100
    user_percent_df = user_percent_df.reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    user_percent_df['percent'] = user_percent_df['percent'].round(2)
    return busy_users, user_percent_df


def create_wordcloud(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['user'] != 'group_notification') & (~df['message'].astype(str).str.contains("<Media omitted>"))]

    def remove_stop_words(message):
        return " ".join(word for word in str(message).lower().split() if word not in stop_words)

    temp['message'] = temp['message'].apply(remove_stop_words)
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(" ".join(temp['message']))
    return df_wc


def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['user'] != 'group_notification') & (~df['message'].astype(str).str.contains("<Media omitted>"))]

    words = []
    for message in temp['message']:
        for word in str(message).lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([ch for ch in str(message) if ch in emoji.EMOJI_DATA])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))), columns=["emoji", "count"])
    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = [f"{row['month']}-{row['year']}" for idx, row in timeline.iterrows()]
    timeline['time'] = time
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return heatmap


# -------------------------
# Emotion Detection Function
# -------------------------
def detect_emotion(message):
    analysis = TextBlob(str(message))
    polarity = analysis.sentiment.polarity

    if polarity > 0.5:
        return "😄 Very Happy"
    elif polarity > 0:
        return "🙂 Happy"
    elif polarity == 0:
        return "😐 Neutral"
    elif polarity > -0.5:
        return "😕 Sad"
    else:
        return "😠 Angry"
