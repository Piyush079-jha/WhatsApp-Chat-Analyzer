import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helper

st.set_page_config(layout="wide")
st.title("📊 WhatsApp Chat Analyzer")

# File uploader
uploaded_file = st.file_uploader("Choose a WhatsApp chat file (.txt)", type="txt")

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    data = bytes_data.decode("utf-8")
    df = helper.preprocess(data)

    st.sidebar.title("Chat Participants")
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Select User", user_list)

    # Preserve analysis display using session state
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    if st.sidebar.button("Show Analysis"):
        st.session_state.show_analysis = True

    # Sidebar Chatbox
    st.sidebar.title("💬 Live Emotion Detection")

    chat_mode = st.sidebar.radio("Choose Mode", ["Custom Message", "Select From Chat"])
    user_message = ""

    if chat_mode == "Custom Message":
        user_message = st.sidebar.text_input("Enter your message")
    else:
        selected_message = st.sidebar.selectbox("Select a message", df['message'].unique())
        user_message = selected_message

    if st.sidebar.button("Detect Emotion"):
        if user_message.strip():
            emotion = helper.detect_emotion(user_message)
            st.sidebar.success(f"Detected Emotion: {emotion}")
        else:
            st.sidebar.warning("Please enter or select a message first.")

    # Main Analysis Section
    if st.session_state.show_analysis:
        st.header("Top Statistics")
        num_messages, words, num_media, links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media)
        col4.metric("Links Shared", links)

        # Monthly Timeline
        st.header("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Daily Timeline
        st.header("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Activity Maps
        st.header("Activity Maps")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Busy Days")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.subheader("Most Busy Months")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Heatmap
        st.header("Weekly Activity Heatmap")
        heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap, ax=ax, cmap='YlGnBu')
        st.pyplot(fig)

        # Busiest Users
        if selected_user == 'Overall':
            st.header("Most Active Users")
            col1, col2 = st.columns(2)
            x, new_df = helper.most_busy_users(df)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='teal')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.header("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most Common Words
        st.header("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1], color='coral')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Emoji Analysis
        st.header("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            if not emoji_df.empty:
                fig, ax = plt.subplots()
                ax.pie(emoji_df['count'].head(10), labels=emoji_df['emoji'].head(10), autopct="%0.2f%%")
                st.pyplot(fig)
