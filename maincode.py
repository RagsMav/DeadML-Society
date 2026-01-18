import streamlit as st
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import ssl

# --- 1. EMERGENCY SSL FIX (For Mac) ---
# This ensures NLTK can download data without crashing
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- 2. SETUP ---
st.set_page_config(page_title="Group Chat Vibe Checker", page_icon="🚀", layout="wide")
nltk.download('vader_lexicon', quiet=True)

# --- 3. THE PARSER (Converts Text to Data) ---
def parse_uploaded_file(uploaded_file):
    # Read the uploaded file directly from memory
    file_content = uploaded_file.getvalue().decode("utf-8")
    
    # Regex for your format: 21/07/25, 10:04 am - Name: Message
    pattern = r'^(\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}\s?[apAP][mM])\s-\s(.*?):\s(.*)$'
    
    data = []
    for line in file_content.split('\n'):
        line = line.strip().replace('\u200e', '')
        match = re.match(pattern, line)
        if match:
            timestamp, author, message = match.groups()
            data.append([timestamp, author, message])
            
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data, columns=['DateTime', 'Author', 'Message'])
    # Convert DateTime
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%y, %I:%M %p', errors='coerce')
    return df

# --- 4. THE ANALYZER (Calculates Stats) ---
def analyze_data(df):
    # Message Count
    user_stats = df['Author'].value_counts().reset_index()
    user_stats.columns = ['Author', 'Message Count']
    
    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = {}
    
    for author in df['Author'].unique():

        msgs = df[df['Author'] == author]['Message'].astype(str).tolist()

        subset = msgs[-500:]
        score = sum([sia.polarity_scores(m)['compound'] for m in subset]) / len(subset) if subset else 0
        sentiment_scores[author] = score
        
    user_stats['Vibe Score'] = user_stats['Author'].map(sentiment_scores)
    

    def get_archetype(row):
        msgs = row['Message Count']
        vibe = row['Vibe Score']
        if msgs < 5: return "👻 The Ghost"
        if vibe > 0.3: return "😇 The Saint"
        if vibe < -0.1 or msgs>100: return "💀 The Menace"
        if msgs > user_stats['Message Count'].quantile(0.9): return "📢 The Yapper"
        return "😐 NPC"

    user_stats['Archetype'] = user_stats.apply(get_archetype, axis=1)
    return user_stats

# --- 5. THE MAIN APP INTERFACE ---
st.title("😎 WhatsApp Chat Vibe-Checker")
st.write("Upload your WhatsApp export file (.txt) to see who is who.")

uploaded_file = st.file_uploader("Choose a file", type=['txt'])

if uploaded_file is not None:
    with st.spinner('Parsing chat logs...'):
        df = parse_uploaded_file(uploaded_file)
        
    if df.empty:
        st.error("❌ Could not parse the file. Ensure it is a valid WhatsApp text export.")
    else:

        stats = analyze_data(df)
        
        # --- DISPLAY STATS ---

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Messages", len(df))
        c2.metric("Active Members", df['Author'].nunique())
        

        valid_dates = df['DateTime'].dropna()
        top_date = str(valid_dates.dt.date.mode()[0]) if not valid_dates.empty else "N/A"
        c3.metric("Most Active Date", top_date)
        
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 The Leaderboard")
            fig = px.bar(stats.head(100), x='Message Count', y='Author', 
                         color='Archetype', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("🆔 Personality Profiles")
            st.dataframe(stats[['Author', 'Archetype', 'Vibe Score']], hide_index=True)
