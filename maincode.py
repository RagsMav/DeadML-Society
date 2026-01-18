import streamlit as st
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import ssl
import requests                         
from streamlit_lottie import st_lottie
import time

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


if 'intro_done' not in st.session_state:
    st.session_state['intro_done'] = False

if not st.session_state['intro_done']:
    intro_placeholder = st.empty()
    
    intro_lottie = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_w51pcehl.json")

    with intro_placeholder.container():
        st.write("##")
        st.write("##")
        
        st.markdown("""
            <h1 style='text-align: center; font-size: 70px; color: #FF4B4B;'>
                DeadMLSociety
            </h1>
            <h3 style='text-align: center; color: grey;'>
                Intelligent Systems. No Limits.
            </h3>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if intro_lottie:
                st_lottie(intro_lottie, height=300, key="intro_anim")
        time.sleep(4)
    
    intro_placeholder.empty()
    
    st.session_state['intro_done'] = True

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

st.set_page_config(page_title="Group Chat Vibe Checker", page_icon="😎", layout="wide")
nltk.download('vader_lexicon', quiet=True)

def parse_uploaded_file(uploaded_file):
    file_content = uploaded_file.getvalue().decode("utf-8")
    
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
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%y, %I:%M %p', errors='coerce')
    return df


def analyze_data(df):
    user_stats = df['Author'].value_counts().reset_index()
    user_stats.columns = ['Author', 'Message Count']
    
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

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

col1, col2 = st.columns([1, 4])

with col1:
    st_lottie(lottie_coding, height=150, key="coding")

with col2:
    st.title("😎 Group Chat Vibe-Checker")
    st.write("Upload your WhatsApp export file (.txt) to see who is who.")


uploaded_file = st.file_uploader("Choose a file", type=['txt'])

if uploaded_file is not None:
    with st.spinner('Parsing chat logs...'):
        df = parse_uploaded_file(uploaded_file)
        
    if df.empty:
        st.error("❌ Could not parse the file. Ensure it is a valid WhatsApp text export.")
    else:

        stats = analyze_data(df)
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
