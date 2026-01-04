import streamlit as st
import tempfile
import os

from fusion import multimodal_fusion
from text_risk import text_risk_score
from audio_risk import audio_risk_score
from safety import crisis_detect
from multilingual import translate_to_english

# --------------------------------------------------
# ENV CHECK
# --------------------------------------------------
IS_CLOUD = "STREAMLIT_SERVER_RUNNING" in os.environ

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="MindCare",
    page_icon="💙",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
body { background-color: #0e1117; }

.card {
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.text-card { background: linear-gradient(135deg, #1e3c72, #2a5298); }
.audio-card { background: linear-gradient(135deg, #134e5e, #71b280); }
.video-card { background: linear-gradient(135deg, #42275a, #734b6d); }
.result-card { background: rgba(255,255,255,0.04); }

.soft-text { color: #e0e0e0; opacity: 0.85; }
.risk-low { color: #00e676; font-weight: 700; }
.risk-mid { color: #ffd54f; font-weight: 700; }
.risk-high { color: #ff5252; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<center>
<h1>💙 MindCare</h1>
<p class='soft-text'>Multimodal Emotional Insight System</p>
</center>
""", unsafe_allow_html=True)

st.info("This tool provides emotional insight and support — not a medical diagnosis.")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🧭 Controls")
st.sidebar.caption("Webcam analysis works only in local deployment.")
st.sidebar.caption("Data is processed locally and not stored.")

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
st.markdown("## 🧩 Inputs")

col1, col2, col3 = st.columns(3)

# -------- TEXT --------
with col1:
    st.markdown("<div class='card text-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Text Input")
    text = st.text_area(
        "",
        height=150,
        placeholder="I feel tired, overwhelmed, and low lately..."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------- AUDIO --------
with col2:
    st.markdown("<div class='card audio-card'>", unsafe_allow_html=True)
    st.markdown("### 🎤 Audio Input")
    audio_file = st.file_uploader("", type=["wav"])
    st.markdown("</div>", unsafe_allow_html=True)

# -------- VIDEO (DISABLED ON CLOUD) --------
video_risk = None
with col3:
    st.markdown("<div class='card video-card'>", unsafe_allow_html=True)
    st.markdown("### 📷 Video Input")
    st.markdown(
        "<div class='soft-text'>Webcam analysis is available in local deployment only.</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
text_risk = None
audio_risk = None
reasons = []

if text and text.strip():
    translated_text, detected_lang = translate_to_english(text)

    if crisis_detect(translated_text):
        st.error("🚨 Please reach out to a trusted person or mental health professional immediately.")
        st.stop()

    text_risk = text_risk_score(translated_text)
    if detected_lang != "en":
        st.caption(f"🌍 Detected language: `{detected_lang}` (translated to English)")

    if text_risk == 2:
        reasons.append("Strong negative sentiment detected in text")
    elif text_risk == 1:
        reasons.append("Mild negative sentiment detected in text")

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_risk = audio_risk_score(tmp.name)

    if audio_risk == 2:
        reasons.append("Sad or fearful tone detected in voice")
    elif audio_risk == 1:
        reasons.append("Stressed vocal tone detected")

# --------------------------------------------------
# RESULT
# --------------------------------------------------
st.markdown("<div class='card result-card'>", unsafe_allow_html=True)
st.markdown("## 🧠 Emotional Insight")

if text_risk is None and audio_risk is None:
    st.markdown("<div class='soft-text'>Provide text or audio input to see insights.</div>", unsafe_allow_html=True)
else:
    final_risk = multimodal_fusion(text_risk, audio_risk, None)

    if final_risk == "LOW":
        st.markdown("<span class='risk-low'>LOW RISK</span>", unsafe_allow_html=True)
        st.progress(0.35)
    elif final_risk == "MEDIUM":
        st.markdown("<span class='risk-mid'>MODERATE RISK</span>", unsafe_allow_html=True)
        st.progress(0.65)
    else:
        st.markdown("<span class='risk-high'>HIGH RISK</span>", unsafe_allow_html=True)
        st.progress(0.9)

    st.markdown("### 🔍 Why this result?")
    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("• No strong negative emotional indicators detected.")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<center class='soft-text'>Built with 💙 using NLP & Speech Emotion Recognition</center>",
    unsafe_allow_html=True
)
