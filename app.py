import streamlit as st
import tempfile
import cv2

from fusion import multimodal_fusion
from text_risk import text_risk_score
from audio_risk import audio_risk_score
from video_risk import video_risk_score
from safety import crisis_detect
from multilingual import translate_to_english

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="MindCare",
    page_icon="💙",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS (COLORS + POLISH)
# --------------------------------------------------
st.markdown("""
<style>

/* Base */
body {
    background-color: #0e1117;
}

/* Generic Card */
.card {
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    transition: all 0.25s ease;
}

/* Hover */
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

/* Cool color cards */
.text-card {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}

.audio-card {
    background: linear-gradient(135deg, #134e5e, #71b280);
}

.video-card {
    background: linear-gradient(135deg, #42275a, #734b6d);
}

.result-card {
    background: rgba(255,255,255,0.04);
    animation: fadeUp 0.6s ease-out;
}

/* Text */
.soft-text {
    color: #e0e0e0;
    font-size: 14px;
    opacity: 0.85;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    color: #9aa0a6;
    font-size: 18px;
}

/* Risk colors */
.risk-low { color: #00e676; font-weight: 700; }
.risk-mid { color: #ffd54f; font-weight: 700; }
.risk-high { color: #ff5252; font-weight: 700; }

/* Animation */
@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Mobile */
@media (max-width: 768px) {
    .big-title { font-size: 32px; }
    .subtitle { font-size: 16px; }
    .card { padding: 18px; }
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; margin-bottom:30px;'>
        <div class='big-title'>💙 MindCare</div>
        <div class='subtitle'>Multimodal Emotional Insight System</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.info("This tool provides emotional insight and support — not a medical diagnosis.")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🧭 Controls")

def toggle_camera():
    st.session_state.camera_on = not st.session_state.camera_on

st.sidebar.checkbox(
    "Enable Webcam",
    value=st.session_state.camera_on,
    on_change=toggle_camera
)

st.sidebar.caption("Data is processed locally and not stored.")

# --------------------------------------------------
# INPUT CARDS
# --------------------------------------------------
st.markdown("## 🧩 Inputs")
st.caption("You may use any one input or a combination of all three.")

col1, col2, col3 = st.columns(3)

# -------- TEXT CARD --------
with col1:
    st.markdown("<div class='card text-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Text Input")
    st.markdown("<div class='soft-text'>Express how you feel in any language</div>", unsafe_allow_html=True)

    text = st.text_area(
        "",
        height=150,
        placeholder="I feel tired, overwhelmed, and low lately..."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------- AUDIO CARD --------
with col2:
    st.markdown("<div class='card audio-card'>", unsafe_allow_html=True)
    st.markdown("### 🎤 Audio Input")
    st.markdown("<div class='soft-text'>Your voice tone and emotion are analyzed</div>", unsafe_allow_html=True)

    audio_file = st.file_uploader(
        "",
        type=["wav"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------- VIDEO CARD --------
video_risk = None

with col3:
    st.markdown("<div class='card video-card'>", unsafe_allow_html=True)
    st.markdown("### 📷 Video Input")
    st.markdown("<div class='soft-text'>Facial expressions are analyzed</div>", unsafe_allow_html=True)

    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_box = st.image([])

        ret, frame = cap.read()
        if ret:
            video_risk = video_risk_score(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_box.image(frame)

        cap.release()
    else:
        st.markdown("<div class='soft-text'>Enable webcam from sidebar</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
text_risk = None
audio_risk = None
reasons = []

# ---- TEXT (MULTILINGUAL) ----
if text is not None and len(text.strip()) > 0:
    translated_text, detected_lang = translate_to_english(text)

    if crisis_detect(translated_text):
        st.error("🚨 Please reach out to a trusted person or mental health professional immediately.")
        st.stop()
    else:
        text_risk = text_risk_score(translated_text)

        if detected_lang != "en":
            st.caption(f"🌍 Detected language: `{detected_lang}` (translated to English)")

        if text_risk == 2:
            reasons.append("Strong negative sentiment detected in text")
        elif text_risk == 1:
            reasons.append("Mild negative sentiment detected in text")

# ---- AUDIO ----
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_risk = audio_risk_score(tmp.name)

    if audio_risk == 2:
        reasons.append("Sad or fearful tone detected in voice")
    elif audio_risk == 1:
        reasons.append("Stressed or tense vocal tone detected")

# --------------------------------------------------
# RESULT (ALWAYS VISIBLE + EXPLAINABLE)
# --------------------------------------------------
st.markdown("<div class='card result-card'>", unsafe_allow_html=True)
st.markdown("## 🧠 Emotional Insight")

if text_risk is None and audio_risk is None and video_risk is None:
    st.markdown(
        "<div class='soft-text'>Provide text, voice, or enable webcam to see insights.</div>",
        unsafe_allow_html=True
    )
else:
    final_risk = multimodal_fusion(text_risk, audio_risk, video_risk)
    confidence_map = {"LOW": 35, "MEDIUM": 65, "HIGH": 90}
    confidence = confidence_map[final_risk]

    if final_risk == "LOW":
        st.markdown("<span class='risk-low'>LOW RISK</span>", unsafe_allow_html=True)
        st.progress(0.35)
    elif final_risk == "MEDIUM":
        st.markdown("<span class='risk-mid'>MODERATE RISK</span>", unsafe_allow_html=True)
        st.progress(0.65)
    else:
        st.markdown("<span class='risk-high'>HIGH RISK</span>", unsafe_allow_html=True)
        st.progress(0.9)

    st.markdown(f"**Confidence:** `{confidence}%`")

    # -------- EXPLANATION --------
    st.markdown("### 🔍 Why this result?")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        if final_risk == "LOW":
            st.write("• No strong negative emotional indicators were detected.")
            st.write("• Your text, voice, and facial expressions appear emotionally stable.")
        elif final_risk == "MEDIUM":
            st.write("• Some mild stress-related emotional signals were detected.")
            st.write("• These may indicate temporary emotional strain.")
        else:
            st.write("• Multiple strong negative emotional indicators were detected.")
            st.write("• This suggests a higher level of emotional distress.")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<center class='soft-text'>Built with 💙 using NLP, Speech Emotion Recognition & Computer Vision</center>",
    unsafe_allow_html=True
)
