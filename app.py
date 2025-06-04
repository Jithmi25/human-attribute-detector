import streamlit as st
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="AI Human Attribute Detector", layout="wide")

if "theme" not in st.session_state:
    st.session_state["theme"] = "Light Mode"

theme = st.radio("Choose theme mode:", ["Light Mode", "Dark Mode"], horizontal=True, index=0 if st.session_state["theme"] == "Light Mode" else 1)
st.session_state["theme"] = theme

if theme == "Light Mode":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: white;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)
  
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")

def convert_emotion_to_emoji(text):
    emoji_map = {
        "Joyful": "😄",
        "Happy": "😊",
        "Sad": "😢",
        "Angry": "😠",
        "Excited": "🤩",
        "Neutral": "😐",
        "Focused": "🧐",
        "Surprised": "😲",
        "Confused": "😕"
    }
    for key in emoji_map:
        if key.lower() in text.lower():
            return emoji_map[key]
    return ""

def analyze_human_attributes(image):
    prompt = """
    You are an AI trained to analyze human attributes from images with high accuracy.
    Carefully analyze the given image and return the following structured details:

    - Gender (Male/Female/Non-binary)
    - Age Estimate (e.g., 25 years)
    - Ethnicity (e.g., Asian, Caucasian, African, etc.)
    - Mood (e.g., Happy, Sad, Neutral, Excited)
    - Facial Expression (e.g., Smiling, Frowning, Neutral, etc.)
    - Glasses (Yes/No)
    - Beard (Yes/No)
    - Hair Color (e.g., Black, Blonde, Brown)
    - Eye Color (e.g., Blue, Green, Brown)
    - Headwear (Yes/No, specify type if applicable)
    - Emotions Detected (e.g., Joyful, Focused, Angry, etc.)
    - Confidence Level (Accuracy of prediction in percentage)

    Also provide a short detailed explanation describing how each trait was interpreted from the image.
    """
    response = model.generate_content([prompt, image])
    return response.text.strip()

st.markdown("<h1 style='text-align: center;'>AI Human Attribute Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and get detailed AI-based analysis</p>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload a clear image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    img = PIL.Image.open(uploaded_image)

    with st.spinner("Analyzing image..."):
        response_text = analyze_human_attributes(img)

    attributes_text = []
    explanation_text = []
    for line in response_text.split("\n"):
        if ":" in line and any(keyword in line.lower() for keyword in ["gender", "age", "ethnicity", "mood", "expression", "glasses", "beard", "hair", "eye", "headwear", "emotions", "confidence"]):
            attributes_text.append(line.strip())
        elif line.strip():
            explanation_text.append(line.strip())

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### 🔍 Detected Attributes")
        for line in attributes_text:
            if "Emotions" in line:
                label, value = line.split(":", 1)
                st.write(f"{label.strip()}: {value.strip()} {convert_emotion_to_emoji(line)}")
            elif "Confidence" in line:
                try:
                    confidence = int(''.join(filter(str.isdigit, line)))
                    st.write(f"{line.strip()}")
                    st.progress(confidence / 100)
                except:
                    st.write(line.strip())
            else:
                st.write(line.strip())

    st.markdown("---")
    st.markdown("### 📘 Detailed Explanations:")
    for line in explanation_text:
        st.write(line)

    st.success("✅ Analysis complete!")

st.markdown("---")
st.markdown("<small style='text-align:center;display:block;'>Powered by Gemini AI & Streamlit</small>", unsafe_allow_html=True)
