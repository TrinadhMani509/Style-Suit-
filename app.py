import streamlit as st
from PIL import Image
import os
from groq import Groq
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# -------------------- LOAD KEYS --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Please add GROQ_API_KEY inside .env file")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -------------------- LOAD VISION MODEL --------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

# -------------------- UI --------------------
st.set_page_config(page_title="StyleSense AI", layout="centered")

st.title("👗 StyleSense AI - Personal Fashion Assistant")
st.write("Upload your photo and get AI-powered outfit recommendations!")

uploaded_file = st.file_uploader("Upload your image", type=["jpg","png","jpeg"])

occasion = st.selectbox(
    "Select Occasion",
    ["Casual Outing", "College", "Interview", "Party", "Wedding", "Festival"]
)

# -------------------- PROCESS --------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Style Recommendation"):

        with st.spinner("Analyzing your outfit... 🤖👕"):

            # Step 1: Describe the image
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            image_description = processor.decode(out[0], skip_special_tokens=True)

            st.info(f"Detected: {image_description}")

            # Step 2: Send to Groq stylist AI
            prompt = f"""
You are a professional fashion stylist.

The person in the image is described as:
{image_description}

Give personalized fashion advice for a {occasion}.

Provide:
1. Outfit recommendations
2. Best color combinations
3. Footwear suggestion
4. Accessories suggestion
5. Grooming tips
6. Confidence/body language tips

Answer in clear bullet points.
"""

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert fashion stylist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )

            result = response.choices[0].message.content

            st.subheader("✨ Your Personalized Style Guide")
            st.write(result)
