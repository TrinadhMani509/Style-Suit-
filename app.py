import streamlit as st
from PIL import Image
import base64
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="StyleSense AI", layout="centered")

st.title("👗 StyleSense AI - Personal Fashion Assistant")
st.write("Upload your photo and get AI-powered outfit recommendations!")

# Upload image
uploaded_file = st.file_uploader("Upload your image", type=["jpg","png","jpeg"])

occasion = st.selectbox(
    "Select Occasion",
    ["Casual Outing", "College", "Interview", "Party", "Wedding", "Festival"]
)

def encode_image(image):
    return base64.b64encode(image.read()).decode("utf-8")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Style Recommendation"):

        with st.spinner("Analyzing your style..."):

            base64_image = encode_image(uploaded_file)

            prompt = f"""
You are a professional fashion stylist.

Analyze the person in the image and give personalized fashion advice for a {occasion}.

Provide:
1. Outfit recommendations
2. Best color combinations
3. Footwear suggestion
4. Accessories suggestion
5. Grooming tips
6. Confidence/body language tips

Answer in clean bullet points.
"""

            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=800,
            )

            result = response.choices[0].message.content

            st.subheader("✨ Your Personalized Style Guide")
            st.write(result)
