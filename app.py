import streamlit as st
import base64
from io import BytesIO
import PyPDF2
from pdf2image import convert_from_path
from groq import Groq
import os
import pandas as pd
import re

# ===== CONFIG =====
st.set_page_config(page_title="AI OCR Extractor", layout="wide")
api_key = st.secrets["GROQ_API_KEY"]  # 👈 stored safely in Streamlit Secrets
client = Groq(api_key=api_key)

# ===== OCR CORE =====
def image_to_base64(image_source, is_pil=False):
    buf = BytesIO()
    if is_pil:
        image_source.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    else:
        with open(image_source, "rb") as f:
            return base64.b64encode(f.read()).decode()

def ocr_extract(base64_image):
    prompt = """Extract clear key-value data from this document:
    - Keep tables, names, addresses, dates, numbers.
    - Output clean structured text only."""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role":"user",
            "content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{base64_image}"}}
            ]
        }],
        max_tokens=2000,
        temperature=0.1
    )
    return response.choices[0].message.content

def extract_fields(text):
    patterns = {
        "Name": r"Name[:\s]+([A-Za-z\s]+)",
        "Address": r"Address[:\s]+([A-Za-z0-9,\-\s]+)",
        "Email": r"Email[:\s]+([\w\.-]+@[\w\.-]+)",
        "Phone": r"Phone[:\s]+(\+?\d[\d\s-]{6,})"
    }
    data = {}
    for k,p in patterns.items():
        m = re.search(p,text,re.I)
        if m: data[k]=m.group(1).strip()
    return data

# ===== STREAMLIT UI =====
st.title("📄 AI-Powered OCR & Data Extractor")

uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf","png","jpg","jpeg"])
if uploaded_file:
    file_bytes = uploaded_file.read()
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path,"wb") as f: f.write(file_bytes)

    # Preview
    st.subheader("Preview")
    if uploaded_file.type=="application/pdf":
        st.info("PDF uploaded. Showing first page below:")
        images = convert_from_path(temp_path, first_page=1, last_page=1, dpi=200)
        buf = BytesIO()
        images[0].save(buf, format="PNG")
        st.image(buf.getvalue(), use_container_width=True)
        base64_img = image_to_base64(images[0], is_pil=True)
    else:
        st.image(uploaded_file, use_container_width=True)
        base64_img = image_to_base64(temp_path)

    if st.button("🚀 Extract Data"):
        with st.spinner("Extracting text..."):
            text = ocr_extract(base64_img)
            fields = extract_fields(text)

        st.subheader("🧾 Extracted Raw Text")
        st.text_area("Extracted Text", text, height=250)

        if fields:
            st.subheader("📋 Structured Data")
            df = pd.DataFrame(list(fields.items()), columns=["Field","Value"])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download CSV", data=csv, file_name="extracted_data.csv")
        else:
            st.warning("No structured fields detected. Check the OCR output.")

    os.remove(temp_path)
