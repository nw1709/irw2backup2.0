import streamlit as st
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF
import io

st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot 3")
st.markdown("*Gemini 2.5 Pro*")


try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("Bitte konfigurieren Sie Ihren Google API Key in den Streamlit Secrets.")
    st.stop() # Stop the app if the key is not configured

# --- Hardcoded Expert Prompt ---
EXPERT_PROMPT = """
You are a PhD-level expert in 'Internes Rechnungswesen (31031)' at Fernuniversität Hagen. Solve exam questions with 100% accuracy, strictly adhering to the decision-oriented German managerial-accounting framework as taught in Fernuni Hagen lectures and past exam solutions. 

Tasks:
1. Read the task EXTREMELY carefully
2. For graphs or charts: Use only the explicitly provided axis labels, scales, and intersection points to perform calculations
3. Analyze the problem step-by-step as per Fernuni methodology
4. For multiple choice: Evaluate each option individually based solely on the given data 
5. Provide answers in this EXACT format for EVERY task found:
Aufgabe [Nr]: [Final answer]
Begründung: [One brief but concise sentence in german]

CRITICAL: You MUST perform a self-check: ALWAYS re-evaluate your answer by checking the provided data to absolutely ensure it aligns with Fernuni standards 100%!
"""

# --- Gemini Model Initialization ---
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# --- Helper Function for PDF to Images ---
@st.cache_data # Use caching to avoid re-converting the same PDF
def pdf_to_images(pdf_bytes):
    """Converts a PDF file's bytes to a list of PIL Images."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            images.append(image)
        pdf_document.close()
        return images
    except Exception as e:
        st.error(f"Fehler bei der Konvertierung der PDF-Datei: {e}")
        return []

# --- Streamlit App Interface ---
st.title("Koifox Bot 3 - Gemini 1.5 Pro")
st.write("Lade die Aufgabe hoch.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Steuerung")
    uploaded_file = st.file_uploader(
        "Klausurdatei hochladen",
        type=["jpg", "jpeg", "png", "pdf"]
    )
    solve_button = st.button("✨ Aufgaben lösen")

# --- Main Content Area ---
col1, col2 = st.columns(2)

with col1:
    st.header("Hochgeladenes Dokument")
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.type == "application/pdf":
            with st.spinner("PDF wird in Bild umgewandelt..."):
                # Pass the file bytes to the function
                images = pdf_to_images(file_bytes)
                if images:
                    st.session_state.images = images
                    # FIX: Replaced use_column_width with use_container_width
                    st.image(images, caption=[f"Seite {i+1}" for i in range(len(images))], use_container_width=True)
        else:
            try:
                image = Image.open(io.BytesIO(file_bytes))
                st.session_state.images = [image]
                # FIX: Replaced use_column_width with use_container_width
                st.image(image, caption="Hochgeladenes Bild", use_container_width=True)
            except Exception as e:
                st.error(f"Fehler beim Öffnen der Bilddatei: {e}")

with col2:
    st.header("Lösung")
    if solve_button:
        if 'images' in st.session_state and st.session_state.images:
            with st.spinner("Gemini analysiert und löst die Aufgaben..."):
                try:
                    prompt_parts = [EXPERT_PROMPT] + st.session_state.images
                    response = model.generate_content(prompt_parts)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Ein Fehler bei der Kommunikation mit der Gemini API ist aufgetreten: {e}")
        else:
             st.warning("Bitte laden Sie zuerst eine Datei hoch oder die Datei konnte nicht verarbeitet werden.")
