import streamlit as st
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF
import io

try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("Bitte konfigurieren Sie Ihren Google API Key in den Streamlit Secrets.")


# --- Hardcoded Expert Prompt ---
# This is the specific, fixed prompt that will be used for every request.
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
# Initialize the Gemini 1.5 Pro model for its powerful OCR and reasoning
model = genai.GenerativeModel('gemini-1.5-pro-latest')


# --- Helper Function for PDF to Images ---
def pdf_to_images(pdf_file):
    """Converts a PDF file to a list of PIL Images."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Increase resolution for better OCR quality
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
st.set_page_config(page_title="Fernuni Exam Solver", layout="wide")
st.title("Koifox Bot 3 - Gemini 2.5 Pro")
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
        if uploaded_file.type == "application/pdf":
            with st.spinner("PDF wird in Bild umgewandelt..."):
                images = pdf_to_images(uploaded_file)
                if images:
                    # Store images in session state to avoid reprocessing
                    st.session_state.images = images
                    st.image(images, caption=[f"Seite {i+1}" for i in range(len(images))], use_column_width=True)
        else:
            try:
                image = Image.open(uploaded_file)
                # Store image in session state
                st.session_state.images = [image]
                st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
            except Exception as e:
                st.error(f"Fehler beim Öffnen der Bilddatei: {e}")

with col2:
    st.header("Lösung")
    if solve_button and uploaded_file:
        if 'images' in st.session_state and st.session_state.images:
            with st.spinner("Gemini analysiert und löst die Aufgaben..."):
                try:
                    # Prepare the content for the API call
                    # The first part is the fixed prompt, followed by the image(s)
                    prompt_parts = [EXPERT_PROMPT] + st.session_state.images

                    # Call the Gemini API
                    response = model.generate_content(prompt_parts)

                    # Display the response
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"Ein Fehler bei der Kommunikation mit der Gemini API ist aufgetreten: {e}")
        else:
             st.warning("Die Datei konnte nicht verarbeitet werden. Bitte laden Sie sie erneut hoch.")

    elif solve_button:
        st.warning("Bitte laden Sie zuerst eine Klausurdatei hoch.")
