import streamlit as st
import google.generativeai as genai
from PIL import Image
import logging
import io
import pdf2image
import os

# Meta-Tags und Icon für iOS Homescreen Shortcut
st.markdown(f'''
<!-- Apple Touch Icon -->
<link rel="apple-touch-icon" sizes="180x180" href="https://em-content.zobj.net/thumbs/120/apple/325/fox-face_1f98a.png">

<!-- Web App Meta Tags -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#FF6600"> 

<!-- Optional: Splashscreen (kann später ergänzt werden) -->
''', unsafe_allow_html=True)

st.set_page_config(layout="centered", page_title="KFB1", page_icon="🦊")

st.title("🦊 Koifox-Bot 3 ")
st.write("made with deep minimal & love by fox 🚀")

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Validation ---
def validate_keys():
    if "google_api_key" not in st.secrets or not st.secrets["google_api_key"]:
        st.error("API Key Problem: 'google_api_key' in Streamlit Secrets fehlt oder ist leer.")
        st.stop()

validate_keys()

# --- API Client ---
try:
    genai.configure(api_key=st.secrets["google_api_key"])
except Exception as e:
    logger.error(f"Fehler bei der Konfiguration des Google GenAI Clients: {str(e)}")
    st.error(f"❌ Fehler bei der Initialisierung des Gemini-Clients: {str(e)}")
    st.stop()


# --- Datei in Bild konvertieren ---
def convert_to_image(uploaded_file):
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        logger.info(f"Processing file with extension: {file_extension}")

        if file_extension in ['.png', '.jpeg', '.jpg', '.gif', '.webp']:
            image = Image.open(uploaded_file)
            # Konvertiere zu RGB, falls es einen Alpha-Kanal gibt (z.B. bei PNG)
            if image.mode in ("RGBA", "P"):
                 image = image.convert("RGB")
            logger.info(f"Loaded image with format: {image.format}")
            return image

        elif file_extension == '.pdf':
            try:
                from pdf2image import convert_from_bytes
            except ImportError:
                st.error("📄 PDF-Unterstützung fehlt: Bitte `pdf2image` in requirements.txt **und** `poppler-utils` in packages.txt hinzufügen (Streamlit Cloud) oder lokal Poppler installieren.")
                st.stop()

            # erste Seite konvertieren (falls mehrere Seiten, kannst du iterieren)
            pages = convert_from_bytes(uploaded_file.read(), fmt='jpeg', dpi=300)
            if not pages:
                st.error("❌ Konnte keine Seite aus dem PDF extrahieren.")
                st.stop()
            image = pages[0].convert('RGB')
            logger.info("Converted first PDF page to image.")
            return image

        else:
            st.error(f"❌ Nicht unterstütztes Format: {file_extension}. Bitte lade PNG, JPEG, GIF, WebP oder PDF hoch.")
            st.stop()

    except Exception as e:
        logger.error(f"Error converting file to image: {str(e)}")
        st.error(f"❌ Fehler bei der Konvertierung: {str(e)}")
        return None

# --- Gemini 2.5 Pro Solver mit Bildverarbeitung ---
def solve_with_gemini(image):
    try:
        logger.info("Preparing image for Gemini 2.5 Pro")
        
        # Konfiguration für das Generierungsmodell
        generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 65535, # Maximal möglicher Wert für Gemini 2.5 Pro. [17]
        }
        
        # Sicherheits-Einstellungen (optional, aber empfohlen)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        system_instruction = """You are a PhD-level expert in 'Internes Rechnungswesen (31031)' at Fernuniversität Hagen. Solve exam questions with 100% accuracy, strictly adhering to the decision-oriented German managerial-accounting framework as taught in Fernuni Hagen lectures and past exam solutions. 

Tasks:
1. Read the task EXTREMELY carefully
2. For graphs or charts: Use only the explicitly provided axis labels, scales, and intersection points to perform calculations
3. Analyze the problem step-by-step as per Fernuni methodology
4. For multiple choice: Evaluate each option individually based solely on the given data
5. Perform a self-check: Re-evaluate your answer to ensure it aligns with Fernuni standards and the exact OCR input

CRITICAL: You MUST provide answers in this EXACT format for EVERY task found:

Aufgabe [Nr]: [Final answer]
Begründung: [1 brief but consise sentence in German]

NO OTHER FORMAT IS ACCEPTABLE."""

        prompt_parts = [
            system_instruction,
            "Extract all text from the provided exam image EXACTLY as written, including every detail from graphs, charts, or sketches. For graphs: Explicitly list ALL axis labels, ALL scales, ALL intersection points with axes (e.g., 'x-axis at 450', 'y-axis at 20'), and EVERY numerical value or annotation. Then, solve ONLY the tasks identified (e.g., Aufgabe 1). Use the following format: Aufgabe [number]: [Your answer here] Begründung: [Short explanation]. Do NOT mention or solve other tasks!",
            image
        ]
        
        logger.info("Sending request to Gemini 2.5 Pro")
        response = model.generate_content(prompt_parts)
        logger.info("Received response from Gemini 2.5 Pro")
        
        return response.text

    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        st.error(f"❌ Gemini API Fehler: {str(e)}")
        return None


# --- HAUPTINTERFACE ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)

uploaded_file = st.file_uploader("**Klausuraufgabe hochladen...**", type=["png", "jpg", "jpeg", "gif", "webp", "pdf"])

if uploaded_file is not None:
    try:
        image = convert_to_image(uploaded_file)
        if image:
            if "rotation" not in st.session_state:
                st.session_state.rotation = 0

            if st.button("Bild drehen"):
                st.session_state.rotation = (st.session_state.rotation + 90) % 360

            rotated_img = image.rotate(-st.session_state.rotation, expand=True)

            st.image(rotated_img, caption=f"Verarbeitetes Bild (gedreht um {st.session_state.rotation}°)", use_container_width=True)

            if st.button("🧮 Aufgabe(n) lösen", type="primary"):
                st.markdown("---")
                with st.spinner("Gemini 2.5 Pro analysiert..."):
                    gemini_solution = solve_with_gemini(rotated_img)

                if gemini_solution:
                    st.markdown("### 🎯 FINALE LÖSUNG")
                    st.markdown(gemini_solution)
                    if debug_mode:
                        with st.expander("🔍 Gemini Rohausgabe"):
                            st.code(gemini_solution)
                else:
                    st.error("❌ Keine Lösung generiert")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"❌ Fehler bei der Verarbeitung: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made by Fox & Koi-9 ❤️ | Google Gemini 2.5 Pro")
