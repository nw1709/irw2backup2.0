import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from PIL import Image
import fitz  # PyMuPDF
import io
import re
import base64
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# --- SEITE EINRICHTEN ---
st.set_page_config(layout="wide", page_title="Koifox-Bot 4.2", page_icon="🦊")
st.title("🦊 Koifox-Bot 3: Multi-Experten-Validierung")
st.markdown("Gemini 2.5 Pro, GPT o3 & Claude Opus 4.1")

# --- API CLIENT INITIALISIERUNG ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    
    GEMINI_MODEL_NAME = "gemini-1.5-pro-latest" 
    GPT_MODEL_NAME = "o3"
    CLAUDE_MODEL_NAME = "claude-opus-4-1-20250805"
    
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

except (KeyError, Exception) as e:
    st.error(f"Fehler bei der Initialisierung der API-Clients. Bitte prüfen Sie Ihre API-Keys in den Streamlit Secrets. Fehler: {e}")
    st.stop()


# --- PROMPTS ---
EXPERT_PROMPT = """
Sie sind ein deutscher Professor für 'Internes Rechnungswesen' (Kurs 31031) an der Fernuniversität Hagen mit dem Ziel, Klausuraufgaben mit 100%iger Genauigkeit zu lösen. Ihre Methodik muss exakt dem entscheidungsorientierten deutschen Controlling-Ansatz entsprechen, wie er in den Kursmaterialien der Fernuni Hagen gelehrt wird.

Ihre Aufgaben:
1.  **Analyse**: Lesen Sie die Aufgabenstellung EXTREM sorgfältig. Identifizieren Sie alle gegebenen Zahlen, Bedingungen und die exakte Fragestellung.
2.  **Diagramme/Graphen**: Verwenden Sie zur Berechnung ausschließlich explizit angegebene Achsenbeschriftungen, Skalenwerte und Schnittpunkte. Extrapolieren Sie nicht und schätzen Sie keine Werte.
3.  **Schritt-für-Schritt-Lösung**: Entwickeln Sie Ihre Lösung stringent nach der Methodik der Fernuni Hagen. Legen Sie Ihre Rechenschritte intern offen.
4.  **Multiple Choice**: Bewerten Sie jede einzelne Option (A, B, C, D) individuell und begründen Sie, warum sie richtig oder falsch ist, basierend auf Ihren Berechnungen.
5.  **Formatierung**: Geben Sie die finale Antwort für JEDE gefundene Aufgabe in diesem EXAKTEN Format aus. Fügen Sie keine weiteren Erklärungen außerhalb dieses Formats hinzu.

Aufgabe [Nummer der Aufgabe]: [Finale Antwort, z.B. eine Zahl, ein Buchstabe oder ein kurzer Satz]
Begründung: [Ein einzelner, prägnanter Satz, der die Herleitung auf den Punkt bringt.]

**Selbstprüfung (KRITISCH)**: Bevor Sie Ihre Antwort ausgeben, überprüfen Sie Ihr Ergebnis nochmals anhand der Daten aus der Aufgabenstellung. Stellen Sie absolut sicher, dass es zu 100% den Standards und der Methodik der Fernuni Hagen entspricht.
"""

# --- HELFERFUNKTIONEN ---
@st.cache_data
def pdf_to_images(pdf_bytes):
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)))
        pdf_document.close()
        return images
    except Exception as e:
        st.error(f"Fehler bei der PDF-Konvertierung: {e}")
        return []

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_solution(text):
    pattern = re.compile(r"Aufgabe\s*\[?(\d+)\]?:\s*(.*?)\s*\nBegründung:\s*(.*)", re.IGNORECASE)
    matches = pattern.findall(text)
    return {match[0]: {"answer": match[1].strip(), "reason": match[2].strip()} for match in matches}


# --- API-AUFRUFFUNKTIONEN ---
def call_gemini(image_list):
    try:
        prompt_parts = [EXPERT_PROMPT] + image_list
        response = gemini_model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"Fehler bei Gemini API: {e}"

def call_gpt(base64_image_list):
    messages = [{"role": "user", "content": [{"type": "text", "text": EXPERT_PROMPT}] + [{"type": "image_url
