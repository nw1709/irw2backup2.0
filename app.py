import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageEnhance
import logging
import io
import pdf2image
import os
import pillow_heif

# --- VORBEREITUNG ---

st.markdown(f'''
<!-- Apple Touch Icon -->
<link rel="apple-touch-icon" sizes="180x180" href="https://em-content.zobj.net/thumbs/120/apple/325/fox-face_1f98a.png">
<!-- Web App Meta Tags -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#FF6600"> 
''', unsafe_allow_html=True)

st.set_page_config(layout="centered", page_title="KFB3", page_icon="🦊")
st.title("🦊 Koifox-Bot 3 ")
st.write("made with deep minimal & love by fox 🚀")

# --- Logger & API Key Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_keys():
    if "google_api_key" not in st.secrets or not st.secrets["google_api_key"]:
        st.error("API Key Problem: 'google_api_key' in Streamlit Secrets fehlt oder ist leer.")
        st.stop()
validate_keys()

try:
    genai.configure(api_key=st.secrets["google_api_key"])
except Exception as e:
    st.error(f"❌ Fehler bei der Initialisierung des Gemini-Clients: {str(e)}")
    st.stop()

# --- BILDVERARBEITUNG & OPTIMIERUNG ---
def process_and_prepare_image(uploaded_file):
    # Diese Funktion ist exakt identisch mit der GPT-5-Version für einen fairen Vergleich.
    try:
        pillow_heif.register_heif_opener()
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in ['.png', '.jpeg', '.jpg', '.gif', '.webp', '.heic']:
            image = Image.open(uploaded_file)
        elif file_extension == '.pdf':
            pages = pdf2image.convert_from_bytes(uploaded_file.read(), fmt='jpeg', dpi=300)
            if not pages:
                st.error("❌ Konnte keine Seite aus dem PDF extrahieren.")
                return None
            image = pages[0]
        else:
            st.error(f"❌ Nicht unterstütztes Format: {file_extension}.")
            return None
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        image_gray = image.convert('L')
        enhancer = ImageEnhance.Contrast(image_gray)
        image_enhanced = enhancer.enhance(1.5)
        final_image = image_enhanced.convert('RGB')
        return final_image
    except Exception as e:
        logger.error(f"Fehler bei der Bildverarbeitung: {str(e)}")
        return None

# --- Gemini 2.5 Pro Solver ---
def solve_with_gemini(image):
    try:
        logger.info("Bereite Anfrage für Gemini 2.5 Pro vor")
        
        generation_config = {
            "temperature": 0.1, # Auf 0.1 für maximale Präzision gesetzt
            "max_output_tokens": 8192,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # KORRIGIERT: Festgelegt auf die verifizierte, stabile Version von Gemini 2.5 Pro
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        system_prompt = """
        [Persona & Wissensbasis]
        Du bist ein wissenschaftlicher Mitarbeiter und Korrektor am Lehrstuhl für Internes Rechnungswesen der Fernuniversität Hagen (Modul 31031). Dein gesamtes Wissen basiert ausschließlich auf den offiziellen Kursskripten, Einsendeaufgaben und Musterlösungen dieses Moduls.
        [Verbot von externem Wissen]
        Ignoriere strikt und ausnahmslos alle Lösungswege, Formeln oder Methoden von anderen Universitäten, aus allgemeinen Lehrbüchern oder von Online-Quellen. Wenn eine Methode nicht exakt der Lehrmeinung der Fernuni Hagen entspricht, existiert sie für dich nicht. Deine Loyalität gilt zu 100% dem Fernuni-Standard.
        [Lösungsprozess]
        1. Analyse: Lies die Aufgabe und die gegebenen Daten (inkl. Graphen) mit äußerster Sorgfalt.
        2. Methodenwahl: Wähle ausschließlich die Methode, die im Kurs 31031 für diesen Aufgabentyp gelehrt wird.
        3. Schritt-für-Schritt-Lösung: Zeige deinen Lösungsweg transparent und nachvollziehbar auf, so wie es in einer Klausur erwartet wird. Benenne die verwendeten Formeln gemäß der Fernuni-Terminologie.
        4. Selbstkorrektur: Überprüfe dein Ergebnis kritisch und frage dich: "Ist dies exakt der Weg, den der Lehrstuhl in einer Musterlösung zeigen würde?"
        [Output-Format]
        Gib deine finale Antwort zwingend im folgenden Format aus. Fasse dich in der Begründung kurz und prägnant.
        Aufgabe [Nr]: [Finales Ergebnis]
        Begründung: [Kurze 1-Satz-Erklärung des Ergebnisses basierend auf der Fernuni-Methode.]
        """

        user_prompt = "Lies die Informationen aus dem bereitgestellten Bild. Löse anschließend die darauf sichtbare Aufgabe gemäß deiner Anweisungen und halte dich strikt an das geforderte Ausgabeformat."
        
        prompt_parts = [system_prompt, user_prompt, image]
        
        logger.info("Sende Anfrage an Gemini...")
        response = model.generate_content(prompt_parts)
        logger.info("Antwort von Gemini erhalten.")
        
        return response.text

    except Exception as e:
        logger.error(f"Gemini API Fehler: {str(e)}")
        st.error(f"❌ Gemini API Fehler: {str(e)}")
        return None

# --- HAUPTINTERFACE ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)
uploaded_file = st.file_uploader("**Klausuraufgabe hochladen...**", type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "heic"])
if uploaded_file is not None:
    try:
        processed_image = process_and_prepare_image(uploaded_file)
        if processed_image:
            if "rotation" not in st.session_state: st.session_state.rotation = 0
            if st.button("🔄 Bild drehen"): st.session_state.rotation = (st.session_state.rotation + 90) % 360
            rotated_img = processed_image.rotate(-st.session_state.rotation, expand=True)
            st.image(rotated_img, caption=f"Optimiertes Bild (gedreht um {st.session_state.rotation}°)", use_container_width=True)
            if st.button("🧮 Aufgabe(n) lösen", type="primary"):
                st.markdown("---")
                with st.spinner("Gemini 2.5 Pro analysiert das Bild..."):
                    gemini_solution = solve_with_gemini(rotated_img)
                if gemini_solution:
                    st.markdown("### 🎯 FINALE LÖSUNG")
                    st.markdown(gemini_solution)
                    if debug_mode:
                        with st.expander("🔍 Gemini Rohausgabe"): st.code(gemini_solution)
                else:
                    st.error("❌ Keine Lösung generiert")
    except Exception as e:
        logger.error(f"Fehler im Hauptprozess: {str(e)}")
        st.error(f"❌ Ein unerwarteter Fehler ist aufgetreten: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made by Fox & Koi-9 ❤️ | Google Gemini 2.5 Pro (stable)")
