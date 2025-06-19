import streamlit as st
from anthropic import Anthropic
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import re

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Validation ---
def validate_keys():
    required_keys = {
        'gemini_key': ('AIza', "Gemini"),
        'claude_key': ('sk-ant', "Claude")
    }
    missing = []
    invalid = []
    
    for key, (prefix, name) in required_keys.items():
        if key not in st.secrets:
            missing.append(name)
        elif not st.secrets[key].startswith(prefix):
            invalid.append(name)
    
    if missing or invalid:
        st.error(f"API Key Problem: Missing {', '.join(missing)} | Invalid {', '.join(invalid)}")
        st.stop()

validate_keys()

# --- UI-Einstellungen ---
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")
st.markdown("*Verbesserte OCR Version*")

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- Verbesserte OCR ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini_improved(_image, file_hash):
    """Extrahiert KOMPLETTEN Text aus Bild"""
    try:
        logger.info(f"Starting GEMINI OCR for file hash: {file_hash}")
        
        # Stelle sicher, dass das Bild nicht zu groß ist
        # Aber behalte genug Auflösung für Text
        max_size = 3000
        if _image.width > max_size or _image.height > max_size:
            _image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            st.sidebar.warning(f"Bild wurde auf {max_size}px skaliert")
        
        # Verbesserter OCR Prompt
        response = vision_model.generate_content(
            [
                """Extract ALL text from this exam image. 
                IMPORTANT:
                - Read EVERYTHING from top to bottom
                - Include ALL questions, formulas, values, and answer options
                - Include question numbers like "Aufgabe 45 (5 RP)"
                - Include ALL multiple choice options (A, B, C, D, E) with their complete text
                - Include mathematical symbols and formulas exactly as shown
                - DO NOT summarize or skip any part
                - DO NOT solve anything
                
                Start from the very top and continue to the very bottom of the image.""",
                _image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 8000  # Erhöht!
            }
        )
        
        ocr_result = response.text.strip()
        
        # Prüfe ob genug Text extrahiert wurde
        if len(ocr_result) < 500:
            st.warning(f"⚠️ Nur {len(ocr_result)} Zeichen extrahiert - möglicherweise unvollständig!")
        
        logger.info(f"GEMINI OCR completed: {len(ocr_result)} characters")
        return ocr_result
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- Claude Solver mit besserer Ausgabe ---
def solve_with_claude_formatted(ocr_text):
    """Claude löst und formatiert korrekt"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

VOLLSTÄNDIGER AUFGABENTEXT:
{ocr_text}

WICHTIGE REGELN:
1. Identifiziere ALLE Aufgaben im Text (z.B. "Aufgabe 45", "Aufgabe 46" etc.)
2. Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
3. Beantworte JEDE Aufgabe die du findest

AUSGABEFORMAT (STRIKT EINHALTEN):
Aufgabe [Nummer]: [Antwort]
Begründung: [Erklärung]

Wiederhole dies für JEDE Aufgabe im Text.

Beispiel:
Aufgabe 45: 500
Begründung: Der Parameter a entspricht dem Achsenabschnitt bei p = 0, also a = 500.

Aufgabe 46: b
Begründung: Um Parameter b zu bestimmen...

WICHTIG: Vergiss keine Aufgabe!"""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=4000,
        temperature=0.1,
        system="Beantworte ALLE Aufgaben die im Text stehen. Überspringe keine.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- Verbesserte Ausgabeformatierung ---
def parse_and_display_solution(solution_text):
    """Parst und zeigt Lösung strukturiert an"""
    
    # Finde alle Aufgaben mit Regex
    task_pattern = r'Aufgabe\s+(\d+)\s*:\s*([^\n]+)'
    tasks = re.findall(task_pattern, solution_text, re.IGNORECASE)
    
    if not tasks:
        st.warning("⚠️ Keine Aufgaben im erwarteten Format gefunden")
        # Zeige trotzdem die Rohantwort
        st.markdown(solution_text)
        return
    
    # Zeige jede Aufgabe strukturiert
    for task_num, answer in tasks:
        st.markdown(f"### Aufgabe {task_num}: **{answer.strip()}**")
        
        # Finde zugehörige Begründung
        begr_pattern = rf'Aufgabe\s+{task_num}\s*:.*?\n\s*Begründung:\s*([^\n]+(?:\n(?!Aufgabe)[^\n]+)*)'
        begr_match = re.search(begr_pattern, solution_text, re.IGNORECASE | re.DOTALL)
        
        if begr_match:
            st.markdown(f"*Begründung: {begr_match.group(1).strip()}*")
        
        st.markdown("---")

# --- UI ---
# Cache leeren
if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# Debug
debug_mode = st.checkbox("🔍 Debug-Modus", value=True)

# Datei-Upload
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Bild verarbeiten
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        image = Image.open(uploaded_file)
        
        # Zeige Original
        st.image(image, caption=f"Originalbild ({image.width}x{image.height}px)", use_container_width=True)
        
        # OCR
        with st.spinner("📖 Lese KOMPLETTEN Text mit Gemini..."):
            ocr_text = extract_text_with_gemini_improved(image, file_hash)
        
        # OCR Ergebnis
        with st.expander(f"🔍 OCR-Ergebnis ({len(ocr_text)} Zeichen)", expanded=debug_mode):
            st.code(ocr_text)
            
            # Prüfe ob Aufgaben gefunden wurden
            found_tasks = re.findall(r'Aufgabe\s+\d+', ocr_text, re.IGNORECASE)
            if found_tasks:
                st.success(f"✅ Gefundene Aufgaben: {', '.join(found_tasks)}")
            else:
                st.error("❌ Keine Aufgaben im Text gefunden!")
        
        # Lösen
        if st.button("🧮 Alle Aufgaben lösen", type="primary"):
            st.markdown("---")
            
            with st.spinner("🧮 Claude löst ALLE Aufgaben..."):
                solution = solve_with_claude_formatted(ocr_text)
            
            if debug_mode:
                with st.expander("💭 Rohe Claude-Antwort"):
                    st.code(solution)
            
            # Formatierte Ausgabe
            st.markdown("### 📊 Lösungen:")
            parse_and_display_solution(solution)
                    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Koifox-Bot | Verbesserte OCR & Formatierung")
