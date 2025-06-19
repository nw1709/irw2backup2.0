import streamlit as st
from anthropic import Anthropic
from PIL import Image
import google.generativeai as genai
import logging
import hashlib

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
st.markdown("*Debug Version - Checking OCR Source*")

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# Zeige welches Modell für OCR verwendet wird
st.sidebar.info(f"🔍 OCR Model: Gemini 1.5 Flash")

# --- OCR mit Gemini (EXPLIZIT) ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    """Extrahiert Text aus Bild MIT GEMINI"""
    try:
        st.sidebar.write("📸 Starting Gemini OCR...")
        logger.info(f"Starting GEMINI OCR for file hash: {file_hash}")
        
        # EXPLIZIT Gemini verwenden
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, formulas, and answer options (A, B, C, D, E). Do NOT interpret or solve.",
                _image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 4000
            }
        )
        
        ocr_result = response.text.strip()
        
        # Debug Info
        st.sidebar.success(f"✅ Gemini OCR completed: {len(ocr_result)} chars")
        logger.info(f"GEMINI OCR completed successfully: {len(ocr_result)} characters")
        
        return ocr_result
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        st.sidebar.error("❌ Gemini OCR failed!")
        raise e

# --- Cache leeren Button ---
if st.sidebar.button("🗑️ Clear OCR Cache"):
    st.cache_data.clear()
    st.sidebar.success("✅ Cache cleared!")
    st.rerun()

# --- Claude für Lösung ---
def solve_with_claude(ocr_text):
    """Claude löst basierend auf Gemini OCR"""
    
    st.sidebar.write("🧮 Starting Claude solver...")
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

AUFGABENTEXT (von Gemini OCR):
{ocr_text}

WICHTIGE REGELN:
1. Bei Homogenität: Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
2. Wenn nur "α + β = konstant" gegeben ist (ohne α = β), dann ist α ≠ β möglich → NICHT homogen

FORMAT:
Aufgabe [Nr]: [Antwort]
Begründung: [Erklärung auf Deutsch]"""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    st.sidebar.success("✅ Claude solving completed")
    return response.content[0].text

# --- UI ---
# Debug Info
with st.expander("🔧 System Info"):
    st.write("**OCR System:** Google Gemini 1.5 Flash")
    st.write("**Solver:** Claude 4 Opus")
    st.write("**Cache Status:** Active (1 hour TTL)")

# Datei-Upload
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        # Hash
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        st.sidebar.write(f"📄 File hash: {file_hash[:8]}...")
        
        # Bild
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        # OCR mit GEMINI
        with st.spinner("📖 OCR mit Gemini Flash..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
        
        # OCR Result anzeigen
        with st.expander("🔍 OCR-Ergebnis (von GEMINI)", expanded=True):
            st.code(ocr_text)
            st.caption(f"Extracted by: Gemini 1.5 Flash | Length: {len(ocr_text)} characters")
        
        # Lösen
        if st.button("🧮 Aufgaben lösen (mit CLAUDE)", type="primary"):
            st.markdown("---")
            
            with st.spinner("🧮 Claude löst basierend auf Gemini OCR..."):
                solution = solve_with_claude(ocr_text)
            
            # Lösung anzeigen
            st.markdown("### 📊 Lösung:")
            
            lines = solution.split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('Aufgabe'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            st.markdown(f"### {parts[0]}: **{parts[1].strip()}**")
                    elif line.startswith('Begründung:'):
                        st.markdown(f"_{line}_")
                    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("OCR: Gemini 1.5 Flash | Solver: Claude 4 Opus")
