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
st.markdown("*Single-Model System mit verbessertem Prompting*")

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- OCR mit Caching ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    """Extrahiert Text aus Bild - gecached basierend auf file_hash"""
    try:
        logger.info(f"Starting OCR for file hash: {file_hash}")
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, and answer options (A, B, C, D, E). Do NOT interpret or solve.",
                _image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 4000
            }
        )
        logger.info("OCR completed successfully")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- Claude Solver mit verbesserten Prompts ---
def solve_with_claude(ocr_text, iteration=1):
    """Claude löst die Aufgabe mit expliziten Regeln"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

KRITISCHE REGELN (SEHR WICHTIG!):
- Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Wenn nur α + β gegeben ist (ohne α = β), ist die Funktion NICHT homogen
- Homogenitätsgrad k bedeutet: f(λr) = λ^k·f(r) für ALLE λ
- Prüfe IMMER ob die Bedingungen für mathematische Eigenschaften erfüllt sind
- Mache KEINE unbegründeten Annahmen

ANALYSIERE DIESEN TEXT (Iteration {iteration}):
{ocr_text}

DENKE SCHRITT FÜR SCHRITT:
1. Was ist gegeben?
2. Was ist gefragt?
3. Welche Bedingungen müssen für die Eigenschaft erfüllt sein?
4. Sind diese Bedingungen erfüllt?
5. Was ist die korrekte Antwort?

FORMAT (WICHTIG):
Aufgabe [Nr]: [Antwort - NUR Buchstabe(n) oder Zahl]
Begründung: [Kurze Erklärung auf Deutsch mit Fachbegriffen]

Beispiel:
Aufgabe 1: CD
Begründung: Die Produktionsfunktion ist nicht homogen, da α ≠ β im allgemeinen Fall."""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0.1,  # Leicht erhöht für besseres Reasoning
        system="Du bist ein präziser Mathematik-Experte. Bei Homogenität: Eine Funktion ist NUR homogen wenn ALLE Bedingungen erfüllt sind. Prüfe kritisch!",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- Selbst-Verifikation ---
def verify_solution(ocr_text, first_solution):
    """Claude verifiziert seine eigene Lösung"""
    
    verify_prompt = f"""Du bist ein ZWEITER, unabhängiger Experte. Prüfe diese Lösung SEHR KRITISCH:

AUFGABENTEXT:
{ocr_text}

ZU PRÜFENDE LÖSUNG:
{first_solution}

KRITISCHE PRÜFPUNKTE:
- Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Hat der erste Experte unbegründete Annahmen gemacht?
- Stimmen die mathematischen Schlussfolgerungen?
- Ist die Antwort wirklich korrekt?

Wenn die Lösung FALSCH ist, gib die KORREKTE Lösung.
Wenn sie RICHTIG ist, bestätige sie.

FORMAT:
PRÜFUNG: [KORREKT/FALSCH]
Aufgabe [Nr]: [Finale Antwort]
Begründung: [Erklärung]"""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0.2,  # Etwas höher für kritisches Denken
        messages=[{"role": "user", "content": verify_prompt}]
    )
    
    return response.content[0].text

# --- UI ---
# Debug-Modus
debug_mode = st.checkbox("🔍 Debug-Modus", value=False, help="Zeigt Zwischenschritte")

# Zwei-Durchgänge Option
use_verification = st.checkbox("✅ Mit Selbst-Verifikation", value=True, help="Claude prüft seine eigene Lösung")

# Datei-Upload
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        # Eindeutiger Hash für die Datei
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Bild laden und anzeigen
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        # OCR (gecached)
        with st.spinner("📖 Lese Text mit Gemini Flash..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
            
        # Debug: OCR-Ergebnis anzeigen
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis", expanded=False):
                st.code(ocr_text)
        
        # Button zum Lösen
        if st.button("🧮 Aufgaben lösen", type="primary"):
            st.markdown("---")
            
            # Erste Lösung
            with st.spinner("🧮 Claude löst die Aufgabe..."):
                first_solution = solve_with_claude(ocr_text, iteration=1)
            
            if debug_mode:
                with st.expander("🔍 Erste Lösung"):
                    st.code(first_solution)
            
            # Verifikation wenn aktiviert
            if use_verification:
                with st.spinner("✅ Claude verifiziert die Lösung..."):
                    final_solution = verify_solution(ocr_text, first_solution)
                
                if debug_mode:
                    with st.expander("🔍 Verifizierte Lösung"):
                        st.code(final_solution)
            else:
                final_solution = first_solution
            
            # Ergebnisse anzeigen
            st.markdown("### 📊 Lösung:")
            
            # Formatierte Ausgabe
            lines = final_solution.split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('Aufgabe'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            st.markdown(f"### {parts[0]}: **{parts[1].strip()}**")
                        else:
                            st.markdown(f"### {line}")
                    elif line.startswith('Begründung:'):
                        st.markdown(f"*{line}*")
                    elif line.startswith('PRÜFUNG:'):
                        if 'KORREKT' in line:
                            st.success("✅ Lösung wurde verifiziert")
                        else:
                            st.warning("⚠️ Lösung wurde korrigiert")
                    else:
                        if line.strip() and not line.startswith('---'):
                            st.markdown(line)
                    
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Made by Fox | Claude Only mit verbessertem Prompting")
