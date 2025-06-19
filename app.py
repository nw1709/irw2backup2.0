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
st.markdown("*Pragmatische Version - Balance zwischen Genauigkeit und Funktionalität*")

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- OCR Standard ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    """Extrahiert Text aus Bild"""
    try:
        logger.info(f"Starting OCR for file hash: {file_hash}")
        
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, formulas, and answer options (A, B, C, D, E). Be precise with mathematical notation.",
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

# --- Claude Pragmatisch ---
def solve_with_claude_pragmatic(ocr_text):
    """Claude löst mit Balance zwischen Striktheit und Praktikabilität"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

AUFGABENTEXT:
{ocr_text}

WICHTIGE REGELN:
1. Bei Homogenität: Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
2. Wenn nur "α + β = konstant" gegeben ist (ohne α = β), dann ist α ≠ β möglich → NICHT homogen
3. Verwende primär die Informationen aus dem Text
4. Nutze Standard-Definitionen aus dem Controlling wenn nötig

ARBEITSSCHRITTE:
1. Identifiziere was gegeben ist
2. Identifiziere was gefragt ist  
3. Wende die relevanten Konzepte an
4. Berechne/bestimme die Antwort
5. Prüfe dein Ergebnis

FORMAT (WICHTIG):
Aufgabe [Nr]: [Antwort - nur Buchstabe(n) oder Zahl]
Begründung: [Präzise Erklärung auf Deutsch]

Beispiel:
Aufgabe 1: CD
Begründung: Die Funktion ist nicht homogen (C richtig), da bei α + β = 3 nicht notwendig α = β gilt. D ist auch richtig, weil..."""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0.1,  # Leicht erhöht für flexibleres Denken
        system="Du bist ein Experte für deutsches Controlling. Sei präzise aber pragmatisch. Fokussiere dich auf die korrekte Lösung.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- Verbesserte Lösungsanzeige ---
def display_solution(solution_text):
    """Zeigt die Lösung strukturiert an"""
    if not solution_text:
        st.error("Keine Lösung generiert")
        return
        
    lines = solution_text.split('\n')
    current_task = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Aufgabe erkennen (verschiedene Formate)
        if any(line.startswith(prefix) for prefix in ['Aufgabe', 'AUFGABE', 'Task']):
            if ':' in line:
                parts = line.split(':', 1)
                task = parts[0].strip()
                answer = parts[1].strip()
                st.markdown(f"### {task}: **{answer}**")
                current_task = task
            else:
                st.markdown(f"### {line}")
                
        # Begründung erkennen
        elif any(line.startswith(prefix) for prefix in ['Begründung:', 'BEGRÜNDUNG:', 'Erklärung:']):
            st.markdown(f"_{line}_")
            
        # Andere relevante Zeilen
        elif current_task and line and not line.startswith('---'):
            # Zusätzliche Erklärungen in kleiner Schrift
            st.markdown(f"<small>{line}</small>", unsafe_allow_html=True)

# --- Quick Validation ---
def quick_validate(solution, ocr_text):
    """Schnelle Validierung ohne zu strikt zu sein"""
    # Prüfe nur ob Aufgabennummern übereinstimmen
    import re
    
    ocr_tasks = set(re.findall(r'Aufgabe\s*(\d+)', ocr_text, re.IGNORECASE))
    solution_tasks = set(re.findall(r'Aufgabe\s*(\d+)', solution, re.IGNORECASE))
    
    if ocr_tasks and solution_tasks:
        if not ocr_tasks.intersection(solution_tasks):
            return False, "Aufgabennummern stimmen nicht überein"
    
    return True, "OK"

# --- UI ---
# Optionen
col1, col2 = st.columns(2)
with col1:
    debug_mode = st.checkbox("🔍 Debug-Modus", value=False)
with col2:
    strict_mode = st.checkbox("🔒 Strikter Modus", value=False, help="Strengere Validierung")

# Datei-Upload
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        # Hash und Bild
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        # OCR
        with st.spinner("📖 Lese Text..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
            
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis"):
                st.code(ocr_text)
        
        # Lösen
        if st.button("🧮 Aufgaben lösen", type="primary"):
            st.markdown("---")
            
            # Lösung generieren
            with st.spinner("🧮 Claude löst die Aufgabe..."):
                solution = solve_with_claude_pragmatic(ocr_text)
            
            if debug_mode:
                with st.expander("💭 Rohe Lösung"):
                    st.code(solution)
            
            # Quick Validation
            valid, msg = quick_validate(solution, ocr_text)
            if not valid and strict_mode:
                st.warning(f"⚠️ Validierungswarnung: {msg}")
            
            # Lösung anzeigen
            st.markdown("### 📊 Lösung:")
            display_solution(solution)
            
            # Confidence Indicator
            if solution:
                if "nicht sicher" in solution.lower() or "unklar" in solution.lower():
                    st.warning("⚠️ Claude ist sich bei dieser Lösung unsicher")
                else:
                    st.success("✅ Lösung generiert")
                    
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")
        
        if "credit balance" in str(e).lower():
            st.error("💳 API-Credits aufgebraucht!")

# --- Footer ---
st.markdown("---")
st.caption("Made by Fox | Pragmatic Balance Version")
