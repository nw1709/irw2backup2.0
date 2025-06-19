import streamlit as st
from anthropic import Anthropic
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import json

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
st.markdown("*Anti-Halluzinations-Version*")

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- OCR mit Validation ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    """Extrahiert Text aus Bild mit Validierung"""
    try:
        logger.info(f"Starting OCR for file hash: {file_hash}")
        
        # Erste OCR
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, and answer options (A, B, C, D, E). Do NOT interpret or solve. Do NOT add any commentary.",
                _image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 4000
            }
        )
        
        ocr_text = response.text.strip()
        
        # Validierung - zweiter Durchgang zur Sicherheit
        validation_response = vision_model.generate_content(
            [
                f"Compare this text with the image and confirm it's accurate:\n{ocr_text}\n\nRespond with 'ACCURATE' if correct or list any errors.",
                _image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 500
            }
        )
        
        if "ACCURATE" not in validation_response.text.upper():
            logger.warning(f"OCR validation failed: {validation_response.text}")
        
        logger.info("OCR completed successfully")
        return ocr_text
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- Claude mit Anti-Halluzination ---
def solve_with_claude_strict(ocr_text):
    """Claude löst mit strikten Anti-Halluzinations-Regeln"""
    
    # Zuerst: Lass Claude die Aufgabe zusammenfassen
    summary_prompt = f"""Fasse NUR die gegebenen Informationen aus diesem Text zusammen:

{ocr_text}

Liste auf:
1. Aufgabennummer(n)
2. Was ist gegeben (Formeln, Werte)
3. Was ist gefragt
4. Antwortoptionen (falls Multiple Choice)

WICHTIG: Füge NICHTS hinzu, was nicht im Text steht!"""

    client = Anthropic(api_key=st.secrets["claude_key"])
    summary = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": summary_prompt}]
    ).content[0].text
    
    # Dann: Löse basierend auf der Zusammenfassung
    solve_prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

AUFGABENZUSAMMENFASSUNG:
{summary}

ORIGINALTEXT (zur Referenz):
{ocr_text}

STRIKTE REGELN:
1. Verwende NUR Informationen aus dem gegebenen Text
2. Erfinde KEINE zusätzlichen Hinweise oder Annahmen
3. Wenn Informationen fehlen, sage es explizit
4. Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
5. Zitiere relevante Textstellen wenn du antwortest

DENKE SCHRITT FÜR SCHRITT:
- Was steht WÖRTLICH im Text?
- Was kann ich daraus DIREKT schließen?
- Was ist die korrekte Antwort?

FORMAT:
Aufgabe [Nr]: [Antwort]
Begründung: [Erklärung mit Verweis auf Textstellen]
Verwendete Informationen: [Zitate aus dem Text]"""

    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0,
        system="Du darfst NUR Informationen verwenden, die explizit im Text stehen. Keine externen Annahmen!",
        messages=[{"role": "user", "content": solve_prompt}]
    )
    
    return response.content[0].text, summary

# --- Halluzinations-Check ---
def check_for_hallucinations(solution, ocr_text):
    """Prüft ob die Lösung Informationen enthält, die nicht im OCR-Text stehen"""
    
    check_prompt = f"""Prüfe ob diese Lösung NUR Informationen aus dem OCR-Text verwendet:

OCR-TEXT:
{ocr_text}

LÖSUNG:
{solution}

Finde Aussagen in der Lösung, die NICHT im OCR-Text stehen.
Antworte mit:
- "KEINE HALLUZINATION" wenn alles korrekt
- "HALLUZINATION GEFUNDEN: [beschreibe was nicht im Text steht]" wenn etwas erfunden wurde"""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": check_prompt}]
    )
    
    return response.content[0].text

# --- UI ---
# Debug-Modus
debug_mode = st.checkbox("🔍 Debug-Modus", value=True, help="Zeigt Zwischenschritte")

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
        
        # OCR mit Validierung
        with st.spinner("📖 Lese und validiere Text..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
            
        # Debug: OCR-Ergebnis anzeigen
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis", expanded=True):
                st.code(ocr_text)
        
        # Button zum Lösen
        if st.button("🧮 Aufgaben lösen", type="primary"):
            st.markdown("---")
            
            # Lösung mit Anti-Halluzination
            with st.spinner("🧮 Claude analysiert strikt nach Text..."):
                solution, summary = solve_with_claude_strict(ocr_text)
            
            if debug_mode:
                with st.expander("📋 Aufgabenzusammenfassung"):
                    st.code(summary)
                
                with st.expander("💭 Claudes Lösung"):
                    st.code(solution)
            
            # Halluzinations-Check
            with st.spinner("🔍 Prüfe auf Halluzinationen..."):
                hallucination_check = check_for_hallucinations(solution, ocr_text)
            
            if "KEINE HALLUZINATION" in hallucination_check:
                st.success("✅ Keine Halluzinationen gefunden")
            else:
                st.error(f"⚠️ {hallucination_check}")
            
            # Ergebnisse anzeigen
            st.markdown("### 📊 Lösung:")
            
            # Formatierte Ausgabe
            lines = solution.split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('Aufgabe'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            st.markdown(f"### {parts[0]}: **{parts[1].strip()}**")
                    elif line.startswith('Begründung:'):
                        st.markdown(f"*{line}*")
                    elif line.startswith('Verwendete Informationen:'):
                        with st.expander("📌 Verwendete Textstellen"):
                            st.markdown(line)
                    
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")
        
        # Bei API-Credit-Fehler
        if "credit balance" in str(e).lower():
            st.error("💳 API-Credits aufgebraucht! Bitte Credits aufladen.")

# --- Footer ---
st.markdown("---")
st.caption("Made by Fox | Anti-Hallucination System")
