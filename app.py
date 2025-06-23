import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
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
        'claude_key': ('sk-ant', "Claude"),
        'openai_key': ('sk-', "OpenAI")
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
st.markdown("*Made with coffee, deep minimal and tiny gummy bears*")

# --- Cache Management ---
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Cache leeren", type="secondary", help="Löscht gespeicherte OCR-Ergebnisse"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- API Clients ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")
claude_client = Anthropic(api_key=st.secrets["claude_key"])
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- GPT Model Detection ---
@st.cache_data
def get_available_gpt_model():
    """Testet welches GPT Modell verfügbar ist"""
    test_models = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    
    for model in test_models:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            return model
        except:
            continue
    return "gpt-3.5-turbo"

GPT_MODEL = get_available_gpt_model()

# --- OCR mit Caching ---
@st.cache_data(ttl=3600)  # Cache für 1 Stunde
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
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- EINFACHE ANTWORTEXTRAKTION ---
def extract_simple_answers(solution_text):
    """Extrahiert nur die finalen Antworten - einfach und robust"""
    answers = {}
    lines = solution_text.split('\n')
    
    for line in lines:
        # Suche nach "Aufgabe X: ANTWORT"
        match = re.match(r'Aufgabe\s*(\d+)\s*:\s*(.+)', line.strip(), re.IGNORECASE)
        if match:
            task_num = match.group(1)
            answer = match.group(2).strip()
            # Normalisiere Buchstaben-Antworten
            if re.match(r'^[A-E,\s]+$', answer):
                answer = ''.join(sorted(c for c in answer.upper() if c in 'ABCDE'))
            answers[f"Aufgabe {task_num}"] = answer
    
    return answers

# --- Claude Solver ---
def solve_with_claude(ocr_text, feedback=None):
    """Dein ursprünglicher Claude-Prompt - unverändert!"""
    
    prompt = f"""You are a highly qualified accounting expert with PhD-level 
knowledge of the university course "Internes Rechnungswesen (31031)" at Fernuniversität Hagen. 
Your task is to answer exam questions with 100% accuracy.

THEORETICAL SCOPE
Use only the decision-oriented German managerial-accounting (Controlling) framework:
- Cost-type, cost-center and cost-unit accounting (Kostenarten-, Kostenstellen-, Kostenträgerrechnung)
- Full, variable, marginal, standard (Plankosten-) and process/ABC costing systems
- Flexible and Grenzplankostenrechnung variance analysis
- Single- and multi-level contribution-margin accounting and break-even logic
- Causality & allocation (Verursachungs- und Zurechnungsprinzip)
- Business-economics MRS convention (MRS = MP₂ / MP₁ unless stated otherwise)
- Activity-analysis production & logistics models (LP, Standort- & Transportprobleme)
- Marketing segmentation, price-elasticity, contribution-based pricing & mix planning

{f"WICHTIGER HINWEIS: {feedback}" if feedback else ""}

WICHTIG: Analysiere NUR den folgenden OCR-Text. Erfinde KEINE anderen Aufgaben! 
Sei extrem präzise und verwende die Lösungswege und die Terminologie der Fernuni Hagen. Es gibt absolut keinen Raum für Fehler!

OCR-TEXT START:
{ocr_text}
OCR-TEXT ENDE

KRITISCHE ANWEISUNGEN:
1. Lies die Aufgabe SEHR sorgfältig
2. Bei Rechenaufgaben:
   - Zeige JEDEN Rechenschritt
   - Prüfe dein Ergebnis nochmal
3. Bei Multiple Choice: Prüfe jede Option einzeln
4. VERIFIZIERE deine Antwort bevor du antwortest
5. Stelle SICHER, dass deine Antwort mit deiner Analyse übereinstimmt!

FORMAT - WICHTIG:
Aufgabe [Nr]: [NUR die finale Antwort - Zahl oder Buchstabe(n)]
Begründung: [1 Satz auf Deutsch]
"""

    response = claude_client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=4000,
        temperature=0,
        top_p=1.0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- GPT Solver ---
def solve_with_gpt(ocr_text, feedback=None):
    """GPT mit demselben Prompt wie Claude"""
    
    prompt = f"""You are a highly qualified accounting expert with PhD-level 
knowledge of the university course "Internes Rechnungswesen (31031)" at Fernuniversität Hagen. 
Your task is to answer exam questions with 100% accuracy.

THEORETICAL SCOPE
Use only the decision-oriented German managerial-accounting (Controlling) framework:
- Cost-type, cost-center and cost-unit accounting (Kostenarten-, Kostenstellen-, Kostenträgerrechnung)
- Full, variable, marginal, standard (Plankosten-) and process/ABC costing systems
- Flexible and Grenzplankostenrechnung variance analysis
- Single- and multi-level contribution-margin accounting and break-even logic
- Causality & allocation (Verursachungs- und Zurechnungsprinzip)
- Business-economics MRS convention (MRS = MP₂ / MP₁ unless stated otherwise)
- Activity-analysis production & logistics models (LP, Standort- & Transportprobleme)
- Marketing segmentation, price-elasticity, contribution-based pricing & mix planning

{f"WICHTIGER HINWEIS: {feedback}" if feedback else ""}

WICHTIG: Analysiere NUR den folgenden OCR-Text. Erfinde KEINE anderen Aufgaben! 
Sei extrem präzise und verwende die Lösungswege und die Terminologie der Fernuni Hagen. Es gibt absolut keinen Raum für Fehler!

OCR-TEXT START:
{ocr_text}
OCR-TEXT ENDE

KRITISCHE ANWEISUNGEN:
1. Lies die Aufgabe SEHR sorgfältig
2. Bei Rechenaufgaben:
   - Zeige JEDEN Rechenschritt
   - Prüfe dein Ergebnis nochmal
3. Bei Multiple Choice: Prüfe jede Option einzeln
4. VERIFIZIERE deine Antwort bevor du antwortest
5. Stelle SICHER, dass deine Antwort mit deiner Analyse übereinstimmt!

FORMAT - WICHTIG:
Aufgabe [Nr]: [NUR die finale Antwort - Zahl oder Buchstabe(n)]
Begründung: [1 Satz auf Deutsch]
"""

    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0
    )
    
    return response.choices[0].message.content

# --- EINFACHE KONFLIKTLÖSUNG ---
def simple_consensus_check(claude_sol, gpt_sol, ocr_text, max_iterations=2):
    """Einfache Konsens-Prüfung ohne Überkomplizierung"""
    
    # Extrahiere Antworten
    claude_answers = extract_simple_answers(claude_sol)
    gpt_answers = extract_simple_answers(gpt_sol)
    
    # Zeige Vergleich
    st.markdown("### 🔍 Antwortvergleich:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Claude:**")
        for task, answer in claude_answers.items():
            st.write(f"{task}: {answer}")
    
    with col2:
        st.markdown(f"**{GPT_MODEL}:**")
        for task, answer in gpt_answers.items():
            st.write(f"{task}: {answer}")
    
    # Prüfe Unterschiede
    all_tasks = set(claude_answers.keys()) | set(gpt_answers.keys())
    differences = []
    
    for task in all_tasks:
        claude_ans = claude_answers.get(task, "")
        gpt_ans = gpt_answers.get(task, "")
        if claude_ans != gpt_ans:
            differences.append(f"{task}: Claude={claude_ans} vs GPT={gpt_ans}")
    
    if not differences:
        st.success("✅ Konsens erreicht!")
        return True, claude_sol
    
    st.warning(f"⚠️ Unterschiede: {differences}")
    
    # Einfache Iterationen
    current_claude = claude_sol
    current_gpt = gpt_sol
    
    for iteration in range(max_iterations):
        st.markdown(f"### 🔄 Iteration {iteration + 2}:")
        
        feedback = f"Das andere Modell hat folgende abweichende Antworten: {'; '.join(differences)}. Prüfe deine Lösung nochmal sehr genau nach Fernuni Hagen Standards."
        
        with st.spinner(f"Iteration {iteration + 2}..."):
            current_claude = solve_with_claude(ocr_text, feedback)
            current_gpt = solve_with_gpt(ocr_text, feedback)
        
        # Neue Antworten prüfen
        claude_answers = extract_simple_answers(current_claude)
        gpt_answers = extract_simple_answers(current_gpt)
        
        differences = []
        for task in all_tasks:
            claude_ans = claude_answers.get(task, "")
            gpt_ans = gpt_answers.get(task, "")
            if claude_ans != gpt_ans:
                differences.append(f"{task}: Claude={claude_ans} vs GPT={gpt_ans}")
        
        if not differences:
            st.success("✅ Konsens erreicht!")
            return True, current_claude
    
    return False, (current_claude, current_gpt)

# --- UI Optionen ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False, help="Zeigt OCR-Ergebnis und Details")

# --- Datei-Upload ---
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
        with st.spinner("Lese Text mit Gemini Flash..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
            
        # Debug: OCR-Ergebnis anzeigen
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis", expanded=False):
                st.code(ocr_text)
                st.info(f"File Hash: {file_hash[:8]}... (für Caching)")
        
        # Button zum Lösen
        if st.button("🧮 Aufgaben lösen", type="primary"):
            
            # Beide Modelle parallel
            with st.spinner("Löse mit beiden Modellen..."):
                try:
                    claude_solution = solve_with_claude(ocr_text)
                    gpt_solution = solve_with_gpt(ocr_text)
                    
                    if debug_mode:
                        with st.expander("🔍 Rohe Antworten"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Claude:**")
                                st.code(claude_solution)
                            with col2:
                                st.markdown(f"**{GPT_MODEL}:**")
                                st.code(gpt_solution)
                    
                    # Konsens-Check
                    consensus, result = simple_consensus_check(claude_solution, gpt_solution, ocr_text)
                    
                    # Ergebnisse anzeigen
                    st.markdown("---")
                    st.markdown("### 🎯 Lösung:")
                    
                    if consensus:
                        # Formatierte Ausgabe der finalen Lösung
                        lines = result.split('\n')
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
                                else:
                                    st.markdown(line)
                    else:
                        st.error("❌ Modelle sind uneinig - manuelle Prüfung erforderlich!")
                        
                        claude_final, gpt_final = result
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Claude Final:**")
                            st.code(claude_final)
                        
                        with col2:
                            st.markdown(f"**{GPT_MODEL} Final:**")
                            st.code(gpt_final)
                    
                    # Info über Caching
                    st.info("💡 OCR-Ergebnisse werden gecached, Lösungen werden immer neu berechnet.")
                    
                except Exception as e:
                    logger.error(f"API Error: {str(e)}")
                    st.error(f"API Fehler: {str(e)}")
                    
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption(f"Made by Fox | Claude-4 Opus + {GPT_MODEL} | OCR cached, Solutions always fresh")
