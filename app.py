import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import re
import io

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- UI-Einstellungen ---
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")
st.markdown("*Fernuni Hagen IRW-konformes Multi-Model System*")

# --- API Key Validation ---
def validate_keys():
    required_keys = {
        'gemini_key': ('AIza', "Gemini"),
        'claude_key': ('sk-ant', "Claude"),
        'openai_key': ('sk-', "OpenAI")
    }
    missing = []
    
    for key, (prefix, name) in required_keys.items():
        if key not in st.secrets:
            missing.append(name)
        elif not st.secrets[key].startswith(prefix):
            missing.append(f"{name} (invalid)")
    
    if missing:
        st.error(f"Fehlende API Keys: {', '.join(missing)}")
        st.stop()

validate_keys()

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
st.sidebar.info(f"🤖 Claude + {GPT_MODEL}")

# --- VERBESSERTE OCR ---
@st.cache_data(ttl=3600, show_spinner=False)
def extract_text_with_gemini(_image_bytes, file_hash):
    """Robuste OCR mit Gemini"""
    try:
        logger.info(f"Starting OCR for hash: {file_hash[:8]}...")
        
        image = Image.open(io.BytesIO(_image_bytes))
        
        # Optimale Größe für OCR
        if max(image.width, image.height) > 3000:
            image.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
        
        # Klarer OCR Prompt
        ocr_prompt = """Extrahiere den EXAKTEN Text aus diesem Klausurbild:

1. Lies von oben nach unten ALLES
2. Inkludiere:
   - Aufgabennummern (z.B. "Aufgabe 1", "Aufgabe 2")
   - Komplette Fragestellungen
   - ALLE Antwortoptionen (A, B, C, D, E) mit vollem Text
   - "(x aus 5)" Angaben
   - Alle Zahlen und Formeln

WICHTIG: 
- Füge KEINE eigenen Kommentare hinzu
- Keine Interpretation
- Nur der reine Text aus dem Bild"""
        
        response = vision_model.generate_content(
            [ocr_prompt, image],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 12000
            }
        )
        
        ocr_text = response.text.strip()
        
        # Entferne falsche Statusmeldungen
        ocr_text = re.sub(r'(Graph|Table|Formel|Data):\s*(Kein|No|Not found|Keine).*?(vorhanden|found)\.?\s*', '', ocr_text, flags=re.IGNORECASE)
        ocr_text = re.sub(r'❌\s*Keine Aufgaben.*?gefunden!?\s*', '', ocr_text)
        ocr_text = re.sub(r'✅\s*Graphen oder.*?gefunden!?\s*', '', ocr_text)
        
        logger.info(f"OCR completed: {len(ocr_text)} characters")
        return ocr_text
        
    except Exception as e:
        logger.error(f"OCR Error: {str(e)}")
        raise e

# --- FERNUNI-SPEZIFISCHE REGELN ---
FERNUNI_RULES = """
FERNUNI HAGEN SPEZIFISCHE REGELN (STRIKT BEFOLGEN!):

1. FUNKTIONENMODELL (Kurs 31031):
   - Originäre Funktionen sind NUR: Beschaffung, Produktion, Absatz
   - Alle anderen sind derivative/unterstützende Funktionen
   
2. HARRIS-FORMEL / EOQ:
   - Nach Fernuni-Definition: Harris-Formel unterstellt KEINE (T,Q)-Politik
   - Formel: Q* = √(2×D×K_B/k_L)
   - Lagerkosten IMMER auf Jahresbasis umrechnen
   
3. HOMOGENITÄT:
   - f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
   
4. MULTIPLE CHOICE:
   - Bei "(x aus 5)" können 0 bis 5 Antworten richtig sein
   - Antworte mit BUCHSTABEN (A,B,C,D,E), nicht mit Zahlen!
"""

# --- VEREINFACHTE ANTWORTEXTRAKTION ---
def extract_final_answers(solution_text):
    """Extrahiert nur die finalen Antworten ohne komplexe Verarbeitung"""
    answers = {}
    lines = solution_text.split('\n')
    
    for line in lines:
        # Suche nach "Aufgabe X: BUCHSTABEN"
        match = re.match(r'Aufgabe\s*(\d+)\s*:\s*([A-E]+)', line.strip(), re.IGNORECASE)
        if match:
            task_num = match.group(1)
            letters = ''.join(sorted(c for c in match.group(2).upper() if c in 'ABCDE'))
            answers[f"Aufgabe {task_num}"] = letters
    
    return answers

# --- Claude Solver mit Fernuni-Fokus ---
def solve_with_claude(ocr_text, previous_feedback=None):
    """Claude löst strikt nach Fernuni-Standards"""
    
    prompt = f"""{FERNUNI_RULES}

DEINE AUFGABE:
Analysiere die folgenden Klausuraufgaben und beantworte sie EXAKT nach Fernuni Hagen Standards.

{f"WICHTIGER HINWEIS VON ANDEREM MODELL: {previous_feedback}" if previous_feedback else ""}

KLAUSURTEXT:
{ocr_text}

ANWEISUNGEN:
1. Identifiziere alle Aufgaben im Text
2. Bei Multiple Choice: Gib die BUCHSTABEN der richtigen Antworten an
3. Nutze NUR Fernuni-Definitionen, keine anderen Quellen!
4. Format:

Aufgabe [Nr]: [Buchstaben der richtigen Antworten, z.B. "CDE" oder "A"]
Begründung: [Kurze Erklärung mit Fernuni-Bezug]

WICHTIG: Keine Zahlen als Antwort, nur Buchstaben!"""

    response = claude_client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=3000,
        temperature=0.1,
        system="Du bist ein Fernuni Hagen Experte für Modul 31031. Verwende AUSSCHLIESSLICH Fernuni-Definitionen.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- GPT Solver mit Fernuni-Fokus ---
def solve_with_gpt(ocr_text, previous_feedback=None):
    """GPT löst strikt nach Fernuni-Standards"""
    
    prompt = f"""{FERNUNI_RULES}

DEINE AUFGABE:
Analysiere die folgenden Klausuraufgaben und beantworte sie EXAKT nach Fernuni Hagen Standards.

{f"WICHTIGER HINWEIS VON ANDEREM MODELL: {previous_feedback}" if previous_feedback else ""}

KLAUSURTEXT:
{ocr_text}

ANWEISUNGEN:
1. Identifiziere alle Aufgaben im Text
2. Bei Multiple Choice: Gib die BUCHSTABEN der richtigen Antworten an
3. Nutze NUR Fernuni-Definitionen, keine anderen Quellen!
4. Format:

Aufgabe [Nr]: [Buchstaben der richtigen Antworten, z.B. "CDE" oder "A"]
Begründung: [Kurze Erklärung mit Fernuni-Bezug]

WICHTIG: Keine Zahlen als Antwort, nur Buchstaben!"""

    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "Du bist ein Fernuni Hagen Experte für Modul 31031. Verwende AUSSCHLIESSLICH Fernuni-Definitionen."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.1
    )
    
    return response.choices[0].message.content

# --- VEREINFACHTE KONFLIKTLÖSUNG ---
def resolve_conflicts_simple(ocr_text, claude_sol, gpt_sol, max_iterations=2):
    """Einfachere Konfliktlösung mit besserer Transparenz"""
    
    # Zeige beide initiale Lösungen
    st.markdown("### 🔄 Initiale Lösungen:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Claude:**")
        st.code(claude_sol, language=None)
    with col2:
        st.markdown(f"**{GPT_MODEL}:**")
        st.code(gpt_sol, language=None)
    
    # Extrahiere Antworten
    claude_answers = extract_final_answers(claude_sol)
    gpt_answers = extract_final_answers(gpt_sol)
    
    st.markdown("### 🔍 Antwortvergleich:")
    st.write(f"**Claude:** {claude_answers}")
    st.write(f"**GPT:** {gpt_answers}")
    
    # Prüfe auf Unterschiede
    all_tasks = set(claude_answers.keys()) | set(gpt_answers.keys())
    discrepancies = []
    
    for task in all_tasks:
        claude_ans = claude_answers.get(task, "")
        gpt_ans = gpt_answers.get(task, "")
        if claude_ans != gpt_ans:
            discrepancies.append(f"{task}: Claude='{claude_ans}' vs GPT='{gpt_ans}'")
    
    if not discrepancies:
        st.success("✅ Konsens erreicht!")
        return True, claude_sol
    
    st.warning(f"⚠️ Unterschiede gefunden: {discrepancies}")
    
    # Iterationen
    current_claude = claude_sol
    current_gpt = gpt_sol
    
    for iteration in range(max_iterations):
        st.markdown(f"### 🔄 Iteration {iteration + 2}:")
        
        feedback = f"""
ACHTUNG: Diskrepanz mit anderem Modell:
{chr(10).join(discrepancies)}

WICHTIG: 
- Prüfe nochmals GENAU nach Fernuni Hagen Definitionen!
- Bei Harris-Formel: Fernuni lehrt, dass sie KEINE (T,Q)-Politik unterstellt
- Originäre Funktionen sind NUR: Beschaffung, Produktion, Absatz
- Gib Antworten als BUCHSTABEN (A,B,C,D,E), nicht als Zahlen!
"""
        
        # Neue Versuche
        with st.spinner(f"Iteration {iteration + 2} läuft..."):
            current_claude = solve_with_claude(ocr_text, feedback)
            current_gpt = solve_with_gpt(ocr_text, feedback)
        
        # Zeige neue Antworten
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Claude (neu):**")
            st.code(current_claude, language=None)
        with col2:
            st.markdown(f"**{GPT_MODEL} (neu):**")
            st.code(current_gpt, language=None)
        
        # Prüfe erneut
        claude_answers = extract_final_answers(current_claude)
        gpt_answers = extract_final_answers(current_gpt)
        
        st.write(f"**Claude:** {claude_answers}")  
        st.write(f"**GPT:** {gpt_answers}")
        
        discrepancies = []
        for task in all_tasks:
            claude_ans = claude_answers.get(task, "")
            gpt_ans = gpt_answers.get(task, "")
            if claude_ans != gpt_ans:
                discrepancies.append(f"{task}: Claude='{claude_ans}' vs GPT='{gpt_ans}'")
        
        if not discrepancies:
            st.success("✅ Konsens erreicht!")
            return True, current_claude
        
        st.warning(f"⚠️ Immer noch Unterschiede: {discrepancies}")
    
    st.error("❌ Kein Konsens nach allen Iterationen!")
    return False, (current_claude, current_gpt)

# --- MAIN UI ---
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption="Klausuraufgabe", use_container_width=True)
    
    # OCR
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    with st.spinner("Lese Text..."):
        try:
            ocr_text = extract_text_with_gemini(file_bytes, file_hash)
            
            # Validierung
            if len(ocr_text) < 100:
                st.error("❌ OCR zu kurz - bitte besseres Bild verwenden")
                st.stop()
            
            # Aufgaben finden
            found_tasks = re.findall(r'Aufgabe\s+(\d+)', ocr_text, re.IGNORECASE)
            if found_tasks:
                st.success(f"✅ Gefunden: Aufgabe {', '.join(set(found_tasks))}")
            
        except Exception as e:
            st.error(f"❌ OCR Fehler: {str(e)}")
            st.stop()
    
    # OCR anzeigen
    with st.expander("OCR-Text"):
        st.text(ocr_text)
    
    # Lösen
    if st.button("🧮 Nach Fernuni-Standards lösen", type="primary"):
        st.markdown("---")
        
        try:
            # Erste Lösungen
            with st.spinner("Generiere Lösungen..."):
                claude_solution = solve_with_claude(ocr_text)
                gpt_solution = solve_with_gpt(ocr_text)
            
            # Konfliktlösung mit voller Transparenz
            consensus, result = resolve_conflicts_simple(ocr_text, claude_solution, gpt_solution)
            
            st.markdown("---")
            st.markdown("### 🎯 FINALE LÖSUNG:")
            
            if consensus:
                # Zeige finale Antwort strukturiert
                lines = result.split('\n')
                current_task = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Aufgabe'):
                        if ':' in line:
                            task, answer = line.split(':', 1)
                            st.markdown(f"### {task}: **{answer.strip()}**")
                            current_task = task
                    elif line.startswith('Begründung:'):
                        st.markdown(f"*{line}*")
                        st.markdown("")  # Leerzeile
            else:
                st.error("❌ Finale Uneinigkeit - manuelle Prüfung erforderlich!")
                
        except Exception as e:
            st.error(f"Fehler: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.caption("Fernuni Hagen konformes System | Strikt nach Modul 31031 Standards")
