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
st.markdown("*Ultra-präzise Klausurlösung mit Anti-Halluzination*")

# --- API Clients ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")
claude_client = Anthropic(api_key=st.secrets["claude_key"])
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- OCR mit Caching ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    try:
        logger.info(f"Starting OCR for file hash: {file_hash}")
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written, including EVERY detail from graphs, charts, or sketches. For graphs: Explicitly list ALL axis labels, ALL scales, ALL intersection points with axes (e.g., 'x-axis at 450', 'y-axis at 20'), and EVERY numerical value or annotation. Do NOT interpret, solve, or infer beyond the visible text and numbers. Output a COMPLETE verbatim transcription with NO omissions.",
                _image
            ],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 12000
            }
        )
        ocr_text = response.text.strip()
        logger.info(f"OCR result length: {len(ocr_text)} characters")
        return ocr_text
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- KRITISCH: Extrahiere TATSÄCHLICHE Aufgaben aus OCR ---
def extract_actual_tasks_from_ocr(ocr_text):
    """Extrahiert NUR die tatsächlich im OCR vorhandenen Aufgabennummern"""
    patterns = [
        r'Aufgabe\s+(\d+)',
        r'Task\s+(\d+)',
        r'Frage\s+(\d+)',
        r'Question\s+(\d+)',
        r'Nr\.\s*(\d+)',
        r'(\d+)\.\s*Aufgabe'
    ]
    
    found_tasks = set()
    for pattern in patterns:
        matches = re.findall(pattern, ocr_text, re.IGNORECASE)
        found_tasks.update(matches)
    
    # Sortiere numerisch
    task_numbers = sorted([int(t) for t in found_tasks])
    logger.info(f"Found actual tasks in OCR: {task_numbers}")
    
    return task_numbers

# --- STRIKTE ANTWORTEXTRAKTION ---
def extract_structured_answers_strict(solution_text, valid_task_numbers):
    """Extrahiert NUR Antworten für valide Aufgabennummern"""
    result = {}
    lines = solution_text.split('\n')
    
    task_patterns = [
        r'Aufgabe\s*(\d+)\s*:\s*(.+)',
        r'Task\s*(\d+)\s*:\s*(.+)',
        r'(\d+)[\.\)]\s*(.+)'
    ]
    
    current_task = None
    current_answer = None
    current_reasoning = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        task_found = False
        for pattern in task_patterns:
            task_match = re.match(pattern, line, re.IGNORECASE)
            if task_match:
                potential_task = int(task_match.group(1))
                
                # NUR verarbeiten wenn Aufgabe WIRKLICH existiert
                if potential_task in valid_task_numbers:
                    # Speichere vorherige valide Aufgabe
                    if current_task and current_answer:
                        result[f"Aufgabe {current_task}"] = {
                            'answer': current_answer,
                            'reasoning': ' '.join(current_reasoning).strip()
                        }
                    
                    current_task = potential_task
                    raw_answer = task_match.group(2).strip()
                    current_answer = normalize_answer(raw_answer)
                    current_reasoning = []
                    task_found = True
                    logger.info(f"Valid task found: {current_task}")
                else:
                    logger.warning(f"Ignoring hallucinated task: {potential_task}")
                break
        
        if not task_found and current_task:
            if line.startswith('Begründung:'):
                current_reasoning.append(line.replace('Begründung:', '').strip())
            elif not any(re.match(p, line, re.IGNORECASE) for p in task_patterns):
                current_reasoning.append(line)
    
    # Letzte Aufgabe speichern
    if current_task and current_answer and current_task in valid_task_numbers:
        result[f"Aufgabe {current_task}"] = {
            'answer': current_answer,
            'reasoning': ' '.join(current_reasoning).strip()
        }
    
    return result

def normalize_answer(raw_answer):
    """Normalisiert Antworten für konsistenten Vergleich"""
    answer = raw_answer.strip()
    
    # Multiple-Choice
    if re.match(r'^[A-E](\s*[,;]\s*[A-E])*$', answer, re.IGNORECASE):
        letters = re.findall(r'[A-E]', answer.upper())
        return ''.join(sorted(set(letters)))
    
    # Numerisch (behalte Komma)
    numeric_match = re.search(r'[\d,.\-]+', answer)
    if numeric_match:
        return numeric_match.group(0)
    
    return answer

# --- ULTRA-STRIKTER CLAUDE PROMPT ---
def create_claude_strict_prompt(ocr_text, valid_task_numbers):
    task_list = ", ".join([str(n) for n in valid_task_numbers])
    
    return f"""KRITISCHE ANWEISUNG: Du darfst NUR die Aufgaben {task_list} lösen!

AUFGABENTEXT:
{ocr_text}

ABSOLUTE REGELN:
1. Löse AUSSCHLIESSLICH Aufgabe(n): {task_list}
2. ERFINDE KEINE neuen Aufgabennummern
3. Multiple-Choice: NUR Buchstaben (z.B. "B")
4. Numerisch: Mit Komma (z.B. "22,5")

AUSGABE NUR SO:
Aufgabe {valid_task_numbers[0]}: [Antwort]
Begründung: [Kurz auf Deutsch]

{f'Aufgabe {valid_task_numbers[1]}: [Antwort]' if len(valid_task_numbers) > 1 else ''}
{f'Begründung: [Kurz auf Deutsch]' if len(valid_task_numbers) > 1 else ''}

KEINE ANDEREN AUFGABENNUMMERN!"""

# --- OPTIMIERTER GPT PROMPT (strukturierter) ---
def create_gpt_strict_prompt(ocr_text, valid_task_numbers):
    task_list = ", ".join([str(n) for n in valid_task_numbers])
    
    # GPT braucht mehr Struktur
    return f"""You are solving EXACTLY {len(valid_task_numbers)} task(s): {task_list}

EXAM TEXT:
{ocr_text}

STRICT RULES:
1. ONLY solve task number(s): {task_list}
2. Multiple choice: Give ONLY letter (e.g., "B")
3. Numeric: Use comma (e.g., "22,5")
4. Think step-by-step but be concise

OUTPUT FORMAT:
Aufgabe {valid_task_numbers[0]}: [Answer]
Begründung: [Brief German explanation]

{f'Aufgabe {valid_task_numbers[1]}: [Answer]' if len(valid_task_numbers) > 1 else ''}
{f'Begründung: [Brief German explanation]' if len(valid_task_numbers) > 1 else ''}

NOTHING ELSE!"""

# --- CLAUDE SOLVER ---
def solve_with_claude(ocr_text, valid_task_numbers):
    prompt = create_claude_strict_prompt(ocr_text, valid_task_numbers)
    
    try:
        response = claude_client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=4000,
            temperature=0.05,  # Noch deterministischer
            system=f"KRITISCH: Löse NUR Aufgabe(n) {', '.join(map(str, valid_task_numbers))}. KEINE ANDEREN NUMMERN ERWÄHNEN!",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Claude Error: {str(e)}")
        raise e

# --- GPT SOLVER mit Pre-Processing ---
def solve_with_gpt(ocr_text, valid_task_numbers):
    prompt = create_gpt_strict_prompt(ocr_text, valid_task_numbers)
    
    try:
        # GPT bekommt zusätzliche System-Instruktionen
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": f"You are an expert in 'Internes Rechnungswesen'. Solve ONLY task(s) {', '.join(map(str, valid_task_numbers))}. Use German Fernuni Hagen methods. Be precise and methodical."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.05,  # Deterministischer
            top_p=0.1  # Fokussierter
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"GPT Error: {str(e)}")
        return None

# --- INTELLIGENTER ANTWORTVERGLEICH ---
def answers_are_equivalent(answer1, answer2):
    """Prüft ob zwei Antworten äquivalent sind"""
    if answer1 == answer2:
        return True
    
    # Multiple-Choice
    if re.match(r'^[A-E]+$', answer1) and re.match(r'^[A-E]+$', answer2):
        return set(answer1) == set(answer2)
    
    # Numerisch mit verbesserter Toleranz
    try:
        num1 = float(answer1.replace(',', '.'))
        num2 = float(answer2.replace(',', '.'))
        
        # Für "ca." Angaben
        relative_tolerance = 0.02  # 2%
        absolute_tolerance = 0.01
        
        return abs(num1 - num2) <= max(absolute_tolerance, relative_tolerance * max(abs(num1), abs(num2)))
    except:
        pass
    
    return answer1.lower() == answer2.lower()

# --- FINALE KREUZVALIDIERUNG mit strikter Filterung ---
def strict_cross_validation(ocr_text, valid_task_numbers):
    """Kreuzvalidierung mit absoluter Anti-Halluzination"""
    st.markdown("### 🔄 Strikte Kreuzvalidierung")
    st.info(f"Löse NUR: Aufgabe {', '.join(map(str, valid_task_numbers))}")
    
    # Claude
    with st.spinner("Claude Opus 4 analysiert..."):
        claude_solution = solve_with_claude(ocr_text, valid_task_numbers)
    
    # STRIKTE Extraktion
    claude_data = extract_structured_answers_strict(claude_solution, valid_task_numbers)
    
    # GPT
    with st.spinner("GPT-4 Turbo validiert..."):
        gpt_solution = solve_with_gpt(ocr_text, valid_task_numbers)
        gpt_data = extract_structured_answers_strict(gpt_solution, valid_task_numbers) if gpt_solution else {}
    
    # Zeige NUR valide Aufgaben
    final_answers = {}
    
    for task_num in valid_task_numbers:
        task_key = f"Aufgabe {task_num}"
        claude_ans = claude_data.get(task_key, {}).get('answer', '')
        gpt_ans = gpt_data.get(task_key, {}).get('answer', '')
        
        col1, col2, col3, col4 = st.columns([2, 3, 3, 1])
        with col1:
            st.write(f"**{task_key}:**")
        with col2:
            st.write(f"Claude: `{claude_ans}`" if claude_ans else "Claude: -")
        with col3:
            st.write(f"GPT: `{gpt_ans}`" if gpt_ans else "GPT: -")
        
        # Entscheidung
        if claude_ans:
            final_answers[task_key] = claude_data[task_key]
            if gpt_ans and answers_are_equivalent(claude_ans, gpt_ans):
                with col4:
                    st.write("✅")
            else:
                with col4:
                    st.write("⚠️")
        elif gpt_ans:
            final_answers[task_key] = gpt_data[task_key]
            with col4:
                st.write("🔄")
        else:
            with col4:
                st.write("❌")
    
    # Erfolgsrate
    solved = len(final_answers)
    total = len(valid_task_numbers)
    if solved == total:
        st.success(f"✅ Alle {total} Aufgaben gelöst!")
    else:
        st.warning(f"⚠️ {solved}/{total} Aufgaben gelöst")
    
    return final_answers, claude_solution, gpt_solution

# --- HAUPTINTERFACE ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Cache leeren"):
        st.cache_data.clear()
        st.rerun()

uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        image = Image.open(uploaded_file)
        
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        # OCR
        with st.spinner("📖 Lese Text mit Gemini Flash 1.5..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
        
        # KRITISCH: Finde tatsächliche Aufgaben
        valid_task_numbers = extract_actual_tasks_from_ocr(ocr_text)
        
        if not valid_task_numbers:
            st.error("❌ Keine Aufgaben im Bild gefunden!")
            st.stop()
        
        st.success(f"✅ Gefunden: Aufgabe {', '.join(map(str, valid_task_numbers))}")
        
        # Debug
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis"):
                st.code(ocr_text)
        
        # Lösen
        if st.button("🧮 Aufgaben lösen", type="primary"):
            st.markdown("---")
            
            # Strikte Validierung
            final_answers, claude_full, gpt_full = strict_cross_validation(ocr_text, valid_task_numbers)
            
            # Finale Ausgabe
            st.markdown("---")
            st.markdown("### 🎯 FINALE LÖSUNG")
            
            if final_answers:
                # Zeige NUR die tatsächlichen Aufgaben in richtiger Reihenfolge
                for task_num in valid_task_numbers:
                    task_key = f"Aufgabe {task_num}"
                    if task_key in final_answers:
                        data = final_answers[task_key]
                        st.markdown(f"### {task_key}: **{data['answer']}**")
                        if data.get('reasoning'):
                            st.markdown(f"*{data['reasoning']}*")
            else:
                st.error("❌ Keine Lösungen generiert")
            
            # Debug
            if debug_mode:
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("Claude Rohausgabe"):
                        st.code(claude_full)
                with col2:
                    with st.expander("GPT Rohausgabe"):
                        st.code(gpt_full if gpt_full else "Keine Lösung")
                        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# Footer
st.markdown("---")
st.caption("🦊 Koifox-Bot | Strikte Anti-Halluzination | Nur echte Aufgaben")
