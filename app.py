import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter
import google.generativeai as genai
import logging
import hashlib
import re
from sentence_transformers import SentenceTransformer, util

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
st.set_page_config(layout="centered", page_title="3.0", page_icon="🦊")
st.title("🦊 Koifox-Bot")
st.markdown("*Made with coffee, deep minimal and tiny gummy bears*")

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- SentenceTransformer für Konsistenzprüfung ---
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_transformer()

# --- Flexibles OCR für alle Aufgabentypen ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini_improved(_image, file_hash):
    """Extrahiert KOMPLETTEN Text und alle relevanten Daten ohne feste Annahmen"""
    try:
        logger.info(f"Starting GEMINI OCR for file hash: {file_hash}")
        
        # Bildvorverarbeitung: Kontrast und Schärfung erhöhen
        enhancer = ImageEnhance.Contrast(_image)
        enhanced_image = enhancer.enhance(2.5)
        sharpened_image = enhanced_image.filter(ImageFilter.SHARPEN)
        if sharpened_image.width > 3000 or sharpened_image.height > 3000:
            sharpened_image.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
            st.sidebar.warning(f"Bild wurde auf 3000px skaliert")
        
        response = vision_model.generate_content(
            [
                """Extract ALL content from this exam image in a structured format, including text, formulas, tables, and visual elements like graphs or diagrams.

                IMPORTANT:
                - Read ALL visible text from top to bottom, including:
                  - Question numbers (e.g., "Aufgabe 45 (5 RP)")
                  - All questions, formulas, values, and answer options (A, B, C, D, E) with their complete text
                  - Mathematical symbols and formulas exactly as shown (e.g., f(r₁,r₂) = (r₁^α + r₂^β)^γ)
                  - ALL small text, annotations, footnotes, and any numerical data
                - For tables, graphs, or diagrams:
                  - Describe ALL visual elements in detail, including:
                    - Axes labels (e.g., x-axis: "Quantity", y-axis: "Cost")
                    - ALL data points or values shown (e.g., "Point at (10, 500)")
                    - Curve shapes or trends (e.g., "Linear increasing curve")
                    - ALL annotations or labels
                    - If a table is present, extract it as a structured table (e.g., rows and columns)
                - Structure the output as follows:
                  - Aufgabe [Nummer]: [Fragetext]
                    - Option A: [Text]
                    - Option B: [Text]
                    - ...
                    - Formeln: [z. B. f(r₁,r₂) = (r₁^α + r₂^β)^γ]
                    - Graph/Table: [Detailed description, including ALL data points and annotations]
                  - Data: [List ALL numerical data or data points found, e.g., 'Point at (0, 450)', 'Cost: 20', 'Revenue: 500']
                - DO NOT summarize or skip any part
                - DO NOT solve anything
                
                Start from the very top and continue to the very bottom of the image.""",
                sharpened_image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 10000
            }
        )
        
        ocr_result = response.text.strip()
        
        if len(ocr_result) < 400:
            st.warning(f"⚠️ Nur {len(ocr_result)} Zeichen extrahiert - möglicherweise unvollständig!")
        
        logger.info(f"GEMINI OCR completed: {len(ocr_result)} characters")
        logger.info(f"Erkannte Daten: {[item for item in re.findall(r'Point at \(\d+,\s*\d+\)', ocr_result)]}")
        
        return ocr_result
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- Numerischer Vergleich der Endantworten ---
def compare_numerical_answers(answers1, answers2):
    """Vergleicht Endantworten numerisch"""
    differences = []
    for a1, a2 in zip(answers1, answers2):
        try:
            num1 = float(a1.replace(',', '.'))
            num2 = float(a2.replace(',', '.'))
            if abs(num1 - num2) > 0.1:
                differences.append((a1, a2))
        except ValueError:
            continue
    return differences

# --- Konsistenzprüfung zwischen LLMs ---
def are_answers_similar(answer1, answer2):
    """Vergleicht die Endantworten auf semantische Ähnlichkeit und numerisch"""
    try:
        task_pattern = r'Aufgabe\s+\d+\s*:\s*([^\n]+)'
        answers1 = re.findall(task_pattern, answer1, re.IGNORECASE)
        answers2 = re.findall(task_pattern, answer2, re.IGNORECASE)
        
        if not answers1 or not answers2:
            logger.warning("Keine Endantworten für Konsistenzprüfung gefunden")
            return False, [], [], []
        
        embeddings = sentence_model.encode([' '.join(answers1), ' '.join(answers2)])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        logger.info(f"Antwortähnlichkeit (Endantworten): {similarity:.2f}")
        
        numerical_differences = compare_numerical_answers(answers1, answers2)
        
        return similarity > 0.8 and not numerical_differences, answers1, answers2, numerical_differences
    except Exception as e:
        logger.error(f"Konsistenzprüfung fehlgeschlagen: {str(e)}")
        return False, [], [], []

# --- Claude Solver für alle Aufgabentypen ---
def solve_with_claude_formatted(ocr_text):
    """Claude löst flexibel basierend auf allen verfügbaren Daten mit Korrekturlogik"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

VOLLSTÄNDIGER AUFGABENTEXT:
{ocr_text}

WICHTIGE REGELN:
1. Identifiziere ALLE Aufgaben im Text (z.B. "Aufgabe 45", "Aufgabe 46" etc.)
2. Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
3. Beantworte JEDE Aufgabe die du findest
4. Denke schrittweise:
   - Lies die Aufgabe sorgfältig
   - Identifiziere alle relevanten Formeln, Werte und Daten (z.B. 'Point at (0, 450)', 'Cost: 20', Tabellen)
   - Leite Funktionen oder Berechnungen aus den verfügbaren Daten ab (z.B. Preis-Satzfunktion aus Graphendaten, falls vorhanden)
   - Wenn Daten unvollständig sind, dokumentiere Annahmen klar und markiere sie als unsicher
   - Führe die Berechnung explizit durch
   - Überprüfe dein Ergebnis und korrigiere es, wenn nötig
   - Die LETZTE berechnete Antwort (nach Korrektur) MUSS als Endantwort verwendet werden
5. Bei Multiple-Choice-Fragen: Analysiere jede Option und begründe, warum sie richtig oder falsch ist
6. Wenn Tabellen, Graphen oder andere visuelle Elemente beschrieben sind, nutze diese Informationen für die Lösung
7. Für Aufgaben wie Gewinnmaximierung: Nutze die im Text verfügbaren Daten (z.B. Graphenpunkte) und setze Standardwerte wie kv = 3 und kf = 20, wenn nicht anders angegeben, aber dokumentiere dies als Annahme
8. Die Endantwort MUSS exakt der LETZTEN berechneten Zahl entsprechen (z.B. 11.50 nach Korrektur) und auf zwei Dezimalstellen formatiert sein

AUSGABEFORMAT (STRIKT EINHALTEN):
Aufgabe [Nummer]: [Antwort auf zwei Dezimalstellen, basierend auf der letzten Berechnung]
Begründung: [Schritt-für-Schritt-Erklärung inklusive Korrekturen]
Berechnung: [Mathematische Schritte, markiere die letzte berechnete Zahl klar]
Annahmen (falls nötig): [z.B. "Preis-Satzfunktion wurde aus Graphendaten abgeleitet" oder "kv = 3, kf = 20 als Standardwerte angenommen"]

Wiederhole dies für JEDE Aufgabe im Text.

Beispiel:
Aufgabe 48: 11.50
Begründung: Der gewinnmaximale Preis wird durch Ableiten der Gewinnfunktion bestimmt... Initiale Annahme war falsch, nach Korrektur...
Berechnung: x = 450 - 22.5·p (aus Graphen), G(p) = (p - 3)·(450 - 22.5·p) - 20, dG/dp = 0, p = 517.5/45 = 11.50 (letzte berechnete Zahl)
Annahmen: Preis-Satzfunktion aus Graphendaten (0, 450) und (20, 0), kv = 3, kf = 20 als Standardwerte

WICHTIG: Vergiss keine Aufgabe!"""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=4000,
        temperature=0.1,
        system="Beantworte ALLE Aufgaben die im Text stehen. Überspringe keine. Nutze die letzte berechnete Antwort als Endantwort.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- GPT-4 Turbo Solver für alle Aufgabentypen ---
def solve_with_gpt(ocr_text):
    """GPT-4 Turbo löst flexibel basierend auf allen verfügbaren Daten mit Korrekturlogik"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

VOLLSTÄNDIGER AUFGABENTEXT:
{ocr_text}

WICHTIGE REGELN:
1. Identifiziere ALLE Aufgaben im Text (z.B. "Aufgabe 45", "Aufgabe 46" etc.)
2. Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
3. Beantworte JEDE Aufgabe die du findest
4. Denke schrittweise:
   - Lies die Aufgabe sorgfältig
   - Identifiziere alle relevanten Formeln, Werte und Daten (z.B. 'Point at (0, 450)', 'Cost: 20', Tabellen)
   - Leite Funktionen oder Berechnungen aus den verfügbaren Daten ab (z.B. Preis-Satzfunktion aus Graphendaten, falls vorhanden)
   - Wenn Daten unvollständig sind, dokumentiere Annahmen klar und markiere sie als unsicher
   - Führe die Berechnung explizit durch
   - Überprüfe dein Ergebnis und korrigiere es, wenn nötig
   - Die LETZTE berechnete Antwort (nach Korrektur) MUSS als Endantwort verwendet werden
5. Bei Multiple-Choice-Fragen: Analysiere jede Option und begründe, warum sie richtig oder falsch ist
6. Wenn Tabellen, Graphen oder andere visuelle Elemente beschrieben sind, nutze diese Informationen für die Lösung
7. Für Aufgaben wie Gewinnmaximierung: Nutze die im Text verfügbaren Daten (z.B. Graphenpunkte) und setze Standardwerte wie kv = 3 und kf = 20, wenn nicht anders angegeben, aber dokumentiere dies als Annahme
8. Die Endantwort MUSS exakt der LETZTEN berechneten Zahl entsprechen (z.B. 11.50 nach Korrektur) und auf zwei Dezimalstellen formatiert sein

AUSGABEFORMAT (STRIKT EINHALTEN):
Aufgabe [Nummer]: [Antwort auf zwei Dezimalstellen, basierend auf der letzten Berechnung]
Begründung: [Schritt-für-Schritt-Erklärung inklusive Korrekturen]
Berechnung: [Mathematische Schritte, markiere die letzte berechnete Zahl klar]
Annahmen (falls nötig): [z.B. "Preis-Satzfunktion wurde aus Graphendaten abgeleitet" oder "kv = 3, kf = 20 als Standardwerte angenommen"]

Wiederhole dies für JEDE Aufgabe im Text.

Beispiel:
Aufgabe 48: 11.50
Begründung: Der gewinnmaximale Preis wird durch Ableiten der Gewinnfunktion bestimmt... Initiale Annahme war falsch, nach Korrektur...
Berechnung: x = 450 - 22.5·p (aus Graphen), G(p) = (p - 3)·(450 - 22.5·p) - 20, dG/dp = 0, p = 517.5/45 = 11.50 (letzte berechnete Zahl)
Annahmen: Preis-Satzfunktion aus Graphendaten (0, 450) und (20, 0), kv = 3, kf = 20 als Standardwerte

WICHTIG: Vergiss keine Aufgabe!"""

    client = OpenAI(api_key=st.secrets["openai_key"])
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Beantworte ALLE Aufgaben die im Text stehen. Überspringe keine. Nutze die letzte berechnete Antwort als Endantwort."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
        temperature=0.1
    )
    
    return response.choices[0].message.content

# --- Verbesserte Ausgabeformatierung mit Korrekturprüfung ---
def parse_and_display_solution(solution_text, model_name="Claude"):
    """Parst und zeigt Lösung strukturiert an, prüft Konsistenz mit letzter Berechnung"""
    
    task_pattern = r'Aufgabe\s+(\d+)\s*:\s*([^\n]+)'
    tasks = re.findall(task_pattern, solution_text, re.IGNORECASE)
    
    if not tasks:
        st.warning(f"⚠️ Keine Aufgaben im erwarteten Format gefunden ({model_name})")
        st.markdown(solution_text)
        return
    
    for task_num, answer in tasks:
        st.markdown(f"### Aufgabe {task_num}: **{answer.strip()}** ({model_name})")
        
        begr_pattern = rf'Aufgabe\s+{task_num}\s*:.*?\n\s*Begründung:\s*([^\n]+(?:\n(?!Aufgabe)[^\n]+)*?)(?:\n\s*Berechnung:\s*([^\n]+(?:\n(?!Aufgabe)[^\n]+)*))?(?:\n\s*Annahmen\s*\(falls\s*nötig\):\s*([^\n]+(?:\n(?!Aufgabe)[^\n]+)*))?(?=\n\s*Aufgabe|\Z)'
        begr_match = re.search(begr_pattern, solution_text, re.IGNORECASE | re.DOTALL)
        
        if begr_match:
            st.markdown(f"*Begründung: {begr_match.group(1).strip()}*")
            if begr_match.group(2):
                st.markdown(f"*Berechnung: {begr_match.group(2).strip()}*")
                calc_pattern = r'p\s*=\s*([\d,.]+)\s*\(letzte\s*berechnete\s*Zahl\)'
                calc_match = re.search(calc_pattern, begr_match.group(2), re.IGNORECASE)
                if not calc_match:
                    calc_pattern_fallback = r'p\s*=\s*([\d,.]+)(?=\s|$|\n)'
                    calc_match = re.search(calc_pattern_fallback, begr_match.group(2), re.IGNORECASE)
                if calc_match:
                    calc_answer = calc_match.group(1).replace(',', '.')
                    if calc_answer != answer.strip():
                        st.warning(f"⚠️ Inkonsistenz in Aufgabe {task_num} ({model_name}): Endantwort ({answer.strip()}) unterscheidet sich von letzter Berechnung ({calc_answer})")
            if begr_match.group(3):
                st.markdown(f"*Annahmen: {begr_match.group(3).strip()}*")
        
        st.markdown("---")

# --- UI ---
if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    st.rerun()

debug_mode = st.checkbox("🔍 Debug-Modus", value=True)

uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        image = Image.open(uploaded_file)
        
        st.image(image, caption=f"Originalbild ({image.width}x{image.height}px)", use_container_width=True)
        
        with st.spinner("Lese Aufgabe mit Gemini..."):
            ocr_text = extract_text_with_gemini_improved(image, file_hash)
        
        with st.expander(f"OCR-Ergebnis ({len(ocr_text)} Zeichen)", expanded=debug_mode):
            st.code(ocr_text)
            
            found_tasks = re.findall(r'Aufgabe\s+\d+', ocr_text, re.IGNORECASE)
            if found_tasks:
                st.success(f"✅ Gefundene Aufgaben: {', '.join(found_tasks)}")
            else:
                st.error("❌ Keine Aufgaben im Text gefunden!")
            
            if "Graph:" in ocr_text or "Table:" in ocr_text:
                st.success("✅ Graphen oder Tabellen im OCR-Text gefunden!")
            
            found_data = re.findall(r'Point at \(\d+,\s*\d+\)|[A-Za-z]+:\s*\d+', ocr_text)
            if found_data:
                st.success(f"✅ Erkannte Daten: {', '.join(found_data)}")
        
        if st.button("Aufgabe lösen", type="primary"):
            st.markdown("---")
            
            with st.spinner("🧮 Claude & GPT-4 lösen Aufgabe..."):
                claude_solution = solve_with_claude_formatted(ocr_text)
                gpt_solution = solve_with_gpt(ocr_text)
                
                is_similar, claude_answers, gpt_answers, numerical_differences = are_answers_similar(claude_solution, gpt_solution)
                if is_similar:
                    st.success("✅ Beide Modelle sind einig!")
                    st.markdown("### Lösungen (Claude):")
                    parse_and_display_solution(claude_solution, model_name="Claude")
                else:
                    st.warning("⚠️ Modelle uneinig! Zeige beide Lösungen zur Überprüfung.")
                    st.markdown("### Lösungen (Claude):")
                    parse_and_display_solution(claude_solution, model_name="Claude")
                    st.markdown("### Lösungen (GPT-4 Turbo):")
                    parse_and_display_solution(gpt_solution, model_name="GPT-4 Turbo")
                    if numerical_differences:
                        st.markdown("### Numerische Unterschiede in Endantworten:")
                        for c_answer, g_answer in numerical_differences:
                            st.markdown(f"- Claude: **{c_answer}**, GPT-4: **{g_answer}**")
            
            if debug_mode:
                with st.expander("💭 Rohe Claude-Antwort"):
                    st.code(claude_solution)
                with st.expander("💭 Rohe GPT-4-Antwort"):
                    st.code(gpt_solution)
                    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

st.markdown("---")
st.caption("Made by Fox with Gemini Flash 1.5, Claude Opus 4, GPT-4 & sponsored with I love you Token™️ by Big Koi-9 ❤️")
