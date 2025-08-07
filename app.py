import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from PIL import Image
import fitz  # PyMuPDF
import io
import re
import base64
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# --- SEITE EINRICHTEN ---
st.set_page_config(layout="wide", page_title="Koifox-Bot 5.0 (Debatte)", page_icon="🦊")
st.title("🦊 Koifox-Bot 5.0: Experten-Panel mit Debatten-Runde")
st.markdown("Ein Bot, der bei Uneinigkeit automatisch eine Debatte zwischen Gemini, GPT und Claude startet, um eine finale Lösung zu finden.")

# --- API CLIENT INITIALISIERUNG ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    
    GEMINI_MODEL_NAME = "gemini-1.5-pro-latest" 
    GPT_MODEL_NAME = "o3"
    CLAUDE_MODEL_NAME = "claude-opus-4-1-20250805"
    
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

except (KeyError, Exception) as e:
    st.error(f"Fehler bei der Initialisierung der API-Clients. Bitte prüfen Sie Ihre API-Keys in den Streamlit Secrets. Fehler: {e}")
    st.stop()


# --- PROMPTS ---
EXPERT_PROMPT = """
Sie sind ein deutscher Professor für 'Internes Rechnungswesen' (Kurs 31031) an der Fernuniversität Hagen mit dem Ziel, Klausuraufgaben mit 100%iger Genauigkeit zu lösen. Ihre Methodik muss exakt dem entscheidungsorientierten deutschen Controlling-Ansatz entsprechen, wie er in den Kursmaterialien der Fernuni Hagen gelehrt wird.

Ihre Aufgaben:
1.  **Analyse**: Lesen Sie die Aufgabenstellung EXTREM sorgfältig. Identifizieren Sie alle gegebenen Zahlen, Bedingungen und die exakte Fragestellung.
2.  **Diagramme/Graphen**: Verwenden Sie zur Berechnung ausschließlich explizit angegebene Achsenbeschriftungen, Skalenwerte und Schnittpunkte. Extrapolieren Sie nicht und schätzen Sie keine Werte.
3.  **Schritt-für-Schritt-Lösung**: Entwickeln Sie Ihre Lösung stringent nach der Methodik der Fernuni Hagen.
4.  **Formatierung**: Geben Sie die finale Antwort für JEDE gefundene Aufgabe in diesem EXAKTEN Format aus.

Aufgabe [Nummer der Aufgabe]: [Finale Antwort, z.B. eine Zahl, ein Buchstabe oder ein kurzer Satz]
Begründung: [Ein einzelner, prägnanter Satz, der die Herleitung auf den Punkt bringt.]

**Selbstprüfung (KRITISCH)**: Bevor Sie Ihre Antwort ausgeben, überprüfen Sie Ihr Ergebnis nochmals anhand der Daten aus der Aufgabenstellung. Stellen Sie absolut sicher, dass es zu 100% den Standards der Fernuni Hagen entspricht.
"""

# NEU: Prompt-Vorlage für die Debatten-Runde
DEBATE_PROMPT_TEMPLATE = """
Sie sind ein Experte für deutsches Rechnungswesen in einer Expertenrunde. Ihre erste Analyse einer Aufgabe führte zu einem Ergebnis, aber Ihre Kollegen sind zu anderen Schlüssen gekommen. Dies ist Ihre Chance zur Neubewertung.

**URSPRÜNGLICHE AUFGABE:** [Siehe beigefügtes Bild]

**Ihre ursprüngliche Antwort (von Ihnen):**
Aufgabe {task_num}: {your_answer}
Begründung: {your_reason}

**Antwort von Experte B (abweichend):**
Aufgabe {task_num}: {other_answer_1}
Begründung: {other_reason_1}

**Antwort von Experte C (abweichend):**
Aufgabe {task_num}: {other_answer_2}
Begründung: {other_reason_2}

**IHRE NEUE AUFGABE:**
1.  Analysieren Sie die Aufgabe und Ihre ursprüngliche Lösung erneut kritisch.
2.  Bewerten Sie die Logik hinter den abweichenden Antworten der anderen Experten. Finden Sie den Fehler – entweder in Ihrer Analyse oder in der der anderen.
3.  Entscheiden Sie sich: Bestätigen Sie Ihre ursprüngliche Antwort (mit besserer Begründung) oder korrigieren Sie sie.
4.  Geben Sie eine FINALE, endgültige Antwort ab. Halten Sie sich dabei STRIKT an das ursprüngliche Format.

Aufgabe [Nummer]: [Ihre FINALE, überarbeitete Antwort]
Begründung: [Ihre FINALE Begründung, warum diese Antwort korrekt ist.]
"""


# --- HELFERFUNKTIONEN ---
@st.cache_data
def pdf_to_images(pdf_bytes):
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = [Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))) for page in pdf_document]
        pdf_document.close()
        return images
    except Exception as e:
        st.error(f"Fehler bei der PDF-Konvertierung: {e}")
        return []

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_solution(text):
    pattern = re.compile(r"Aufgabe\s*\[?(\d+)\]?:\s*(.*?)\s*\nBegründung:\s*(.*)", re.IGNORECASE)
    matches = pattern.findall(text)
    return {match[0]: {"answer": match[1].strip(), "reason": match[2].strip()} for match in matches}

# --- API-AUFRUFFUNKTIONEN (Jetzt mit "prompt" als Argument) ---
def call_gemini(prompt, image_list):
    try:
        response = gemini_model.generate_content([prompt] + image_list)
        return response.text
    except Exception as e:
        return f"Fehler bei Gemini API: {e}"

def call_gpt(prompt, base64_image_list):
    content = [{"type": "text", "text": prompt}]
    for b64_img in base64_image_list:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "high"}})
    messages = [{"role": "user", "content": content}]
    try:
        response = openai_client.chat.completions.create(model=GPT_MODEL_NAME, messages=messages, max_completion_tokens=1500, timeout=45.0)
        return response.choices[0].message.content if response.choices and response.choices[0].message else "Leere Antwort von GPT."
    except Exception as e:
        return f"Fehler bei OpenAI API: {str(e)}"

def call_claude(prompt, base64_image_list):
    content = [{"type": "text", "text": prompt}]
    for b64_img in base64_image_list:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_img}})
    messages = [{"role": "user", "content": content}]
    try:
        response = anthropic_client.messages.create(model=CLAUDE_MODEL_NAME, max_tokens=1500, messages=messages, timeout=45.0)
        return response.content[0].text if response.content else "Leere Antwort von Claude."
    except Exception as e:
        return f"Fehler bei Anthropic API: {str(e)}"

# --- STREAMLIT UI & HAUPTLOGIK ---
st.sidebar.header("Steuerung")
uploaded_file = st.sidebar.file_uploader("Klausurdatei hochladen (JPG, PNG, PDF)", type=["jpg", "jpeg", "png", "pdf"])
solve_button = st.sidebar.button("✨ Aufgaben mit Experten-Panel lösen", type="primary", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.header("Hochgeladenes Dokument")
    if uploaded_file:
        images_to_process = []
        if uploaded_file.type == "application/pdf":
            with st.spinner("PDF wird in Bilder umgewandelt..."):
                images_to_process = pdf_to_images(uploaded_file.getvalue())
        else:
            try:
                images_to_process = [Image.open(uploaded_file)]
            except Exception as e:
                st.error(f"Fehler beim Öffnen der Bilddatei: {e}")
        if images_to_process:
            st.session_state.images = images_to_process
            st.image(images_to_process, caption=[f"Seite {i+1}" for i in range(len(images_to_process))], use_container_width=True)

with col2:
    st.header("Lösungen der Experten")
    if solve_button:
        if 'images' in st.session_state and st.session_state.images:
            pil_images = st.session_state.images
            base64_images = [image_to_base64(img) for img in pil_images]
            
            with st.spinner("Runde 1: Alle drei Experten analysieren die Aufgaben parallel..."):
                with ThreadPoolExecutor() as executor:
                    future_gemini = executor.submit(call_gemini, EXPERT_PROMPT, pil_images)
                    future_gpt = executor.submit(call_gpt, EXPERT_PROMPT, base64_images)
                    future_claude = executor.submit(call_claude, EXPERT_PROMPT, base64_images)
                    gemini_raw, gpt_raw, claude_raw = future_gemini.result(), future_gpt.result(), future_claude.result()

            st.subheader("Runde 1: Erste Einschätzungen")
            with st.expander("Antwort von Gemini 2.5 Pro"): st.markdown(gemini_raw)
            with st.expander("Antwort von GPT o3"): st.markdown(gpt_raw)
            with st.expander("Antwort von Claude Opus 4.1"): st.markdown(claude_raw)

            gemini_sols, gpt_sols, claude_sols = parse_solution(gemini_raw), parse_solution(gpt_raw), parse_solution(claude_raw)
            all_task_numbers = sorted(list(set(gemini_sols.keys()) | set(gpt_sols.keys()) | set(claude_sols.keys())), key=int)

            st.subheader("Validiertes Endergebnis")
            if not all_task_numbers:
                st.error("Keines der Modelle konnte eine Aufgabe im erwarteten Format finden.")
            
            for task_num in all_task_numbers:
                st.markdown(f"--- \n#### Analyse für Aufgabe {task_num}")
                sols = {"Gemini": gemini_sols.get(task_num), "GPT": gpt_sols.get(task_num), "Claude": claude_sols.get(task_num)}
                answers = [s['answer'] for s in sols.values() if s and 'answer' in s]
                
                if not answers:
                    st.warning(f"Kein Modell hat eine auswertbare Antwort für Aufgabe {task_num} geliefert.")
                    continue
                
                answer_counts = Counter(answers)
                most_common = answer_counts.most_common(1)[0]

                if most_common[1] >= 2:
                    final_answer = most_common[0]
                    final_reason = next((s['reason'] for s in sols.values() if s and s.get('answer') == final_answer), "Keine Begründung gefunden.")
                    st.success(f"**Konsens-Lösung (Runde 1): {final_answer}**")
                    st.info(f"**Begründung:** {final_reason}")
                else:
                    st.warning(f"**Kein Konsens in Runde 1.** Die Experten sind sich uneinig. Starte automatische Debatten-Runde...")
                    
                    # --- DEBATTEN-RUNDE LOGIK ---
                    with st.spinner(f"Debatte für Aufgabe {task_num} läuft... Die Experten bewerten die Meinungen der anderen."):
                        # Erstelle die individuellen Debatten-Prompts
                        prompts = {}
                        model_names = list(sols.keys())
                        for i, name in enumerate(model_names):
                            # Stelle sicher, dass das Modell eine Antwort hatte, um teilzunehmen
                            if not sols[name]: continue

                            # Sammle die Antworten der ANDEREN Modelle
                            other_answers = [sols[other_name] for j, other_name in enumerate(model_names) if i != j and sols[other_name]]
                            
                            prompts[name] = DEBATE_PROMPT_TEMPLATE.format(
                                task_num=task_num,
                                your_answer=sols[name]['answer'],
                                your_reason=sols[name]['reason'],
                                other_answer_1=other_answers[0]['answer'] if len(other_answers) > 0 else "N/A",
                                other_reason_1=other_answers[0]['reason'] if len(other_answers) > 0 else "N/A",
                                other_answer_2=other_answers[1]['answer'] if len(other_answers) > 1 else "N/A",
                                other_reason_2=other_answers[1]['reason'] if len(other_answers) > 1 else "N/A"
                            )

                        # Starte die zweite Runde der API-Aufrufe
                        with ThreadPoolExecutor() as executor:
                            future_gemini_debate = executor.submit(call_gemini, prompts.get("Gemini", ""), pil_images)
                            future_gpt_debate = executor.submit(call_gpt, prompts.get("GPT", ""), base64_images)
                            future_claude_debate = executor.submit(call_claude, prompts.get("Claude", ""), base64_images)
                            gemini_final_raw, gpt_final_raw, claude_final_raw = future_gemini_debate.result(), future_gpt_debate.result(), future_claude_debate.result()
                        
                        st.subheader(f"Runde 2: Finale Antworten nach Debatte (Aufgabe {task_num})")
                        with st.expander("Finale Antwort von Gemini"): st.markdown(gemini_final_raw)
                        with st.expander("Finale Antwort von GPT"): st.markdown(gpt_final_raw)
                        with st.expander("Finale Antwort von Claude"): st.markdown(claude_final_raw)
                        
                        # Finale Auswertung nach der Debatte
                        final_sols = {
                            "Gemini": parse_solution(gemini_final_raw).get(task_num),
                            "GPT": parse_solution(gpt_final_raw).get(task_num),
                            "Claude": parse_solution(claude_final_raw).get(task_num)
                        }
                        final_answers = [s['answer'] for s in final_sols.values() if s and 'answer' in s]
                        final_answer_counts = Counter(final_answers)
                        most_common_final = final_answer_counts.most_common(1)[0] if final_answers else (None, 0)
                        
                        if most_common_final[1] >= 2:
                            final_answer = most_common_final[0]
                            final_reason = next((s['reason'] for s in final_sols.values() if s and s.get('answer') == final_answer), "Keine Begründung gefunden.")
                            st.success(f"**Konsens nach Debatte: {final_answer}**")
                            st.info(f"**Finale Begründung:** {final_reason}")
                        else:
                            st.error(f"**Kein Konsens nach Debatte für Aufgabe {task_num}.**")
                            st.write("Die Experten konnten sich nicht einigen. Hier sind die finalen, unterschiedlichen Meinungen:")
                            for name, sol in final_sols.items():
                                if sol:
                                    st.markdown(f"**{name}:** `{sol['answer']}` \n- *Begründung:* {sol['reason']}")
        else:
            st.warning("Bitte laden Sie zuerst eine Datei hoch, damit die Experten sie analysieren können.")
