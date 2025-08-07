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
st.set_page_config(layout="wide", page_title="Koifox-Bot 4.3", page_icon="🦊")
st.title("🦊 Koifox-Bot 3: Multi-Experten-Validierung")
st.markdown("Gemini 2.5 Pro, GPT o3 & Claude Opus 4.1 zur Kreuzvalidierung")

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
3.  **Schritt-für-Schritt-Lösung**: Entwickeln Sie Ihre Lösung stringent nach der Methodik der Fernuni Hagen. Legen Sie Ihre Rechenschritte intern offen.
4.  **Multiple Choice**: Bewerten Sie jede einzelne Option (A, B, C, D) individuell und begründen Sie, warum sie richtig oder falsch ist, basierend auf Ihren Berechnungen.
5.  **Formatierung**: Geben Sie die finale Antwort für JEDE gefundene Aufgabe in diesem EXAKTEN Format aus. Fügen Sie keine weiteren Erklärungen außerhalb dieses Formats hinzu.

Aufgabe [Nummer der Aufgabe]: [Finale Antwort, z.B. eine Zahl, ein Buchstabe oder ein kurzer Satz]
Begründung: [Ein einzelner, prägnanter Satz, der die Herleitung auf den Punkt bringt.]

**Selbstprüfung (KRITISCH)**: Bevor Sie Ihre Antwort ausgeben, überprüfen Sie Ihr Ergebnis nochmals anhand der Daten aus der Aufgabenstellung. Stellen Sie absolut sicher, dass es zu 100% den Standards und der Methodik der Fernuni Hagen entspricht.
"""

# --- HELFERFUNKTIONEN ---
@st.cache_data
def pdf_to_images(pdf_bytes):
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)))
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


# --- API-AUFRUFFUNKTIONEN ---
def call_gemini(image_list):
    try:
        prompt_parts = [EXPERT_PROMPT] + image_list
        response = gemini_model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        return f"Fehler bei Gemini API: {e}"

# KORRIGIERT: Die Erstellung der 'messages'-Liste wurde zur besseren Lesbarkeit und zur Vermeidung von Syntaxfehlern umstrukturiert.
def call_gpt(base64_image_list):
    content = [{"type": "text", "text": EXPERT_PROMPT}]
    for b64_img in base64_image_list:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}})
    
    messages = [{"role": "user", "content": content}]
    
    try:
        response = openai_client.chat.completions.create(model=GPT_MODEL_NAME, messages=messages, max_completion_tokens=1500)
        return response.choices[0].message.content
    except Exception as e:
        return f"Fehler bei OpenAI API: {e}"

# KORRIGIERT: Auch hier wurde die 'messages'-Liste zur besseren Lesbarkeit umstrukturiert.
def call_claude(base64_image_list):
    content = [{"type": "text", "text": EXPERT_PROMPT}]
    for b64_img in base64_image_list:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_img}})

    messages = [{"role": "user", "content": content}]
    
    try:
        response = anthropic_client.messages.create(model=CLAUDE_MODEL_NAME, max_tokens=1500, messages=messages)
        return response.content[0].text
    except Exception as e:
        return f"Fehler bei Anthropic API: {e}"


# --- STREAMLIT BENUTZEROBERFLÄCHE UND HAUPTLOGIK ---
st.sidebar.header("Steuerung")
uploaded_file = st.sidebar.file_uploader("Klausurdatei hochladen (JPG, PNG, PDF)", type=["jpg", "jpeg", "png", "pdf"])
solve_button = st.sidebar.button("✨ Aufgaben mit 3 Modellen lösen", type="primary", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.header("Hochgeladenes Dokument")
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        images_to_process = []
        if uploaded_file.type == "application/pdf":
            with st.spinner("PDF wird in Bilder umgewandelt..."):
                images_to_process = pdf_to_images(file_bytes)
        else:
            try:
                images_to_process = [Image.open(io.BytesIO(file_bytes))]
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
                    future_gemini = executor.submit(call_gemini, pil_images)
                    future_gpt = executor.submit(call_gpt, base64_images)
                    future_claude = executor.submit(call_claude, base64_images)
                    
                    gemini_raw_solution = future_gemini.result()
                    gpt_raw_solution = future_gpt.result()
                    claude_raw_solution = future_claude.result()

            st.subheader("Runde 1: Erste Einschätzungen (Rohdaten)")
            with st.expander("Antwort von Gemini 2.5 Pro"):
                st.markdown(gemini_raw_solution)
            with st.expander("Antwort von GPT o3"):
                st.markdown(gpt_raw_solution)
            with st.expander("Antwort von Claude Opus 4.1"):
                st.markdown(claude_raw_solution)

            gemini_solutions = parse_solution(gemini_raw_solution)
            gpt_solutions = parse_solution(gpt_raw_solution)
            claude_solutions = parse_solution(claude_raw_solution)

            st.subheader("Validiertes Endergebnis")
            
            all_task_numbers = sorted(list(set(gemini_solutions.keys()) | set(gpt_solutions.keys()) | set(claude_solutions.keys())), key=int)

            if not all_task_numbers:
                st.error("Keines der Modelle konnte eine Aufgabe im erwarteten Format finden. Bitte überprüfen Sie das Bild und die Roh-Antworten der Modelle.")
            
            for task_num in all_task_numbers:
                st.markdown(f"--- \n#### Analyse für Aufgabe {task_num}")
                
                sols = {
                    "Gemini 2.5 Pro": gemini_solutions.get(task_num),
                    "GPT o3": gpt_solutions.get(task_num),
                    "Claude Opus 4.1": claude_solutions.get(task_num)
                }

                answers = [s['answer'] for s in sols.values() if s and 'answer' in s]

                if not answers:
                    st.warning(f"Kein Modell hat eine auswertbare Antwort für Aufgabe {task_num} geliefert.")
                    continue
                
                answer_counts = Counter(answers)
                most_common = answer_counts.most_common(1)[0]

                if most_common[1] >= 2:
                    final_answer = most_common[0]
                    final_reason = ""
                    for model_name, sol_data in sols.items():
                        if sol_data and sol_data.get('answer') == final_answer:
                            final_reason = sol_data.get('reason', 'Keine Begründung angegeben.')
                            break
                    
                    st.success(f"**Konsens-Lösung: {final_answer}**")
                    st.info(f"**Begründung (aus Konsens):** {final_reason}")
                    
                    st.markdown("**Detail-Übersicht:**")
                    for model_name, sol_data in sols.items():
                        if sol_data and sol_data.get('answer') == final_answer:
                            st.markdown(f"- ✅ **{model_name}:** `{sol_data['answer']}` (Stimmt überein)")
                        elif sol_data:
                            st.markdown(f"- ❌ **{model_name}:** `{sol_data['answer']}` (Weicht ab)")
                        else:
                            st.markdown(f"- ❓ **{model_name}:** (Keine Antwort gefunden)")
                else:
                    st.warning(f"**Kein Konsens für Aufgabe {task_num}. Die Experten sind sich uneinig.**")
                    st.markdown("**Die unterschiedlichen Antworten:**")
                    for model_name, sol_data in sols.items():
                        if sol_data:
                            st.markdown(f"- **{model_name}:** `{sol_data['answer']}` \n  - *Begründung:* {sol_data['reason']}")
                        else:
                            st.markdown(f"- **{model_name}:** (Keine Antwort gefunden)")
                    st.info("Hier könnte eine automatische Debatten-Runde gestartet werden, um eine finale Antwort zu erzwingen.")

        else:
            st.warning("Bitte laden Sie zuerst eine Datei hoch, damit die Experten sie analysieren können.")
