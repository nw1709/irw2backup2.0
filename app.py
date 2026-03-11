import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import os

# --- 1. UI SETUP ---
st.set_page_config(layout="wide", page_title="KFB3", page_icon="🦊")

st.markdown(f'''
<link rel="apple-touch-icon" sizes="180x180" href="https://em-content.zobj.net/thumbs/120/apple/325/fox-face_1f98a.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#FF6600"> 
''', unsafe_allow_html=True)

st.title("🦊 KFB3")

# --- 2. API KONFIGURATION (JETZT MIT RETRY-LOGIK) ---
def get_client():
    if 'gemini_key' not in st.secrets:
        st.error("API Key fehlt. Bitte in den Secrets hinterlegen.")
        st.stop()
    
    # Konfiguration der Wiederholungsversuche bei Serverfehlern (503, 504 etc.)
    retry_options = types.HttpRetryOptions(
        initial_delay=2.0,  # 2 Sekunden warten nach dem ersten Fehler
        attempts=6,         # Insgesamt 6 Versuche (ca. 1-2 Minuten Puffer)
        exp_base=2.0,       # Zeit zwischen Versuchen verdoppelt sich
        max_delay=30.0,     # Maximal 30s Pause zwischen zwei Versuchen
        http_status_codes=[429, 500, 502, 503, 504] # Fehler, bei denen wiederholt wird
    )

    return genai.Client(
        api_key=st.secrets["gemini_key"],
        http_options=types.HttpOptions(retry_options=retry_options)
    )

client = get_client()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    pdfs = st.file_uploader("PDF-Skripte hochladen", type=["pdf"], accept_multiple_files=True)
    if pdfs:
        st.success(f"{len(pdfs)} Skripte geladen.")
    st.divider()
    st.info("model: Gemini 3.1 Pro Preview mit Re-Try")

# --- 4. DER MASTER-SOLVER ---
def solve_everything(image, pdf_files):
    try:
        sys_instr = """...""" # Dein System-Prompt bleibt hier identisch

        # Multimodaler Input
        parts = []
        if pdf_files:
            for pdf in pdf_files:
                # Wir lesen die PDF-Daten einmal ein
                pdf_data = pdf.read()
                parts.append(types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"))
                # Zeiger zurücksetzen, falls die Funktion mehrfach aufgerufen wird
                pdf.seek(0)
        
        # Bildbytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        parts.append(types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/jpeg"))
        
        # Auftrag
        parts.append("Löse ALLE Aufgaben auf dem Bild unter strikter Einhaltung deines Lösungsprozesses")

        # API Aufruf (Der Client nutzt nun automatisch die Retry-Logik aus Step 2)
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=sys_instr,
                temperature=0,
                max_output_tokens=15000,
            )
        )

        return response.text

    except Exception as e:
        # Spezifische Fehlermeldung für den User
        if "503" in str(e) or "overloaded" in str(e).lower():
            return "Fehler: Die Google-Server sind aktuell überlastet. Trotz 6 Wiederholungsversuchen konnte keine Antwort geladen werden. Bitte in 2 Minuten erneut versuchen."
        return f"Fehler: {str(e)}"

# --- 5. UI LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Klausurblatt hochladen...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        if "rot" not in st.session_state: st.session_state.rot = 0
        if st.button("🔄 Bild drehen"): st.session_state.rot = (st.session_state.rot + 90) % 360
        img = img.rotate(-st.session_state.rot, expand=True)
        st.image(img, use_container_width="stretch")

with col2:
    if uploaded_file:
        if st.button("Aufgaben lösen", type="primary"):
            # Ein Status-Container sieht professioneller aus
            status_container = st.empty()
            with status_container.status("Gemini 3.1 Pro analysiert... (bei Störungen werden autom. Retries durchgeführt)", expanded=True) as status:
                result = solve_everything(img, pdfs)
                st.markdown("### Ergebnis")
                st.write(result)
                status.update(label="Analyse abgeschlossen!", state="complete", expanded=False)
    else:
        st.info("Bitte lade links ein Bild hoch.")
