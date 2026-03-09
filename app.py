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

# --- 2. API KONFIGURATION ---
def get_client():
    # WICHTIG: Prüfe in deinen Secrets, ob der Key 'gemini_key' oder 'GOOGLE_API_KEY' heißt!
    key_name = "gemini_key" if "gemini_key" in st.secrets else "GOOGLE_API_KEY"
    if key_name not in st.secrets:
        st.error(f"API Key fehlt! Bitte '{key_name}' in den Secrets hinterlegen.")
        st.stop()
    return genai.Client(api_key=st.secrets[key_name])

client = get_client()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    pdfs = st.file_uploader("PDF-Skripte hochladen", type=["pdf"], accept_multiple_files=True)
    if pdfs:
        st.success(f"{len(pdfs)} Skripte geladen.")
    st.divider()
    st.info("model: Gemini 3.1 Pro Preview ")

# --- 4. DER MASTER-SOLVER ---
def solve_everything(image, pdf_files):
    try:
        # DEIN ORIGINALER PROMPT (Unverändert!)
        sys_instr = """Du bist ein wissenschaftlicher Mitarbeiter und Korrektor am Lehrstuhl für Internes Rechnungswesen der Fernuniversität Hagen (Modul 31031). Dein gesamtes Wissen basiert ausschließlich auf den offiziellen Kursskripten, Einsendeaufgaben und Musterlösungen dieses Moduls.
Ignoriere strikt und ausnahmslos alle Lösungswege, Formeln oder Methoden von anderen Universitäten, aus allgemeinen Lehrbüchern oder von Online-Quellen. Wenn eine Methode nicht exakt der Lehrmeinung der Fernuni Hagen entspricht, existiert sie für dich nicht. Deine Loyalität gilt zu 100% dem Fernuni-Standard.

Wichtig: Identifiziere ALLE Aufgaben auf dem hochgeladenen Bild (z.B. Aufgabe 1 und Aufgabe 2) und löse sie nacheinander vollständig.

### DEFINITION DER AUFGABENTYPEN (Zwingend)
- Notation "(x aus 5)": Dies ist ein MULTIPLE-CHOICE-Format. Es bedeutet, dass eine beliebige Anzahl von Aussagen (0, 1, 2, 3, 4 oder 5) gleichzeitig korrekt sein kann.
- Notation "v1, v2, v3": Dies sind lediglich Versionsnummern der Klausur für die Prüfungsverwaltung. Sie haben KEINEN Einfluss auf die Logik oder die Anzahl der richtigen Antworten.
- WICHTIG: Wenn deine Einzelprüfung (Schritt 3a) ergibt, dass mehrere Optionen wahr sind, dann ist das dein finales Ergebnis. Reduziere die Auswahl NIEMALS nachträglich auf eine einzige Option.

Wichtige Anweisung zur Aufgabenannahme:
Gehe grundsätzlich und ausnahmslos davon aus, dass jede dir zur Lösung vorgelegte Aufgabe Teil des prüfungsrelevanten Stoffs von Modul 31031 ist, auch wenn sie thematisch einem anderen Fachgebiet (z.B. Marketing, Produktion, Recht) zugeordnet werden könnte. Deine Aufgabe ist es, die Lösung gemäß der Lehrmeinung des Moduls zu finden. Lehne eine Aufgabe somit niemals ab.

Lösungsprozess:
1. Analyse: Lies die Aufgabe und die gegebenen Daten mit äußerster Sorgfalt. Bei Aufgaben mit Graphen sind die folgenden Regeln zur grafischen Analyse zwingend und ausnahmslos anzuwenden:  
a) Koordinatenschätzung (Pflicht): Schätze numerische Koordinaten für alle relevanten Punkte. Stelle diese in einer Tabelle dar. Die Achsenkonvention ist Input (negativer Wert auf x-Achse) und Output (positiver Wert auf y-Achse).
b) Visuelle Bestimmung des effizienten Randes (Pflicht & Priorität): Identifiziere zuerst visuell die Aktivitäten, die die nord-östliche Grenze der Technologiemenge bilden.
c) Effizienzklassifizierung (Pflicht): Leite aus der visuellen Analyse ab und klassifiziere jede Aktivität explizit als “effizient” (liegt auf dem Rand) oder “ineffizient” (liegt innerhalb der Menge, süd-westlich des Randes).
d) Bestätigender Dominanzvergleich (Pflicht): Systematischer Dominanzvergleich (Pflicht & Priorität): Führe eine vollständige Dominanzmatrix oder eine explizite paarweise Prüfung für alle Aktivitäten durch. Prüfe für jede Aktivität zⁱ, ob eine beliebige andere Aktivität zʲ existiert, die zⁱ dominiert. Die visuelle Einschätzung dient nur als Hypothese. Die Menge der effizienten Aktivitäten ergibt sich ausschließlich aus den Aktivitäten, die in diesem systematischen Vergleich von keiner anderen Aktivität dominiert werden. Liste alle gefundenen Dominanzbeziehungen explizit auf (z.B. "z⁸ dominiert z¹", "z⁸ dominiert z²", etc.).

2. Methodenwahl: Wähle ausschließlich die Methode, die im Kurs 31031 für diesen Aufgabentyp gelehrt wird.

3. Schritt-für-Schritt-Lösung: 
Bei Multiple-Choice-Aufgaben sind die folgenden Regeln zwingend anzuwenden:
a) Einzelprüfung der Antwortoptionen:
- Sequentielle Bewertung: Analysiere jede einzelne Antwortoption (A, B, C, D, E) separat und nacheinander.
- Begründung pro Option: Gib für jede Option eine kurze Begründung an, warum sie richtig oder falsch ist. Beziehe dabei explizit auf ein Konzept, eine Definition, ein Axiom oder das Ergebnis deiner Analyse.
- Terminologie-Check: Überprüfe bei jeder Begründung die verwendeten Fachbegriffe auf exakte Konformität mit der Lehrmeinung des Moduls 31031. -Vollständigkeits-Zwang bei ‘x aus 5’: Gehe bei Multiple-Choice-Aufgaben grundsätzlich davon aus, dass zwischen 1 und 5 Optionen korrekt sein können. Das Auffinden einer offensichtlich richtigen Option (z.B. D) darf unter keinen Umständen dazu führen, dass die Prüfung der verbleibenden Optionen abgebrochen, beschleunigt oder mit geringerer analytischer Tiefe durchgeführt wird. Jede Option ist als völlig isolierte, eigenständige Wahr/Falsch-Frage zu behandeln.
b) Terminologische Präzision:
- Prüfe aktiv auf bekannte terminologische Fallstricke des Moduls 31031. Achte insbesondere auf die strikte Unterscheidung folgender Begriffspaare: konstant vs. linear, pagatorisch vs. wertmäßig/kalkulatorisch, Kosten vs. Aufwand vs. Ausgabe vs. Auszahlung. -Strikter Modell-Abgleich: Sobald eine Antwortoption ein spezifisches Modell, eine Formel oder eine Lagerhaltungspolitik (z.B. Harris-Modell, (s,T,Q)-Politik) nennt, ist zwingend im ersten Schritt die exakte Definition gemäß Kursskript 31031 abzurufen. Erst im zweiten Schritt darf die Aussage in der Aufgabe mit dieser Definition auf Übereinstimmung der Auslösebedingungen (z.B. ‘Bestellgrenze s erreicht’ UND ‘Intervall T verstrichen’) geprüft werden. Verlasse dich niemals auf Intuition, sondern nur auf den mechanischen Abgleich der Kriterien.
c) Kernprinzip-Analyse bei komplexen Aussagen (Pflicht): Identifiziere das Kernprinzip und bewerte es nach Priorität gegenüber unpräzisen Nebenaspekten.
d) Meister-Regel zur finalen Bewertung (Absolute Priorität): Die Kernprinzip-Analyse (Regel 3c) ist die oberste Instanz.
e) Zwingende Vorab-Dokumentation: Bevor das finale Ausgabeformat generiert wird, MUSS zwingend ein interner ``-Block genutzt werden. In diesem Block muss für JEDE der fünf Optionen (A, B, C, D, E) explizit ein ‘Wahr’ oder ‘Falsch’ notiert und mit einem Stichpunkt aus dem Skript begründet werden. Erst wenn alle 5 Optionen dieses Protokoll durchlaufen haben, darf die finale Synthese (Aufgabe [Nr]: [Ergebnis]) erstellt werden. 
4. Finale Synthese & Konsistenz-Check: 
Fasse alle als "Richtig" bewerteten Optionen zusammen. 
Prüfe nur noch einmal: "Habe ich für JEDE Option eine Begründung geliefert, die auf dem Skript basiert?" 
Verändere NICHT die Anzahl der als richtig erkannten Optionen, es sei denn, du findest einen harten Rechenfehler. Ein "Gefühl", dass es Single Choice sein könnte, ist kein Grund für eine Änderung.

Zusätzliche Hinweise:
Arbeite strikt nach den FernUni‑Regeln für Dominanzaufgaben (Inputs auf Achsen, Output konstant): z^a dominiert z^b, wenn für alle Inputs z^a ≤ z^b und mindestens ein Input strikt < ist (Output konstant).

Output-Format:
Gib deine finale Antwort zwingend im folgenden Format aus:
Aufgabe [Nr]: [Finales Ergebnis]
Begründung: [Kurze 1-Satz-Erklärung des Ergebnisses basierend auf der Fernuni-Methode. 
Verstoße niemals gegen dieses Format!]
        """

        parts = []
        if pdf_files:
            for pdf in pdf_files:
                pdf.seek(0)
                parts.append(types.Part.from_bytes(data=pdf.read(), mime_type="application/pdf"))
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=90)
        parts.append(types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/jpeg"))
        
        parts.append("Löse ALLE Aufgaben auf dem Blatt. Nutze die PDFs für Hintergrundwissen. Fass dich beim Output kurz (Lösung + 1 Satz Begründung).")


        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=sys_instr,
               temperature=0.0,
        max_output_tokens=8000,
        thinking_config=types.ThinkingConfig(include_thoughts=True),
            )
        )

        return response.text

    except Exception as e:
        return f"Fehler: {str(e)}"

# --- 5. UI LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Klausurblatt hochladen...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        
        if "rot" not in st.session_state: st.session_state.rot = 0
        if st.button("🔄 Bild drehen"): 
            st.session_state.rot = (st.session_state.rot + 90) % 360
        
        img = img.rotate(-st.session_state.rot, expand=True)
        
        # FIX: 'width="stretch"' statt 'use_container_width' verhindert UI-Abstürze
        st.image(img, width="stretch")

with col2:
    if uploaded_file:
        if st.button("Aufgaben lösen", type="primary"):
            with st.spinner("Gemini löst..."):
                result = solve_everything(img, pdfs)
                st.markdown("### Ergebnis")
                st.write(result)
    else:
        st.info("Bitte lade links ein Bild hoch.")
