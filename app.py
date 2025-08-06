import streamlit as st
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF
import io


try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    st.error("Please configure your Google API key in the Streamlit secrets.")


# --- Gemini Model Initialization ---
# Initialize the Gemini Pro Vision model
model = genai.GenerativeModel('gemini-pro-vision')


# --- Helper Function for PDF to Images ---
def pdf_to_images(pdf_file):
    """Converts a PDF file to a list of PIL Images."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            images.append(image)
        pdf_document.close()
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return []


# --- Streamlit App Interface ---
st.set_page_config(page_title="Gemini Exam Solver", layout="wide")
st.title("Koifoxbot Gemini 2.5 Pro")
st.write("Upload your exam question")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload your exam file",
        type=["jpg", "jpeg", "png", "pdf"]
    )
    user_prompt = st.text_area(
        "Your Prompt",
        (
            "Please act as an expert in the subject matter of the following text. "
            "Thoroughly analyze the extracted text and provide a detailed, step-by-step solution "
            "to the following question:"
        ),
        height=150
    )
    question = st.text_input("Enter the specific question to solve from the document")
    solve_button = st.button("Solve the Question")

# --- Main Content Area ---
col1, col2 = st.columns(2)

with col1:
    st.header("Uploaded Document")
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            with st.spinner("Converting PDF to images..."):
                images = pdf_to_images(uploaded_file)
                if images:
                    st.image(images, caption=[f"Page {i+1}" for i in range(len(images))], use_column_width=True)
        else:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error opening image file: {e}")

with col2:
    st.header("Solution")
    if solve_button and uploaded_file and user_prompt and question:
        with st.spinner("Gemini is thinking..."):
            try:
                # Prepare the image(s) for the model
                if uploaded_file.type == "application/pdf":
                    if 'images' in locals() and images:
                        image_parts = images
                    else:
                        st.error("Could not process the PDF file.")
                        image_parts = []
                else:
                    if 'image' in locals():
                        image_parts = [image]
                    else:
                        st.error("Could not process the image file.")
                        image_parts = []

                if image_parts:
                    # Construct the final prompt for Gemini
                    final_prompt = f"{user_prompt}\n\nQuestion: {question}"

                    # Prepare the content for the API call
                    prompt_parts = [final_prompt] + image_parts

                    # Call the Gemini API
                    response = model.generate_content(prompt_parts)

                    # Display the response
                    st.markdown(response.text)

            except Exception as e:
                st.error(f"An error occurred while communicating with the Gemini API: {e}")
    elif solve_button:
        st.warning("Please upload a file, enter a prompt, and a question before solving.")

