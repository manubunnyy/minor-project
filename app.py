import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import re
import google.generativeai as genai
import torch
import base64
from io import BytesIO
import tempfile
import os

# -------------------- Hardcoded (but obfuscated) Gemini API Key --------------------
part1 = "enter_api_key_here"
part2 = "enter_api_key_here"
GEMINI_API_KEY = part1 + part2

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model for chat
@st.cache_resource
def load_chat_model():
    return genai.GenerativeModel('gemini-1.5-flash')

# -------------------- Load EasyOCR with GPU if available --------------------
@st.cache_resource
def load_easyocr_reader(langs=['en']):
    use_gpu = torch.cuda.is_available()
    return easyocr.Reader(langs, gpu=use_gpu)

# -------------------- Preprocessing --------------------
def enhance_image_for_ocr(image, page_idx=0, config=None):
    if config is None:
        config = {}
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    out_img = gray.copy()
    steps = {}
    steps['Grayscale'] = out_img.copy()

    # Invert image
    if config.get('invert', False):
        out_img = cv2.bitwise_not(out_img)
        steps['Inverted'] = out_img.copy()

    # Remove noise (median blur)
    if config.get('remove_noise', False):
        out_img = cv2.medianBlur(out_img, 3)
        steps['Denoised'] = out_img.copy()

    # Dilation and Erosion
    if config.get('dilate', False):
        kernel = np.ones((2,2), np.uint8)
        out_img = cv2.dilate(out_img, kernel, iterations=1)
        steps['Dilated'] = out_img.copy()
    if config.get('erode', False):
        kernel = np.ones((2,2), np.uint8)
        out_img = cv2.erode(out_img, kernel, iterations=1)
        steps['Eroded'] = out_img.copy()

    # Auto Deskew
    if config.get('auto_deskew', False):
        coords = np.column_stack(np.where(out_img < 255))
        if coords.shape[0] > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = out_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            out_img = cv2.warpAffine(out_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            steps['Deskewed'] = out_img.copy()

    # Remove borders
    if config.get('remove_borders', False):
        contours, _ = cv2.findContours(out_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            out_img = out_img[y:y+h, x:x+w]
            steps['Borders Removed'] = out_img.copy()

    # Add missing borders (pad image)
    if config.get('add_borders', False):
        out_img = cv2.copyMakeBorder(out_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        steps['Borders Added'] = out_img.copy()

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out_img = clahe.apply(out_img)
    steps['CLAHE Enhanced'] = out_img.copy()

    # Otsu Binarization
    _, binary = cv2.threshold(out_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps['Otsu Thresholded'] = binary.copy()

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), steps

def process_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [
        np.array(Image.frombytes("RGB", [page.get_pixmap().width, page.get_pixmap().height], page.get_pixmap().samples))
        for page in pdf_document
    ]

def process_image(image):
    image = np.array(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def clean_text(text):
    text = re.sub(r'[^\w\s\u0900-\u097F\u0C00-\u0C7F\u0D00-\u0D7F]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# -------------------- OCR Engines --------------------
def easyocr_ocr(image_np, reader):
    results = reader.readtext(
        image_np,
        detail=1,           # get bbox + text + confidence
        paragraph=False,    # raw lines, not merged paragraphs
        batch_size=8,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=3.0
    )
    
    # Sort results by y-coordinate first, then x-coordinate
    sorted_results = sorted(results, key=lambda x: (min(point[1] for point in x[0]), min(point[0] for point in x[0])))
    
    # Group results by approximate y-coordinate (same line)
    lines = []
    current_line = []
    current_y = None
    line_threshold = 10  # pixel tolerance for grouping into lines
    
    for bbox, text, conf in sorted_results:
        y = min(point[1] for point in bbox)
        if current_y is None or abs(y - current_y) < line_threshold:
            current_line.append((text, bbox))
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            # Sort current line by x-coordinate
            current_line.sort(key=lambda x: min(point[0] for point in x[1]))
            lines.append([text for text, _ in current_line])
            current_line = [(text, bbox)]
            current_y = y
    
    if current_line:
        current_line.sort(key=lambda x: min(point[0] for point in x[1]))
        lines.append([text for text, _ in current_line])
    
    # Join lines with proper spacing
    formatted_text = "\n".join([" ".join(line) for line in lines])
    return formatted_text.strip(), sorted_results

def gemini_ocr(image_pil):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(["Extract all the text from this image:", image_pil])
        return response.text.strip()
    except Exception:
        return ""

# -------------------- Billing OCR Specific Functions --------------------
def extract_billing_info(text, results=None):
    # Common patterns for billing information
    patterns = {
        'invoice_number': r'(?i)invoice\s*(?:no|number|#)?[:#]?\s*([A-Z0-9-]+)',
        'date': r'(?i)(?:date|dated)[:#]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        'amount': r'(?i)(?:total|amount|sum)[:#]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
        'gst': r'(?i)(?:GST|CGST|SGST)[:#]?\s*(?:Rs\.?|INR)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
    }
    
    extracted_info = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            extracted_info[key] = match.group(1)
    
    # If we have EasyOCR results, use them to preserve structure
    if results:
        # Sort results by y-coordinate first, then x-coordinate
        sorted_results = sorted(results, key=lambda x: (min(point[1] for point in x[0]), min(point[0] for point in x[0])))
        
        # Group results by approximate y-coordinate (same line)
        lines = []
        current_line = []
        current_y = None
        line_threshold = 10  # pixel tolerance for grouping into lines
        
        for bbox, text, conf in sorted_results:
            y = min(point[1] for point in bbox)
            if current_y is None or abs(y - current_y) < line_threshold:
                current_line.append((text, bbox))
                current_y = y if current_y is None else (current_y + y) / 2
            else:
                # Sort current line by x-coordinate
                current_line.sort(key=lambda x: min(point[0] for point in x[1]))
                lines.append([text for text, _ in current_line])
                current_line = [(text, bbox)]
                current_y = y
        
        if current_line:
            current_line.sort(key=lambda x: min(point[0] for point in x[1]))
            lines.append([text for text, _ in current_line])
        
        # Join lines with proper spacing
        structured_text = "\n".join([" ".join(line) for line in lines])
        return extracted_info, structured_text
    
    return extracted_info, text

def format_billing_output(text, extracted_info, structured_text=None):
    formatted_output = "=== Billing Information ===\n\n"
    
    # Add extracted structured information
    for key, value in extracted_info.items():
        formatted_output += f"{key.replace('_', ' ').title()}: {value}\n"
    
    formatted_output += "\n=== Structured Text ===\n"
    formatted_output += structured_text if structured_text else text
    
    return formatted_output

# -------------------- Ensemble Strategy --------------------
def ensemble_ocr(image_np, image_pil, reader, ocr_method="general"):
    gemini_text = gemini_ocr(image_pil)
    easy_results = None

    if reader is not None:
        easy_results = reader.readtext(
            image_np,
            detail=1,
            paragraph=False,
            batch_size=8,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.4,
            canvas_size=2560,
            mag_ratio=3.0
        )
        easy_text, _ = easyocr_ocr(image_np, reader)
    else:
        easy_text = ""

    gemini_clean = clean_text(gemini_text)
    easy_clean = clean_text(easy_text)

    if easy_text == "":
        final = gemini_text.strip() or "No text detected."
    else:
        if len(gemini_clean) > len(easy_clean) * 0.6:
            final = gemini_text.strip() or easy_text.strip()
        elif len(easy_clean) > 0:
            final = easy_text.strip()
        else:
            final = gemini_text.strip() or easy_text.strip() or "No text detected."

    if ocr_method == "billing":
        extracted_info, structured_text = extract_billing_info(final, easy_results)
        final = format_billing_output(final, extracted_info, structured_text)

    return final

# -------------------- Chat Interface --------------------
def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

def display_chat_interface(extracted_text):
    st.markdown("---")
    st.markdown("### 💬 Chat with Document")
    st.markdown("Ask questions about the extracted text from your document.")
    
    # Initialize chat history
    initialize_chat_history()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history
        add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create context-aware prompt
                    context_prompt = f"""Context from document:
                    {extracted_text}
                    
                    Question: {prompt}
                    
                    Please provide a helpful answer based on the document context. If the answer cannot be found in the context, say so."""
                    
                    # Get response from Gemini
                    model = load_chat_model()
                    response = model.generate_content(context_prompt)
                    
                    # Display response
                    st.write(response.text)
                    
                    # Add assistant message to chat history
                    add_message("assistant", response.text)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    add_message("assistant", "I apologize, but I encountered an error while processing your question. Please try again.")

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Reliable OCR", layout="wide")
st.title("📑 Reliable OCR - Optimized EasyOCR")

# Add OCR method selector in sidebar
ocr_method = st.sidebar.radio(
    "Select OCR Method",
    ["General OCR", "Billing OCR"],
    help="Choose between general text extraction or specialized billing document processing"
)

st.write("Upload an image or PDF. Automatically combines results and gives the best single result.")

LANGUAGES = {
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Nepali': 'ne',
    'Urdu': 'ur',
    'English': 'en'
}

SUPPORTED_EASYOCR_LANGS = {'hi', 'bn', 'ta', 'te', 'kn', 'ne', 'ur', 'en'}

selected_langs = st.sidebar.multiselect("EasyOCR Indian Languages", list(LANGUAGES.keys()), default=["English"])
lang_codes = [LANGUAGES[lang] for lang in selected_langs]

all_supported = all(code in SUPPORTED_EASYOCR_LANGS for code in lang_codes)

if all_supported:
    reader = load_easyocr_reader(lang_codes)
else:
    reader = None

uploaded_file = st.file_uploader("Upload image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    # Feature controls below uploader
    st.markdown("### Preprocessing Features")
    with st.form(f"preprocessing_form"):
        invert = st.checkbox("Invert Image (swap background/text)")
        remove_noise = st.checkbox("Remove Noise (Median Blur)")
        dilate = st.checkbox("Dilation (expand text)")
        erode = st.checkbox("Erosion (thin text)")
        auto_deskew = st.checkbox("Auto Deskew (Align Text)")
        remove_borders = st.checkbox("Remove Borders (crop to content)")
        add_borders = st.checkbox("Add Missing Borders (pad)")
        show_steps = st.checkbox("Show Preprocessing Steps")
        submitted = st.form_submit_button("Apply Features")
    config = {
        'invert': invert,
        'remove_noise': remove_noise,
        'dilate': dilate,
        'erode': erode,
        'auto_deskew': auto_deskew,
        'remove_borders': remove_borders,
        'add_borders': add_borders
    }
    if uploaded_file.type == "application/pdf":
        images = process_pdf(uploaded_file)
    else:
        images = [process_image(Image.open(uploaded_file))]

    for i, img_np in enumerate(images):
        st.subheader(f"Page {i+1} Result")
        st.image(img_np, caption=f"Input Image - Page {i+1}", use_container_width=True)
        proc_img, steps = enhance_image_for_ocr(img_np, i, config)
        image_pil = Image.fromarray(proc_img)
        if show_steps:
            st.markdown(f"#### Preprocessing Steps (Page {i+1})")
            for step_name, step_img in steps.items():
                st.image(step_img, caption=step_name, channels="GRAY")
        
        with st.spinner("Processing..."):
            final_text = ensemble_ocr(proc_img, image_pil, reader, ocr_method.lower().replace(" ", "_"))
            st.text_area(f"OCR Output (Page {i+1})", final_text, height=200)
            
            # Add chat interface after OCR results
            display_chat_interface(final_text)
