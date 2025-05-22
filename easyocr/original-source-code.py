import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import os
import fitz  # PyMuPDF for PDF processing

def get_text_position(bbox):
    # Get the top-left point of the bounding box
    top_left = min(bbox, key=lambda p: (p[1], p[0]))  # First by y (top), then by x (left)
    return (top_left[1], top_left[0])  # Return (y, x) for sorting

def sort_by_position(ocr_results):
    # Sort results by position (top-to-bottom, then left-to-right)
    return sorted(ocr_results, key=lambda x: get_text_position(x[0]))

OCR_CONFIDENCE_THRESHOLD = 0.3

def filter_ocr_results(ocr_results, threshold=OCR_CONFIDENCE_THRESHOLD):
    return [res for res in ocr_results if res[2] >= threshold]

def enhance_image_for_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply bilateral filter to preserve edges while removing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(cleaned, None, 10, 7, 21)
    
    # Convert back to RGB
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

def process_pdf(pdf_file):
    # Open the PDF
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    
    # Convert each page to an image
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    
    return images

def process_image(image):
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is in RGB format
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image

# Set page config
st.set_page_config(
    page_title="EasyOCR Text Extraction",
    page_icon="📝",
    layout="wide"
)

# Title
st.title("📝 EasyOCR Text Extraction")
st.write("Upload an image to extract text")

# Available languages
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'French': 'fr',
    'German': 'de',
    'Spanish': 'es',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese (Simplified)': 'ch_sim',
    'Chinese (Traditional)': 'ch_tra',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Thai': 'th',
    'Vietnamese': 'vi',
    'Arabic': 'ar',
    'Bengali': 'bn',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Nepali': 'ne',
    'Urdu': 'ur'
}

# Language selection in sidebar
st.sidebar.title("Language Selection")
selected_languages = st.sidebar.multiselect(
    "Select languages for OCR",
    options=list(LANGUAGES.keys()),
    default=['English']
)

# Convert selected language names to language codes
language_codes = [LANGUAGES[lang] for lang in selected_languages]

# Always put 'en' first if present
if 'en' in language_codes:
    language_codes = ['en'] + [code for code in language_codes if code != 'en']

def build_allowlist(lang_codes):
    allowlist = set()
    char_dir = os.path.join(os.path.dirname(__file__), 'easyocr', 'character')
    
    # Add basic Latin characters and common symbols for all languages
    allowlist.update('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    allowlist.update('.,!?@#$%^&*()_+-=[]{}|;:"\'<>/')
    
    # Add characters for each selected language
    for code in lang_codes:
        char_file = os.path.join(char_dir, f'{code}_char.txt')
        if os.path.exists(char_file):
            with open(char_file, encoding='utf-8-sig') as f:
                chars = f.read().splitlines()
                allowlist.update(''.join(chars))
    
    return ''.join(sorted(allowlist))

# Initialize EasyOCR reader with selected languages
@st.cache_resource
def load_reader(langs):
    return easyocr.Reader(langs)

if language_codes:
    reader = load_reader(language_codes)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        # Process the uploaded file
        if uploaded_file.type == "application/pdf":
            images = process_pdf(uploaded_file)
            if not images:
                st.error("No pages found in the PDF")
                st.stop()
            # Use the first page for now
            image = images[0]
            if len(images) > 1:
                st.info(f"Processing first page of {len(images)} pages")
        else:
            # Process image file
            image = Image.open(uploaded_file)
            image = process_image(image)
        
        # Display original image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Add preprocessing options
        st.subheader("Preprocessing Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Grayscale conversion
            grayscale = st.checkbox("Convert to Grayscale", value=True)
            
            # Contrast adjustment
            contrast = st.slider("Contrast Adjustment", 0.0, 2.0, 1.0, 0.1)
            
        with col2:
            # Brightness adjustment
            brightness = st.slider("Brightness Adjustment", -100, 100, 0, 10)
            
            # Noise reduction
            denoise = st.checkbox("Apply Noise Reduction", value=True)
        
        # Add a button to process the image
        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                # Apply preprocessing
                processed_image = image.copy()
                
                # Apply OCR-specific enhancement
                processed_image = enhance_image_for_ocr(processed_image)
                
                # Apply user-selected preprocessing
                if grayscale:
                    if len(processed_image.shape) == 3:
                        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                
                # Apply contrast
                processed_image = cv2.convertScaleAbs(processed_image, alpha=contrast, beta=brightness)
                
                # Apply denoising
                if denoise:
                    processed_image = cv2.fastNlMeansDenoisingColored(processed_image, None, 10, 10, 7, 21)
                
                # Perform OCR with preprocessing
                allowlist = None
                if len(language_codes) >= 1:
                    allowlist = build_allowlist(language_codes)
                
                # Configure OCR parameters
                results = reader.readtext(
                    processed_image,
                    allowlist=allowlist,
                    detail=1,
                    paragraph=False,
                    batch_size=8,
                    contrast_ths=0.1,
                    adjust_contrast=0.5,
                    text_threshold=0.6,
                    link_threshold=0.4,
                    low_text=0.3,
                    canvas_size=2560,
                    mag_ratio=2.0,
                    slope_ths=0.2,
                    ycenter_ths=0.5,
                    height_ths=0.5,
                    width_ths=0.5,
                    add_margin=0.1
                )
                
                # Extract and display text
                st.subheader("Extracted Text:")
                # Get all text with confidence scores, then filter and extract text
                filtered_results = filter_ocr_results(results, threshold=OCR_CONFIDENCE_THRESHOLD)
                # Sort results by position
                sorted_results = sort_by_position(filtered_results)
                # Extract text in order
                text_results = [text for (bbox, text, prob) in sorted_results]
                ocr_text = ' '.join(text_results)
                # Display full text
                st.text_area("Complete Text", ocr_text, height=100)
                
                # Display confidence scores
                st.subheader("Text with Confidence Scores:")
                for bbox, text, prob in sorted_results:
                    st.write(f"Text: {text} (Confidence: {prob:.2f})")

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info(
    """
    This is an enhanced GUI for EasyOCR text extraction.
    Features:
    - Support for multiple languages
    - Image preprocessing options
    - Text extraction with confidence scores
    - PDF support
    """
) 