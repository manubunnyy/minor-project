# 📑 Reliable OCR - Advanced Document Processing System

## 🌟 Overview
Reliable OCR is a powerful document processing system that combines multiple OCR engines and AI capabilities to provide accurate text extraction from images and PDFs. The system features advanced preprocessing options, multi-language support, and an interactive chat interface for document analysis.

## ✨ Features
- 🔄 **Multi-Engine OCR**: Combines EasyOCR and Gemini Vision for optimal results
- 🌐 **Multi-Language Support**: Supports multiple Indian languages including Hindi, Bengali, Tamil, and more
- 🖼️ **Advanced Image Preprocessing**:
  - Image inversion
  - Noise removal
  - Dilation and erosion
  - Auto deskew
  - Border management
  - CLAHE enhancement
  - Otsu binarization
- 📄 **PDF Support**: Process multi-page PDF documents
- 💬 **Interactive Chat Interface**: Ask questions about extracted text
- 🎯 **Specialized Billing OCR**: Optimized for processing billing documents
- 🚀 **GPU Acceleration**: Automatic GPU utilization when available

## 🛠️ Technical Stack
- **Frontend**: Streamlit
- **OCR Engines**: 
  - EasyOCR
  - Google Gemini Vision
- **Image Processing**: OpenCV
- **PDF Processing**: PyMuPDF
- **AI/ML**: Google Gemini AI
- **Language Support**: Multiple Indian languages

## 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for better performance)
- Google Gemini API key

## 🚀 Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/reliable-ocr.git
cd reliable-ocr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

## 💻 Usage
1. Start the application:
```bash
streamlit run revised.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your document (image or PDF)

4. Configure preprocessing options as needed

5. View the extracted text and interact with the chat interface

## 🔧 Configuration
The system can be configured through the Streamlit interface:
- Select OCR method (General or Billing)
- Choose supported languages
- Adjust preprocessing parameters
- Configure chat interface settings

## 📊 Architecture
The system follows a modular architecture:
- Frontend Layer (Streamlit)
- Processing Layer (Image/PDF processing)
- OCR Engine Layer (EasyOCR + Gemini)
- AI Layer (Gemini for chat)

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- EasyOCR team for the OCR engine
- Google for Gemini AI capabilities
- Ensemble approach using both
- Streamlit for the web interface framework

## 📞 Support
For support, please open an issue in the GitHub repository or contact the maintainers.

---
Made with ❤️ by [@manubunnyy]
