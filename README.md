# PolyOCR - Advanced OCR Pipeline with BERT Autocorrection

A comprehensive OCR pipeline that combines EasyOCR with BERT-based autocorrection and provides a modern web interface for text extraction and correction.

## Project Structure

```
📁 Project Root/
├── 📄 ocr_pipeline.py            # Main OCR pipeline script
├── 📄 analysis.py                # OCR accuracy analysis tool
├── 📄 run.py                     # FastAPI server startup script
├── 📄 config.py                  # Project configuration
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # This file
│
├── 📁 modules/                   # Core OCR modules
│   ├── 📄 ocr_engine.py              # EasyOCR engine wrapper
│   ├── 📄 bert_autocorrector.py        # BERT-based text correction
│   ├── 📄 bert_language_detector_fix.py # Language detection
│   └── 📄 image_preprocessing.py       # (Currently unused) Image preprocessing
│
├── 📁 PolyOCR/                   # Web application
│   ├── 📄 app.py                # FastAPI application
│   ├── 📄 main.py               # Alternative server entry point
│   ├── 📁 templates/            # HTML templates
│   │   └── 📄 index.html        # Main web interface
│   └── 📁 static/               # Static assets (CSS/JS)
│       ├── 📄 styles.css        # Main stylesheet
│       └── 📄 app.js            # JavaScript functionality
│
├── 📁 saved/                     # Saved models
│   └── 📄 language_detector_model.pkl
│
└── 📄 ocr_results.csv           # OCR pipeline results
```

## Core Components

### 1. OCR Pipeline (`ocr_pipeline.py`)
- Processes images using EasyOCR with GPU acceleration
- Applies BERT-based autocorrection
- Outputs results to CSV format

### 2. BERT Autocorrector (`modules/bert_autocorrector.py`)
- Uses multilingual BERT for context-aware text correction
- Implements multiple correction strategies
- Handles common OCR errors (character confusions, spacing, etc.)

### 3. Web Interface (`PolyOCR/`)
- Modern FastAPI-based web application
- Image upload and real-time OCR processing
- Language detection and correction visualization
- Adjustable confidence thresholds

### 4. Analysis Tools (`analysis.py`)
- Comprehensive accuracy analysis
- BERT correction effectiveness measurement
- Quality scoring and recommendations

## Current Performance

Based on the latest analysis of the dataset:

- **OCR Detection Rate**: Varies based on `OCR_CONFIDENCE_THRESHOLD`
- **BERT Correction Rate**: Varies based on `BERT_CONFIDENCE_THRESHOLD`
- **Overall Quality Score**: Check the output of `analysis.py` for the latest score.

## Key Features

✅ **GPU-Accelerated OCR** - Uses EasyOCR with CUDA support  
✅ **BERT Autocorrection** - Context-aware text correction  
✅ **Language Detection** - Automatic language identification  
✅ **Web Interface** - Modern, responsive UI  
✅ **Batch Processing** - Handle large image datasets  
✅ **Accuracy Analysis** - Comprehensive performance metrics  

## Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run OCR Pipeline**:
   ```bash
   python ocr_pipeline.py
   ```

3. **Start Web Server**:
   ```bash
   python run.py
   ```
   Then open http://127.0.0.1:8000 in your browser

4. **Analyze Results**:
   ```bash
   python analysis.py
   ```

## Configuration (`config.py`)

- **OCR Confidence**: Adjustable threshold for text detection
- **BERT Confidence**: Adjustable threshold for autocorrection
- **GPU Support**: Automatic CUDA detection and usage
- **Language Support**: Multilingual OCR and correction

## Data Files

- `ocr_results.csv` - Pipeline output with original and corrected text

## Notes

- Project optimized for GPU processing (if available)
- Default BERT model: `bert-base-multilingual-cased`
- Supports 50+ languages for OCR and correction
- Web interface includes correction visualization and confidence controls
