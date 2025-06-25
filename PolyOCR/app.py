import sys
import os

# Add project root to Python path to allow module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple
import numpy as np
import cv2
import re
import warnings
from modules.ocr_engine import initialize_ocr, run_ocr, extract_all_text, get_text_statistics
from modules.bert_language_detector_fix import detect_language, BERTLanguageDetector
from modules.bert_autocorrector import BertAutocorrector
import logging
import base64
import io
from PIL import Image

# Filter scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Helper to convert numpy types to native Python types for JSON serialization
def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_types(i) for i in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data

# Simple heuristic-based language detection as backup
def detect_language_heuristic(text: str) -> Tuple[str, float]:
    """
    Detect language using simple heuristics as a fallback
    
    Args:
        text: Input text string
        
    Returns:
        Tuple of (language_code, confidence)
    """
    if not text or len(text.strip()) < 5:
        return "en", 0.5  # Default to English with low confidence for very short text
    
    # Define character sets and patterns for different languages
    patterns = {
        # Latin script languages
        "en": (r'[a-zA-Z]', r'\b(the|and|is|in|to|of|a|for|that|this|with)\b'),  # English
        "fr": (r'[a-zA-Z]', r'\b(le|la|les|des|et|en|un|une|dans|pour|ce|cette|ces)\b'),  # French
        "es": (r'[a-zA-Z]', r'\b(el|la|los|las|de|en|un|una|y|o|que|por|para|con)\b'),  # Spanish
        "de": (r'[a-zA-Z]', r'\b(der|die|das|und|in|zu|den|dem|ein|eine|mit|für)\b'),  # German
        
        # Non-Latin scripts
        "ru": (r'[а-яА-Я]', None),  # Russian (Cyrillic)
        "zh": (r'[\u4e00-\u9fff]', None),  # Chinese
        "ja": (r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', None),  # Japanese
        "ko": (r'[\uac00-\ud7af]', None),  # Korean
        "ar": (r'[\u0600-\u06ff]', None),  # Arabic
        "hi": (r'[\u0900-\u097f]', None),  # Hindi
        "th": (r'[\u0e00-\u0e7f]', None),  # Thai
    }
    
    # Count script occurrences
    script_counts = {}
    for lang, (script_pattern, _) in patterns.items():
        if script_pattern:
            matches = re.findall(script_pattern, text)
            script_counts[lang] = len(matches)
    
    # Check for common words if we have Latin script
    word_counts = {}
    latin_script_langs = ["en", "fr", "es", "de"]
    has_latin = any(script_counts.get(lang, 0) > 0 for lang in latin_script_langs)
    
    if has_latin:
        for lang, (_, word_pattern) in patterns.items():
            if word_pattern:
                matches = re.findall(word_pattern, text.lower())
                word_counts[lang] = len(matches)
    
    # Determine language based on script and words
    if not script_counts:
        return "en", 0.3  # Default to English with low confidence
    
    # For non-Latin scripts, use the script with most matches
    non_latin = {k: v for k, v in script_counts.items() if k not in latin_script_langs}
    if non_latin:
        best_lang = max(non_latin.items(), key=lambda x: x[1])[0]
        total = sum(non_latin.values())
        confidence = non_latin[best_lang] / total if total > 0 else 0.5
        return best_lang, min(0.9, confidence)  # Cap at 0.9 confidence
    
    # For Latin script languages, use word matches
    if word_counts:
        best_lang = max(word_counts.items(), key=lambda x: x[1])[0]
        total = sum(word_counts.values())
        confidence = word_counts[best_lang] / total if total > 0 else 0.5
        return best_lang, min(0.8, confidence)  # Cap at 0.8 confidence
    
    # If we have Latin script but no word matches, use script with most matches
    best_lang = max(script_counts.items(), key=lambda x: x[1])[0]
    return best_lang, 0.4  # Lower confidence without word evidence

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Supported languages for demo (can be extended)
SUPPORTED_LANGS = [
    "en", "fr", "de", "es", "nl", "pl", "sl", "cz", "bg", "fi"
]

# OCR preprocessing levels
PREPROCESSING_LEVELS = ["light", "medium", "aggressive"]

# Initialize BERT language detector model
try:
    # Create language detector with XLM-RoBERTa model
    language_detector = BERTLanguageDetector(model_name="papluca/xlm-roberta-base-language-detection")
    print(f"✅ Language detector loaded: {language_detector.model_name}")
    if hasattr(language_detector, "supported_languages"):
        print(f"Languages: {sorted(language_detector.supported_languages)}")
except Exception as e:
    logger.error(f"Failed to load BERT language detector: {e}")
    language_detector = None
    print(f"⚠️ Warning: Failed to load BERT language detector: {e}")

# Initialize BERT autocorrector
try:
    bert_autocorrector = BertAutocorrector()
    print(f"✅ BERT autocorrector initialized")
except Exception as e:
    logger.error(f"Failed to load BERT autocorrector: {e}")
    bert_autocorrector = None
    print(f"⚠️ Warning: Failed to load BERT autocorrector: {e}")

def get_ocr_engine(languages: List[str]):
    # Use GPU for OCR if available
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("CUDA is available! Using GPU for OCR processing.")
        else:
            logger.info("CUDA not available. Using CPU for OCR processing.")
    except:
        use_gpu = False
        logger.info("PyTorch not found or error checking GPU. Defaulting to CPU.")
    
    return initialize_ocr(languages=languages, use_gpu=use_gpu)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Use direct Jinja2 templating
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "languages": SUPPORTED_LANGS,
            "preprocessing_levels": PREPROCESSING_LEVELS
        }
    )

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.3),
    correction_threshold: float = Form(0.7)
):
    """
    Analyze a single image with OCR, language detection, and BERT autocorrection.
    Returns text extraction, detected language, and autocorrected text.
    """
    logger.info(f"Analyzing image: {file.filename} with OCR confidence: {confidence_threshold}, correction strength: {correction_threshold}")
    
    try:
        # Read uploaded file
        file_contents = await file.read()
        if not file_contents:
            return JSONResponse({"error": "Empty file uploaded"}, status_code=400)
        
        # Convert to PIL Image and then to OpenCV format
        pil_image = Image.open(io.BytesIO(file_contents))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Initialize OCR engine (auto-detect GPU)
        ocr_engine = get_ocr_engine(['en'])  # Start with English, can be extended
        
        # Run OCR
        ocr_blocks = run_ocr(
            ocr_engine, 
            image, 
            confidence_threshold=confidence_threshold,
            preprocessing_level='none',  # Use raw image for best results
            post_process=True
        )
        
        # Extract text
        extracted_text = extract_all_text(ocr_blocks, min_confidence=confidence_threshold)
        
        # Detect language
        detected_language = "unknown"
        language_confidence = 0.0
        
        if extracted_text.strip():
            try:
                if language_detector and language_detector.is_initialized:
                    detected_language, language_confidence = language_detector.predict(extracted_text)
                else:
                    detected_language, language_confidence = detect_language_heuristic(extracted_text)
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
                detected_language, language_confidence = detect_language_heuristic(extracted_text)
        
        # Apply BERT autocorrection with custom threshold
        corrected_text = extracted_text
        if bert_autocorrector and extracted_text.strip():
            try:
                # Use a lower threshold for more aggressive correction on OCR text
                effective_threshold = max(0.3, correction_threshold * 0.8)
                corrected_text = bert_autocorrector.autocorrect(extracted_text, effective_threshold)
                logger.info(f"BERT correction applied with threshold {effective_threshold}")
                
                # Log if any corrections were made
                if corrected_text != extracted_text:
                    logger.info(f"✅ Text corrected: '{extracted_text[:50]}...' → '{corrected_text[:50]}...'")
                else:
                    logger.info("ℹ️ No corrections needed")
                    
            except Exception as e:
                logger.warning(f"BERT autocorrection failed: {e}")
                corrected_text = extracted_text
        
        # Get statistics
        statistics = get_text_statistics(ocr_blocks)
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        # Convert to RGB if necessary (handles palette mode 'P' and other modes)
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            # Convert to RGB, handling transparency by using white background
            rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGB')
            elif pil_image.mode in ('RGBA', 'LA'):
                rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
                pil_image = rgb_image
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        pil_image.save(img_buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        result = {
            "success": True,
            "filename": file.filename,
            "original_text": extracted_text,
            "corrected_text": corrected_text,
            "detected_language": detected_language,
            "language_confidence": round(language_confidence, 3),
            "statistics": {
                "total_blocks": statistics.get("total_blocks", 0),
                "avg_confidence": round(statistics.get("avg_confidence", 0), 3),
                "total_characters": statistics.get("total_characters", 0)
            },
            "image_base64": img_base64,
            "text_blocks": [
                {
                    "text": block["text"],
                    "confidence": round(block["confidence"], 3),
                    "bbox": block["bbox"]
                }
                for block in ocr_blocks
            ]
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        logger.exception(f"Error analyzing image: {e}")
        return JSONResponse({"error": f"Failed to analyze image: {str(e)}"}, status_code=500)

@app.post("/ocr")
async def ocr_images(
    files: List[UploadFile] = File(...),
    languages: str = Form(...),
    preprocessing_level: str = Form("aggressive"),
    confidence_threshold: float = Form(0.01)  # Very low threshold for maximum text detection
):
    logger.info(f"Received OCR request for {len(files)} files and languages: {languages}")
    logger.info(f"Preprocessing level: {preprocessing_level}, Confidence threshold: {confidence_threshold}")
    
    langs = [l.strip() for l in languages.split(',') if l.strip()]
    if not langs:
        logger.warning("No languages selected.")
        return JSONResponse({"error": "No languages selected."}, status_code=400)
    
    # Validate preprocessing level
    if preprocessing_level not in PREPROCESSING_LEVELS:
        logger.warning(f"Invalid preprocessing level: {preprocessing_level}. Defaulting to 'aggressive'.")
        preprocessing_level = "aggressive"
    
    # Validate confidence threshold
    if confidence_threshold < 0 or confidence_threshold > 1:
        logger.warning(f"Invalid confidence threshold: {confidence_threshold}. Defaulting to 0.05.")
        confidence_threshold = 0.05
    
    logger.info(f"Initializing OCR engine for languages: {langs}")
    ocr_engine = get_ocr_engine(langs)
    results = []

    for file in files:
        try:
            logger.info(f"Processing file: {file.filename}")
            
            # Create a unique temp path using a timestamp to avoid collisions
            import time
            timestamp = int(time.time() * 1000)
            temp_path = f"temp_{timestamp}_{file.filename}"
            
            # Save uploaded file temporarily
            try:
                file_contents = await file.read()
                if not file_contents:
                    logger.error(f"Uploaded file {file.filename} is empty")
                    results.append({"error": f"Empty file: {file.filename}", "text": ""})
                    continue
                    
                logger.info(f"Read {len(file_contents)} bytes from uploaded file")
                with open(temp_path, "wb") as buffer:
                    buffer.write(file_contents)
                logger.info(f"File saved to temporary path: {temp_path}")
                
                # Reset file stream position for future reads if needed
                await file.seek(0)
            except Exception as e:
                logger.exception(f"Error saving uploaded file: {e}")
                results.append({"error": f"Error saving file: {str(e)}", "text": ""})
                continue

            # Read image with OpenCV
            logger.info(f"Attempting to read image from: {temp_path}")
            logger.info(f"File exists: {os.path.exists(temp_path)}, Size: {os.path.getsize(temp_path) if os.path.exists(temp_path) else 'N/A'} bytes")
            
            # Try reading with OpenCV
            image = cv2.imread(temp_path)
            
            if image is None:
                # If OpenCV fails, try with PIL
                logger.warning(f"OpenCV failed to read image. Trying with PIL...")
                try:
                    from PIL import Image
                    pil_image = Image.open(temp_path)
                    # Convert PIL image to OpenCV format
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    logger.info(f"Successfully loaded image with PIL")
                except Exception as e:
                    logger.error(f"Both OpenCV and PIL failed to read image: {str(e)}")
                    results.append({"error": f"Invalid or unsupported image file: {file.filename}", "text": ""})
                    os.remove(temp_path)
                    continue
            
            logger.info(f"Image loaded successfully. Shape: {image.shape}, Type: {type(image)}, Dtype: {image.dtype}")

            # Determine primary language for optimization if multiple are selected
            primary_language = langs[0] if langs else None
            
            # Run OCR with enhanced parameters
            logger.info(f"Running OCR with preprocessing level: {preprocessing_level}")
            ocr_blocks = run_ocr(
                ocr_engine, 
                image, 
                confidence_threshold=confidence_threshold,
                preprocessing_level=preprocessing_level,
                language_optimization=primary_language,
                post_process=True
            )
            logger.info(f"OCR completed. Found {len(ocr_blocks)} text blocks.")

            text = extract_all_text(ocr_blocks, min_confidence=confidence_threshold)
            statistics = get_text_statistics(ocr_blocks)
            logger.info(f"Extracted text length: {len(text)}")
            logger.info(f"Statistics: {statistics}")

            # Language detection with BERT model
            language, lang_conf = None, None
            try:
                if text.strip():
                    logger.info("Running BERT language detection...")
                    if language_detector and language_detector.is_initialized:
                        language, lang_conf = language_detector.predict(text)
                    else:
                        # Use the standalone function that will initialize the model if needed
                        language, lang_conf = detect_language(text)
                    logger.info(f"Detected language: {language} with confidence {lang_conf}")
            except Exception as e:
                logger.warning(f"BERT language detection failed: {e}")
                language, lang_conf = None, None
            
            if not text.strip():
                results.append({
                    "filename": file.filename,
                    "error": "No text detected in the image.",
                    "text": "",
                    "blocks": ocr_blocks,
                    "statistics": statistics,
                    "language": language,
                    "language_confidence": lang_conf,
                    "preprocessing_level": preprocessing_level
                })
            else:
                results.append({
                    "filename": file.filename,
                    "text": text,
                    "blocks": ocr_blocks,
                    "statistics": statistics,
                    "language": language,
                    "language_confidence": lang_conf,
                    "preprocessing_level": preprocessing_level
                })
        except Exception as e:
            logger.exception(f"An error occurred while processing {file.filename}: {e}")
            results.append({"error": f"An error occurred: {str(e)}", "text": ""})
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")

    logger.info(f"Returning {len(results)} results.")
    # Convert numpy types to native Python types for JSON serialization
    results = convert_numpy_types(results)
    return {"results": results}

@app.post("/download", response_class=PlainTextResponse)
async def download_results(text: str = Form(...)):
    return PlainTextResponse(text, headers={
        'Content-Disposition': 'attachment; filename="polyocr_results.txt"'
    })

@app.get("/test-ocr")
def test_ocr():
    sample_images = [
        os.path.join(os.path.dirname(__file__), '../assets/images_examples/images.png'),
        os.path.join(os.path.dirname(__file__), '../assets/images_examples/download.jpeg')
    ]
    langs = ["en"]  # Default to English for test
    ocr_engine = get_ocr_engine(langs)
    results = []
    for img_path in sample_images:
        if not os.path.exists(img_path):
            results.append({"filename": os.path.basename(img_path), "error": "File not found."})
            continue
        image = cv2.imread(img_path)
        if image is None:
            results.append({"filename": os.path.basename(img_path), "error": "Could not read image."})
            continue
        ocr_blocks = run_ocr(
            ocr_engine, 
            image, 
            confidence_threshold=0.05,
            preprocessing_level='aggressive'
        )
        text = extract_all_text(ocr_blocks, min_confidence=0.05)
        statistics = get_text_statistics(ocr_blocks)
        results.append({
            "filename": os.path.basename(img_path),
            "text": text,
            "blocks": ocr_blocks,
            "statistics": statistics
        })
    results = convert_numpy_types(results)
    return {"results": results}
