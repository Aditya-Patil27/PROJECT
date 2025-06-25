 #!/usr/bin/env python3
"""
OCR Pipeline Configuration
Adjust these parameters to optimize performance
"""

# Batch Processing Settings
BATCH_SIZE = 20  # Number of batches to create
PARALLEL_PROCESSING = False  # Enable parallel processing (experimental)

# OCR Settings
OCR_CONFIDENCE_THRESHOLD = 0.25  # Lower = more text detected (0.1-0.9)
OCR_LANGUAGES = ['en']  # Languages to detect
USE_GPU = True  # Use GPU acceleration if available

# BERT Autocorrection Settings
BERT_CONFIDENCE_THRESHOLD = 0.5  # Lower = more corrections (0.1-0.9)
BERT_MODEL = "bert-base-multilingual-cased"  # BERT model to use

# Image Enhancement Settings
ENABLE_IMAGE_ENHANCEMENT = True
ENHANCEMENT_METHODS = ['contrast', 'sharpen']  # Available: contrast, sharpen, denoise, resize

# Performance Settings
CLEAR_GPU_CACHE = True  # Clear GPU cache between batches
SAVE_BATCH_FILES = True  # Save individual batch files (for recovery)
CLEANUP_BATCH_FILES = True  # Remove batch files after completion

# Output Settings
OUTPUT_FILE = 'ocr_results.csv'
SHOW_PROGRESS = True
VERBOSE_OUTPUT = True

# Advanced Settings
MAX_WORKERS = 4  # For parallel processing
TIMEOUT_PER_IMAGE = 30  # Seconds before timing out on an image
RETRY_FAILED_IMAGES = False  # Retry images that failed processing

# Quality Improvements Based on Analysis
QUALITY_IMPROVEMENTS = {
    'lower_ocr_threshold': True,     # Detect more text
    'aggressive_bert_correction': True,  # More corrections
    'enhanced_preprocessing': True,   # Better image processing
    'character_error_correction': True,  # Fix common OCR errors
    'context_aware_correction': True,    # Use context for corrections
}

# Logging Settings
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = False
LOG_FILE = 'ocr_pipeline.log'
