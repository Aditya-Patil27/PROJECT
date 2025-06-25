import os
import cv2
import re
from typing import List, Union, Dict, Optional, Any, Callable
import logging
import numpy as np

# Fix for EasyOCR compatibility with Pillow 10+
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import easyocr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("../logs", exist_ok=True)
handler = logging.FileHandler("../logs/ocr_engine.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class OCREngine:
    """Singleton OCR Engine using EasyOCR"""
    _instance = None

    def __new__(cls, langs: List[str] = None, use_gpu: bool = True):
        if cls._instance is None:
            cls._instance = super(OCREngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, langs: List[str] = None, use_gpu: bool = True):
        if not self.initialized:
            if langs is None:
                langs = ['en']  # Default to English
            try:
                # Check if CUDA is available when GPU is requested
                gpu_available = False
                if use_gpu:
                    try:
                        import torch
                        gpu_available = torch.cuda.is_available()
                        if not gpu_available:
                            logger.warning("CUDA is not available even though GPU was requested. Will use CPU instead.")
                            logger.warning("To use GPU, ensure you have a CUDA-compatible PyTorch version and proper drivers.")
                            logger.warning("See GPU_SETUP.md for instructions on setting up GPU support.")
                    except ImportError:
                        logger.warning("Unable to check CUDA availability - torch not found")
                
                # Initialize EasyOCR engine using the top-level import
                self.reader = easyocr.Reader(
                    lang_list=langs, 
                    gpu=use_gpu,  # We'll still request GPU even if unavailable - EasyOCR will handle fallback
                    detector=True,  # Use default text detector
                    recognizer=True,  # Use default recognition network
                    download_enabled=True  # Allow downloading required models
                )
                
                # Store configuration
                self.langs = langs
                self.use_gpu = use_gpu and gpu_available
                self.initialized = True
                logger.info(f"OCREngine initialized with languages: {langs}, Requested GPU: {use_gpu}, Actual GPU usage: {self.use_gpu}")
                
                # Log device information if available
                try:
                    if hasattr(self.reader, 'detector') and hasattr(self.reader.detector, 'device'):
                        logger.info(f"EasyOCR detector device: {self.reader.detector.device}")
                    if hasattr(self.reader, 'recognizer') and hasattr(self.reader.recognizer, 'device'):
                        logger.info(f"EasyOCR recognizer device: {self.reader.recognizer.device}")
                except:
                    pass
            except ImportError:
                logger.error("EasyOCR not installed. Please install it with: pip install easyocr")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize OCR engine: {e}")
                raise
    
    def get_supported_languages(self) -> List[str]:
        """Return list of currently loaded languages"""
        return self.langs if hasattr(self, 'langs') else []
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU is being used"""
        return self.use_gpu if hasattr(self, 'use_gpu') else False

def initialize_ocr(
        languages: List[str] = None,
        use_gpu: bool = True
) -> OCREngine:
    """Initialize OCR engine with specified languages and GPU setting"""
    if languages is None:
        languages = ['en']  # Default to English
    
    try:
        return OCREngine(langs=languages, use_gpu=use_gpu)
    except Exception as e:
        logger.error(f"Failed to initialize OCR: {e}")
        raise

def preprocess_for_detection(image):
    """
    Special preprocessing just for text detection phase
    This helps find text regions more accurately
    """
    processed = image.copy()
    if len(processed.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(blurred)
        
        return contrast
    return processed

def run_ocr(
        engine: OCREngine,
        image_path: Union[str, np.ndarray],
        confidence_threshold: float = 0.1,  # Lower threshold for more results
        enhance_image: bool = False,  # Default to no enhancement - use raw images for better OCR accuracy
        enhancement_methods: Optional[List[str]] = None,
        language_optimization: Optional[str] = None,
        preprocessing_level: str = 'none',  # Default to no preprocessing
        post_process: bool = True
) -> List[Dict]:
    """
    Run OCR on a single image with improved accuracy
    
    Args:
        engine: OCREngine instance
        image_path: Path to image file or numpy array
        confidence_threshold: Minimum confidence threshold for text detection (lower = more results)
        enhance_image: Whether to apply image enhancement
        enhancement_methods: List of enhancement methods to apply
        language_optimization: Language code to optimize for
        preprocessing_level: Level of preprocessing: 'light', 'medium', or 'aggressive'
        post_process: Whether to apply post-processing to improve results
    
    Returns:
        List of dictionaries containing detected text blocks with bbox, text, and confidence
    """
    try:
        # Validate engine
        if not hasattr(engine, 'reader') or not engine.initialized:
            logger.error("OCR engine not properly initialized")
            return []
        
        # Load image
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                logger.error(f"Image path does not exist: {image_path}")
                return []
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return []
        else:
            image = image_path
          # Store original image for visualization
        original_image = image.copy()
        
        # No image enhancement - use the loaded image directly
        enhanced_image = image
            
        # Run OCR
        logger.info(f"Running OCR on image with dimensions: {enhanced_image.shape}")
        logger.info(f"Using GPU: {engine.is_gpu_enabled()}")
        
        # Use standard OCR approach with enhanced image
        try:
            # Optimized parameters for maximum accuracy
            raw_results = engine.reader.readtext(
                enhanced_image, 
                detail=1,  # Include position and confidence
                paragraph=False,  # Treat each text element separately
                width_ths=0.9,      # More strict width threshold for better accuracy
                height_ths=0.9,     # More strict height threshold for better accuracy
                text_threshold=0.5,  # Higher threshold for better accuracy
                low_text=0.3,       # Higher threshold for better accuracy
                link_threshold=0.2, # Higher threshold for connecting text
                canvas_size=3000,   # Larger canvas for better processing                
                mag_ratio=1.2       # Slight magnification for better recognition
            )
        except Exception as e:
            logger.warning(f"Enhanced OCR failed: {e}, falling back to basic OCR")
            try:
                # Fallback with balanced parameters for more results
                raw_results = engine.reader.readtext(
                    enhanced_image, 
                    detail=1, 
                    text_threshold=0.4,  # Moderate threshold
                    low_text=0.25,      # Moderate threshold
                    link_threshold=0.15, # Moderate threshold
                    canvas_size=2560,
                    mag_ratio=1.1
                )
            except Exception as e2:
                logger.error(f"Basic OCR also failed: {e2}, using most minimal parameters")
                raw_results = engine.reader.readtext(enhanced_image, detail=1)
        
        # Process results
        blocks = []
        for detection in raw_results:
            # Handle different return formats
            if len(detection) == 3:
                box, text, conf = detection
            else:
                # Handle other formats
                continue
                
            if conf >= confidence_threshold:  # Filter by confidence
                logger.info(f"Detected text: '{text}' with confidence: {conf:.3f}")
                blocks.append({
                    "bbox": [[int(x), int(y)] for x, y in box],
                    "text": text.strip(),
                    "confidence": float(conf)
                })
            else:
                logger.debug(f"Filtered out low confidence text: '{text}' (conf: {conf:.3f})")
        
        # Apply post-processing if requested
        if post_process and blocks:
            blocks = post_process_ocr_results(blocks)
        
        logger.info(f"OCR completed. Found {len(blocks)} text blocks above threshold {confidence_threshold}")
        return blocks
        
    except Exception as e:
        logger.exception(f"OCR failed: {e}")
        return []

def post_process_ocr_results(ocr_blocks: List[Dict]) -> List[Dict]:
    """
    Apply post-processing to OCR results to improve quality
    
    Args:
        ocr_blocks: List of OCR result blocks
        
    Returns:
        List of improved OCR blocks
    """
    improved_blocks = []
    
    for block in ocr_blocks:
        # Get the original values
        text = block["text"]
        confidence = block["confidence"]
        
        # 1. Fix common OCR errors
        improved_text = text
        
        # 1a. Fix common character confusions
        improved_text = improved_text.replace('0', 'O') if improved_text.isalpha() and '0' in improved_text else improved_text
        improved_text = improved_text.replace('1', 'I') if improved_text.isalpha() and '1' in improved_text else improved_text
        
        # 1b. Fix spaces in words
        if len(improved_text) > 3:
            # Merge words split by spaces if they appear to be a single word
            words = improved_text.split()
            if len(words) > 1:
                joined_candidate = ''.join(words)
                if joined_candidate.isalpha() and len(joined_candidate) < 15:  # Reasonable word length
                    improved_text = joined_candidate
        
        # 1c. Remove non-alphanumeric characters at the start/end
        improved_text = re.sub(r'^[^\w]+', '', improved_text)
        improved_text = re.sub(r'[^\w]+$', '', improved_text)
        
        # 2. Fix case if needed
        if improved_text.isupper() and len(improved_text) > 10:
            # Long text shouldn't be all caps unless it's an acronym
            improved_text = improved_text.capitalize()
        
        # Only add blocks that have text after processing
        if improved_text.strip():
            improved_block = block.copy()
            improved_block["text"] = improved_text
            improved_block["original_text"] = text  # Keep original for reference
            improved_blocks.append(improved_block)
    
    return improved_blocks

def run_batch_ocr(
        engine: OCREngine,
        image_paths: List[Union[str, np.ndarray]],
        confidence_threshold: float = 0.1,
        preprocessing_level: str = 'none',  # Default to no preprocessing - use raw images
        language_optimization: Optional[str] = None,
        save_results: bool = False,
        output_dir: str = None
) -> List[List[Dict]]:
    """
    Run OCR on multiple images
    
    Args:
        engine: OCREngine instance
        image_paths: List of image paths or numpy arrays
        confidence_threshold: Minimum confidence threshold for text detection
        preprocessing_level: Level of preprocessing
        language_optimization: Language code to optimize for
        save_results: Whether to save results to files
        output_dir: Directory to save results (if save_results=True)
    
    Returns:
        List of lists containing OCR results for each image
    """
    results = []
    
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing image {i+1}/{len(image_paths)}")
        result = run_ocr(
            engine, 
            image_path, 
            confidence_threshold=confidence_threshold,
            preprocessing_level=preprocessing_level,
            language_optimization=language_optimization,
            post_process=True
        )
        results.append(result)
        
        # Save individual results if requested
        if save_results and output_dir and result:
            filename = f"ocr_result_{i+1}.txt" if isinstance(image_path, np.ndarray) else f"ocr_result_{os.path.basename(image_path)}.txt"
            result_path = os.path.join(output_dir, filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"OCR Results for: {image_path}\n")
                f.write("="*50 + "\n\n")
                for block in result:
                    f.write(f"Text: {block['text']}\n")
                    f.write(f"Confidence: {block['confidence']:.3f}\n")
                    f.write(f"Bounding Box: {block['bbox']}\n")
                    f.write("-"*30 + "\n")
            
            logger.info(f"Results saved to: {result_path}")
    
    logger.info(f"Batch OCR completed. Processed {len(image_paths)} images")
    return results

def extract_all_text(ocr_results: List[Dict], min_confidence: float = 0.1) -> str:
    """
    Extract all text from OCR results as a single string
    
    Args:
        ocr_results: List of OCR result dictionaries
        min_confidence: Minimum confidence to include text
    
    Returns:
        Combined text string
    """
    texts = []
    for block in ocr_results:
        if block['confidence'] >= min_confidence:
            texts.append(block['text'])
    
    return ' '.join(texts)

def filter_by_confidence(ocr_results: List[Dict], min_confidence: float) -> List[Dict]:
    """Filter OCR results by confidence threshold"""
    return [block for block in ocr_results if block['confidence'] >= min_confidence]

def get_text_statistics(ocr_results: List[Dict]) -> Dict:
    """Get statistics about OCR results"""
    if not ocr_results:
        return {"total_blocks": 0, "avg_confidence": 0, "total_characters": 0}
    
    confidences = [block['confidence'] for block in ocr_results]
    total_chars = sum(len(block['text']) for block in ocr_results)
    
    return {
        "total_blocks": len(ocr_results),
        "avg_confidence": sum(confidences) / len(confidences),
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
        "total_characters": total_chars
    }

# Example usage and testing
def main():
    """Example usage of the OCR engine"""
    try:
        # Initialize OCR engine
        logger.info("Initializing OCR engine...")
        ocr_engine = initialize_ocr(languages=['en'], use_gpu=True)
        
        logger.info(f"GPU enabled: {ocr_engine.is_gpu_enabled()}")
        if not ocr_engine.is_gpu_enabled():
            logger.warning("GPU acceleration not available. Using CPU only.")
            logger.warning("For better performance, install PyTorch with CUDA support.")
        
        # Test with a single image
        test_image = "../assets/test_image.jpg"  # Replace with actual image path
        
        if os.path.exists(test_image):
            logger.info(f"Testing OCR on: {test_image}")
            results = run_ocr(
                ocr_engine, 
                test_image, 
                confidence_threshold=0.1, 
                preprocessing_level='none',  # Use raw image
                enhance_image=False  # No enhancement
            )
            
            if results:
                print(f"\nOCR Results for {test_image}:")
                print("="*50)
                
                for i, block in enumerate(results, 1):
                    print(f"Block {i}:")
                    print(f"  Text: {block['text']}")
                    print(f"  Confidence: {block['confidence']:.3f}")
                    print(f"  Bounding Box: {block['bbox']}")
                    print("-"*30)
                
                # Get statistics
                stats = get_text_statistics(results)
                print(f"\nStatistics:")
                print(f"  Total blocks: {stats['total_blocks']}")
                print(f"  Average confidence: {stats['avg_confidence']:.3f}")
                print(f"  Total characters: {stats['total_characters']}")
                
                # Extract all text
                all_text = extract_all_text(results, min_confidence=0.1)
                print(f"\nExtracted text:\n{all_text}")
            else:
                print("No text detected in the image")
        else:
            print(f"Test image not found: {test_image}")
            print("Please provide a valid image path to test the OCR engine")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
