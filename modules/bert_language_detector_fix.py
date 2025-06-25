"""
BERT-based language detection module using Hugging Face transformers.
"""
import logging
from typing import List, Tuple, Union
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTLanguageDetector:
    """
    Language detection using pre-trained BERT models from Hugging Face.
    """
    def __init__(self, model_name="papluca/xlm-roberta-base-language-detection", token=None):
        """
        Initialize the language detector with a pre-trained model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            token: Optional Hugging Face token for private models
        """
        self._model = None
        self._tokenizer = None
        self.model_name = model_name
        self.token = token
        self.is_initialized = False
        
        # Try to initialize the model
        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(f"BERT model initialization deferred: {e}")
    
    def _initialize_model(self):
        """Initialize the BERT model for language detection."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # Set authentication token if provided
            if self.token:
                os.environ["HUGGINGFACE_TOKEN"] = self.token
            
            # Initialize the tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Get the id2label mapping
            if hasattr(self._model, "config") and hasattr(self._model.config, "id2label"):
                self.id2label = self._model.config.id2label
            else:
                # Default mapping if not available
                self.id2label = {
                    0: "en", 1: "fr", 2: "de", 3: "es", 4: "it", 5: "ja",
                    6: "nl", 7: "pl", 8: "pt", 9: "ru", 10: "zh"
                }
            
            self.is_initialized = True
            logger.info(f"BERT language detector initialized with model: {self.model_name}")
            
            # Get supported languages
            self.supported_languages = list(self.id2label.values())
            logger.info(f"Supported languages: {', '.join(sorted(self.supported_languages))}")
            
        except ImportError:
            logger.error("Could not import 'transformers' or 'torch'. Make sure they're installed: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Error initializing BERT language detector: {e}")
            raise
    
    def predict(self, text: Union[str, List[str]]) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """
        Detect the language of the provided text(s).
        
        Args:
            text: Input text string or list of strings
            
        Returns:
            For single input: Tuple of (language_code, confidence)
            For list input: List of tuples (language_code, confidence)
        """
        import torch
        
        if not self.is_initialized:
            try:
                self._initialize_model()
            except Exception as e:
                logger.error(f"Failed to initialize BERT model: {e}")
                # Return English as fallback with low confidence
                if isinstance(text, list):
                    return [("en", 0.1) for _ in text]
                return "en", 0.1
        
        try:
            # Ensure input is properly formatted
            input_was_string = isinstance(text, str)
            if input_was_string:
                texts = [text]
            else:
                texts = text
            
            # Skip empty texts
            texts = [t for t in texts if t and t.strip()]
            if not texts:
                return ("en", 0.1) if input_was_string else [("en", 0.1)]
            
            # Process each text
            results = []
            for t in texts:
                # Tokenize and prepare input
                inputs = self._tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                # Run the model
                with torch.no_grad():
                    outputs = self._model(**inputs)
                
                # Get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
                # Get the highest probability and its index
                max_prob, max_idx = torch.max(probs, dim=0)
                
                # Convert to language code and confidence
                lang_code = self.id2label[max_idx.item()]
                confidence = max_prob.item()
                
                results.append((lang_code, confidence))
            
            # Return single result or list based on input
            if input_was_string:
                return results[0]
            else:
                return results
                
        except Exception as e:
            logger.error(f"Error during language detection: {e}")
            # Return English as fallback with low confidence
            if isinstance(text, list) and not input_was_string:
                return [("en", 0.1) for _ in text]
            return "en", 0.1

# Create a singleton instance for easy import
language_detector = BERTLanguageDetector()

def detect_language(text: Union[str, List[str]]) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
    """
    Simple function to detect language using the BERT model.
    
    Args:
        text: Input text string or list of strings
        
    Returns:
        For single input: Tuple of (language_code, confidence)
        For list input: List of tuples (language_code, confidence)
    """
    return language_detector.predict(text)
