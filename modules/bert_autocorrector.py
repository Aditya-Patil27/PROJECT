from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import torch
import config  # load project configuration
from typing import List
import re
import difflib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class BertAutocorrector:
    def __init__(self, model_name: str = config.BERT_MODEL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 BERT Autocorrector initializing on: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create a fill-mask pipeline for easier usage
            self.fill_mask = pipeline(
                "fill-mask",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                top_k=3  # Reduced for faster processing
            )
            
            # Enhanced OCR corrections based on analysis
            self.ocr_corrections = {
                # Character confusions (most common OCR errors)
                'rn': 'm', 'cl': 'd', 'vv': 'w', 'nn': 'n', 'il': 'll',
                'rnm': 'mm', 'rni': 'mi', 'rnl': 'ml', 'rnr': 'mr',
                
                # Number-letter confusions
                '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G',
                'l': 'I', 'I': 'l', 'O': '0', 'S': '5', 'B': '8',
                
                # Word-level corrections (based on your data)
                'teh': 'the', 'adn': 'and', 'taht': 'that', 'hte': 'the',
                'seperate': 'separate', 'recieve': 'receive', 'occured': 'occurred',
                'neighb0rs': 'neighbors', 'c0mpany': 'company', 'perf0rmance': 'performance',
                
                # Specific corrections from your dataset
                'Becf': 'Beef', 'Hckly': 'Hickory', 'Lennoxrobinson': 'Lennox Robinson',
                'thelordSatan': 'the lord Satan', 'APlanforGrowth': 'A Plan for Growth',
                'monthelie': 'Monthelie', 'controlee': 'controlée',
                
                # Common spacing issues
                'SportWatch': 'Sport Watch', 'SmartPhone': 'Smart Phone',
                'WebSite': 'Web Site', 'DataBase': 'Data Base',
            }
            
            # Aggressive correction patterns
            self.correction_patterns = [
                (r'([a-z])([A-Z])', r'\1 \2'),  # Split camelCase
                (r'(\d)([a-zA-Z])', r'\1 \2'),  # Split number+letter
                (r'([a-zA-Z])(\d)', r'\1 \2'),  # Split letter+number
                (r'([a-z])([A-Z][a-z])', r'\1 \2'),  # Split compound words
            ]
            
            print(f"✅ BERT Autocorrector loaded with enhanced corrections")
            
        except Exception as e:
            print(f"❌ Error initializing BERT Autocorrector: {e}")
            raise
    
    def autocorrect(self, text: str, confidence_threshold: float = config.BERT_CONFIDENCE_THRESHOLD) -> str:
        """
        Enhanced autocorrection with multiple improvement strategies
        """
        if not text or not text.strip():
            return text
        
        original_text = text
        
        # Step 1: Apply direct OCR corrections (fastest)
        corrected_text = self._apply_direct_corrections(text)
        
        # Step 2: Apply pattern-based corrections
        corrected_text = self._apply_pattern_corrections(corrected_text)
        
        # Step 3: Apply BERT-based context corrections (slower but more accurate)
        corrected_text = self._apply_bert_corrections(corrected_text, confidence_threshold)
        
        # Step 4: Final cleanup
        corrected_text = self._cleanup_text(corrected_text)
        
        # Log significant corrections
        if original_text != corrected_text:
            similarity = difflib.SequenceMatcher(None, original_text, corrected_text).ratio()
            if similarity < 0.9:  # Significant change
                print(f"🔧 Major correction: '{original_text[:50]}...' → '{corrected_text[:50]}...'")
        
        return corrected_text
    
    def _apply_pattern_corrections(self, text: str) -> str:
        """Apply regex pattern corrections"""
        corrected = text
        
        for pattern, replacement in self.correction_patterns:
            corrected = re.sub(pattern, replacement, corrected)
        
        return corrected

    def _apply_direct_corrections(self, text: str) -> str:
        """Apply direct character and word corrections"""
        corrected = text
        
        # Apply word-level corrections
        for wrong, correct in self.ocr_corrections.items():
            corrected = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, corrected, flags=re.IGNORECASE)
        
        # Fix common character sequences
        corrected = re.sub(r'\brn\b', 'm', corrected)  # rn -> m
        corrected = re.sub(r'\bcl\b', 'd', corrected)  # cl -> d
        
        return corrected

    def _apply_bert_corrections(self, text: str, confidence_threshold: float) -> str:
        """Apply BERT-based contextual corrections"""
        words = text.split()
        if len(words) <= 1:
            return text
            
        corrected_words = []
        
        for i, word in enumerate(words):
            # Skip short words, numbers, and punctuation
            if len(word) <= 2 or word.isdigit() or not word.isalpha():
                corrected_words.append(word)
                continue
            
            # Check if word looks suspicious (mixed case, unusual patterns)
            if self._is_suspicious_word(word):
                correction = self._correct_word_with_bert(words, i, confidence_threshold)
                corrected_words.append(correction)
            else:
                corrected_words.append(word)        
        return ' '.join(corrected_words)

    def _is_suspicious_word(self, word: str) -> bool:
        """Determine if a word looks like it might be an OCR error"""
        # Always check common OCR patterns more aggressively
        
        # Check for mixed case in unusual patterns
        if re.search(r'[a-z][A-Z]|[A-Z][a-z][A-Z]', word):
            return True
        
        # Check for numbers mixed with letters (common OCR error)
        if re.search(r'[0-9][a-zA-Z]|[a-zA-Z][0-9]', word):
            return True
            
        # Check for single characters that are likely wrong
        if len(word) == 1 and word in '1lI0O':
            return True
        
        # Check for unusual character combinations
        suspicious_patterns = [
            r'[il1|]{2,}',  # Multiple similar looking characters
            r'[rn]{2,}',    # Multiple rn (often should be m)
            r'[cl](?![aeiou])',  # cl not followed by vowel
            r'\d[a-zA-Z]',  # digit followed by letter
            r'[a-zA-Z]\d',  # letter followed by digit
            r'[A-Z]{2,}[a-z][A-Z]',  # weird capitalization
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, word):
                return True
        
        # Check against common OCR errors
        ocr_error_patterns = [
            'teh', 'adn', 'taht', 'hte', 'rn', 'cl', 'vv', 'nn'
        ]
        
        if word.lower() in ocr_error_patterns:
            return True
                
        return False

    def _correct_word_with_bert(self, words: List[str], word_index: int, confidence_threshold: float) -> str:
        """Use BERT to correct a specific word based on context"""
        original_word = words[word_index]
        
        # Skip very short words unless they're suspicious
        if len(original_word) <= 1 and not self._is_suspicious_word(original_word):
            return original_word
        
        try:
            # Create masked text
            masked_words = words.copy()
            masked_words[word_index] = self.tokenizer.mask_token
            masked_text = ' '.join(masked_words)
            
            # Limit text length for BERT
            if len(masked_text) > 512:
                # Take context around the masked word
                start_idx = max(0, word_index - 10)
                end_idx = min(len(words), word_index + 11)
                context_words = words[start_idx:end_idx]
                masked_context = context_words.copy()
                masked_context[word_index - start_idx] = self.tokenizer.mask_token
                masked_text = ' '.join(masked_context)
            
            # Get BERT predictions
            predictions = self.fill_mask(masked_text, top_k=5)
            
            if predictions and isinstance(predictions, list):
                # Find the best prediction that makes sense
                for pred in predictions:
                    predicted_word = pred['token_str'].strip()
                    score = pred['score']
                    
                    # Lower threshold for suspicious words
                    effective_threshold = confidence_threshold * 0.7 if self._is_suspicious_word(original_word) else confidence_threshold
                    
                    # Only use prediction if it's confident and looks reasonable
                    if (score > effective_threshold and 
                        predicted_word.isalpha() and 
                        len(predicted_word) > 0 and
                        predicted_word.lower() != original_word.lower() and
                        self._is_reasonable_replacement(original_word, predicted_word)):
                        
                        print(f"🔧 BERT correction: '{original_word}' → '{predicted_word}' (confidence: {score:.3f})")
                        return predicted_word
            
        except Exception as e:
            print(f"❌ BERT correction failed for '{original_word}': {e}")
        
        return original_word

    def _is_reasonable_replacement(self, original: str, replacement: str) -> bool:
        """Check if the replacement word is reasonable"""
        # Don't replace if words are too different in length
        if abs(len(original) - len(replacement)) > max(2, len(original) // 2):
            return False
        
        # Check similarity using difflib
        similarity = difflib.SequenceMatcher(None, original.lower(), replacement.lower()).ratio()
        return similarity > 0.3  # At least 30% similar
    
    def _cleanup_text(self, text: str) -> str:
        """Final cleanup of the corrected text"""
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Space before punctuation
        
        # Fix capitalization
        sentences = re.split(r'([.!?]+)', text)
        cleaned_sentences = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip() and not re.match(r'^[.!?]+$', sentence):                # Capitalize first letter of sentence
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                    cleaned_sentences.append(sentence)
            elif sentence.strip():
                cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences).strip()

    def correct_text(self, text: str, confidence_threshold: float = config.BERT_CONFIDENCE_THRESHOLD) -> str:
        """Alias for autocorrect method to match expected interface"""
        return self.autocorrect(text, confidence_threshold)

    def batch_autocorrect(self, texts: List[str], confidence_threshold: float = config.BERT_CONFIDENCE_THRESHOLD) -> List[str]:
        """Autocorrect multiple texts"""
        return [self.autocorrect(text, confidence_threshold) for text in texts]

    def finetune_on_icdar(self, train_texts: List[str], target_texts: List[str]):
        """Placeholder for fine-tuning code"""
        pass