#!/usr/bin/env python3
"""
OCR Accuracy Analysis Tool
"""
import pandas as pd
from collections import Counter

def analyze_ocr_accuracy():
    """Analyze OCR accuracy from the results CSV"""
    print("📊 OCR Accuracy Analysis")
    print("="*50)
    
    try:
        # Load the results
        df = pd.read_csv('ocr_results.csv')
        total_images = len(df)
        
        print(f"📈 Dataset Overview:")
        print(f"  Total images processed: {total_images:,}")
        
        # Analyze text extraction success
        has_text = df['easyocr_text'].notna() & (df['easyocr_text'].str.strip() != '')
        success_count = has_text.sum()
        failure_count = total_images - success_count
        
        print(f"\n🎯 Text Detection Performance:")
        print(f"  Successful extractions: {success_count:,} ({success_count/total_images*100:.1f}%)")
        print(f"  Failed extractions: {failure_count:,} ({failure_count/total_images*100:.1f}%)")
        
        # Analyze text lengths
        text_lengths = df.loc[has_text, 'easyocr_text'].str.len()
        if len(text_lengths) > 0:
            print(f"\n📏 Text Length Statistics:")
            print(f"  Average length: {text_lengths.mean():.1f} characters")
            print(f"  Median length: {text_lengths.median():.1f} characters")
            print(f"  Max length: {text_lengths.max()} characters")
            print(f"  Min length: {text_lengths.min()} characters")
        
        # Analyze BERT corrections
        text_data = df[has_text].copy()
        corrections_made = (text_data['easyocr_text'] != text_data['corrected_easyocr']).sum()
        
        print(f"\n🔧 BERT Autocorrection Performance:")
        print(f"  Corrections applied: {corrections_made} ({corrections_made/success_count*100:.1f}% of successful extractions)")
        print(f"  No corrections needed: {success_count - corrections_made}")
        
        # Analyze correction types
        if corrections_made > 0:
            print(f"\n✨ Sample Corrections Made:")
            correction_examples = text_data[text_data['easyocr_text'] != text_data['corrected_easyocr']].head(10)
            
            for i, (_, row) in enumerate(correction_examples.iterrows(), 1):
                original = row['easyocr_text']
                corrected = row['corrected_easyocr'] 
                image = row['image']
                
                # Calculate edit distance
                import difflib
                diff = difflib.SequenceMatcher(None, original, corrected)
                similarity = diff.ratio()
                
                print(f"  {i}. {image}")
                print(f"     Original:  '{original}'")
                print(f"     Corrected: '{corrected}'")
                print(f"     Similarity: {similarity:.2f}")
                print()
        
        # Character-level accuracy analysis
        print(f"🔍 Character-Level Error Analysis:")
        all_text = ' '.join(df.loc[has_text, 'easyocr_text'].fillna(''))
        
        # Count character frequencies
        char_counts = Counter(all_text)
        total_chars = sum(char_counts.values())
        
        # Common OCR error characters
        error_chars = {'0', '1', 'I', 'l', 'O', '5', 'S', '8', 'B'}
        error_char_count = sum(char_counts.get(c, 0) for c in error_chars)
        
        print(f"  Total characters extracted: {total_chars:,}")
        print(f"  Potential error characters (0,1,I,l,O,5,S,8,B): {error_char_count} ({error_char_count/total_chars*100:.1f}%)")
        
        # Most common characters
        print(f"  Most common characters: {dict(char_counts.most_common(10))}")
        
        # Quality score calculation
        quality_factors = {
            'detection_rate': success_count / total_images,
            'avg_text_length': min(text_lengths.mean() / 50, 1) if len(text_lengths) > 0 else 0,  # Normalize to 50 chars
            'correction_rate': corrections_made / success_count if success_count > 0 else 0,
            'error_char_ratio': 1 - (error_char_count / total_chars) if total_chars > 0 else 0
        }
        
        overall_quality = (
            quality_factors['detection_rate'] * 0.4 +
            quality_factors['avg_text_length'] * 0.2 +
            quality_factors['correction_rate'] * 0.2 +
            quality_factors['error_char_ratio'] * 0.2
        )
        
        print(f"\n🎖️ Overall OCR Quality Score: {overall_quality*100:.1f}%")
        print(f"  Components:")
        print(f"    Text Detection: {quality_factors['detection_rate']*100:.1f}%")
        print(f"    Text Length: {quality_factors['avg_text_length']*100:.1f}%") 
        print(f"    Correction Rate: {quality_factors['correction_rate']*100:.1f}%")
        print(f"    Character Quality: {quality_factors['error_char_ratio']*100:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if success_count / total_images < 0.8:
            print("  - Lower OCR confidence threshold to detect more text")
        if corrections_made / success_count < 0.1:
            print("  - Lower BERT correction threshold for more aggressive correction") 
        if error_char_count / total_chars > 0.1:
            print("  - Add more OCR-specific character corrections")
        if text_lengths.mean() < 10:
            print("  - Images may have very short text - consider different preprocessing")
            
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_ocr_accuracy()
