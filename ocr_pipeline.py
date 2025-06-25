import os
import pandas as pd
import torch
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.ocr_engine import run_ocr, initialize_ocr
from modules.bert_autocorrector import BertAutocorrector

# Configure GPU settings
def setup_gpu_optimization():
    """Configure GPU settings for optimal performance"""
    if torch.cuda.is_available():
        print(f"✅ CUDA Available - GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        # Optimize GPU memory usage
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()  # Clear GPU cache
        return True
    else:
        print("⚠️ CUDA not available, using CPU")
        return False

def create_batches(data, batch_size):
    """Split data into batches"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_batch(batch_data, batch_num, total_batches, easyocr_engine, bert_autocorrector):
    """Process a batch of images"""
    print(f"\n🔄 Processing Batch {batch_num}/{total_batches} ({len(batch_data)} images)")
    
    batch_results = []
    
    with tqdm(total=len(batch_data), desc=f"Batch {batch_num}", ncols=100, position=0) as pbar:
        for img_data in batch_data:
            img_path, source = img_data
            
            # Check if image exists
            if not os.path.exists(img_path):
                # Try alternative paths
                alt_paths = [
                    os.path.join("train", os.path.basename(img_path)),
                    os.path.join("train_val_images", "train_images", os.path.basename(img_path)),
                    os.path.join("train_images", os.path.basename(img_path))
                ]
                
                found = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        found = True
                        break
                
                if not found:
                    pbar.set_postfix_str(f"❌ Not found: {os.path.basename(img_path)}")
                    pbar.update(1)
                    continue
            
            pbar.set_postfix_str(f"📷 {os.path.basename(img_path)}")
            
            try:
                # EasyOCR with optimized settings
                easyocr_blocks = run_ocr(
                    easyocr_engine, 
                    img_path, 
                    confidence_threshold=0.25,  # Lowered for better detection
                    enhancement_methods=['contrast', 'sharpen'],  # Add enhancement
                    post_process=True
                )
                easyocr_text = ' '.join([b['text'] for b in easyocr_blocks])
                
                # BERT autocorrect with lower threshold for more corrections
                corrected_easyocr = ""
                if easyocr_text.strip():
                    corrected_easyocr = bert_autocorrector.autocorrect(
                        easyocr_text, 
                        confidence_threshold=0.5  # Lowered from 0.7 for more corrections
                    )
                
                batch_results.append({
                    'image': img_path,
                    'easyocr_text': easyocr_text,
                    'corrected_easyocr': corrected_easyocr,
                    'source': source,
                    'blocks_count': len(easyocr_blocks),
                    'batch_num': batch_num
                })
                
            except Exception as e:
                pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}...")
                
            pbar.update(1)
    
    # Save batch results immediately
    if batch_results:
        batch_df = pd.DataFrame(batch_results)
        batch_filename = f'batch_{batch_num:02d}_results.csv'
        batch_df.to_csv(batch_filename, index=False)
        print(f"✅ Batch {batch_num} saved: {batch_filename} ({len(batch_results)} results)")
    
    return batch_results

# Setup GPU optimization
gpu_available = setup_gpu_optimization()

# Load image CSVs
print("📂 Loading CSV files...")
df_img = pd.read_csv('img.csv')
df_annot = pd.read_csv('annot.csv')

print(f"📊 Found {len(df_img)} images in img.csv")
print(f"📊 Found {len(df_annot)} annotations in annot.csv")

# Initialize OCR engines with GPU optimization
print("🚀 Initializing OCR engines...")
easyocr_engine = initialize_ocr(languages=['en'], use_gpu=gpu_available)
print(f"✅ EasyOCR GPU enabled: {easyocr_engine.is_gpu_enabled()}")

# Initialize BERT with GPU support and optimized settings
print("🤖 Initializing BERT Autocorrector...")
bert_autocorrector = BertAutocorrector(model_name="bert-base-multilingual-cased")
print("✅ BERT Autocorrector ready")

# Prepare image data for batch processing
print("📋 Preparing image data...")
image_data = []

# Add images from img.csv
for idx, row in df_img.iterrows():
    image_data.append((row['file_name'], 'img.csv'))

# Add images from annot.csv if file_name column exists
if 'file_name' in df_annot.columns:
    for idx, row in df_annot.iterrows():
        image_data.append((row['file_name'], 'annot.csv'))
else:
    print('⚠️ annot.csv does not contain a file_name column. Skipping.')

total_images = len(image_data)
print(f"📊 Total images to process: {total_images:,}")

# Calculate batch size for 20 batches
batch_size = max(1, total_images // 20)  # Ensure at least 1 image per batch
if total_images % 20 != 0:
    batch_size += 1  # Round up to ensure all images are processed

print(f"🔢 Batch size: {batch_size} images per batch")
print(f"📦 Will create approximately {min(20, (total_images + batch_size - 1) // batch_size)} batches")

# Process images in batches
all_results = []
start_time = time.time()

batches = list(create_batches(image_data, batch_size))
total_batches = len(batches)

print(f"\n� Starting batch processing...")
print(f"⏱️ Estimated time: {total_batches * 2:.1f} minutes (assuming 2 min/batch)")

for batch_num, batch_data in enumerate(batches, 1):
    batch_start = time.time()
    
    try:
        batch_results = process_batch(
            batch_data, batch_num, total_batches, 
            easyocr_engine, bert_autocorrector
        )
        all_results.extend(batch_results)
        
        batch_time = time.time() - batch_start
        remaining_batches = total_batches - batch_num
        estimated_remaining = remaining_batches * batch_time
        
        print(f"⏱️ Batch {batch_num} completed in {batch_time:.1f}s")
        print(f"📈 Progress: {batch_num}/{total_batches} ({batch_num/total_batches*100:.1f}%)")
        if remaining_batches > 0:
            print(f"🕐 Estimated remaining time: {estimated_remaining/60:.1f} minutes")
        
        # Clear GPU cache between batches
        if gpu_available:
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ Error in batch {batch_num}: {str(e)}")
        continue
    
    print("-" * 60)

# Combine all batch results and save final CSV
print(f"\n📊 Combining results from all batches...")
total_time = time.time() - start_time

if all_results:
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('ocr_results.csv', index=False)
    
    # Statistics
    successful_extractions = len([r for r in all_results if r['easyocr_text'].strip()])
    corrections_made = len([r for r in all_results if r['easyocr_text'] != r['corrected_easyocr']])
    
    print("\n📊 OCR Pipeline Complete!")
    print("="*60)
    print(f"✅ Total images processed: {len(all_results):,}")
    print(f"✅ Successfully extracted text from: {successful_extractions:,} ({successful_extractions/len(all_results)*100:.1f}%)")
    print(f"🔧 BERT corrections applied: {corrections_made:,} ({corrections_made/successful_extractions*100:.1f}%)")
    print(f"⏱️ Total processing time: {total_time/60:.1f} minutes")
    print(f"⚡ Average time per image: {total_time/len(all_results):.2f} seconds")
    print(f"✅ Results saved to: ocr_results.csv")
    
    # Show sample results
    print(f"\n📋 Sample results (first 3):")
    for i, result in enumerate(all_results[:3], 1):
        print(f"\n{i}. {os.path.basename(result['image'])}")
        print(f"   Original: {result['easyocr_text'][:80]}{'...' if len(result['easyocr_text']) > 80 else ''}")
        print(f"   Corrected: {result['corrected_easyocr'][:80]}{'...' if len(result['corrected_easyocr']) > 80 else ''}")
        print(f"   Blocks: {result['blocks_count']}")
        print(f"   Batch: {result['batch_num']}")
    
    # Cleanup batch files (optional)
    print(f"\n🧹 Cleaning up batch files...")
    batch_file_count = 0
    for i in range(1, total_batches + 1):
        batch_filename = f'batch_{i:02d}_results.csv'
        if os.path.exists(batch_filename):
            os.remove(batch_filename)
            batch_file_count += 1
    
    if batch_file_count > 0:
        print(f"✅ Removed {batch_file_count} temporary batch files")
    
    print(f"\n🎉 Pipeline execution completed successfully!")
    print(f"📈 Performance improvements applied:")
    print(f"   • Batch processing (20 batches)")
    print(f"   • Lowered OCR confidence threshold (0.25)")
    print(f"   • More aggressive BERT corrections (0.5 threshold)")
    print(f"   • Enhanced image processing")
    print(f"   • GPU memory optimization")
    
else:
    print("❌ No results generated. Check your image paths and try again.")
