# -*- coding: utf-8 -*-
"""BLIP-2 image caption in json
"""

#==============blip2================

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import json
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from google.colab import drive


drive.mount('/content/drive', force_remount=True)

class SlakeDataset(Dataset):
    def __init__(self, json_file, img_dir, processor, max_length=100):
        with open(json_file, 'r') as f:
            data = json.load(f)

        
        self.image_groups = {}
        for item in data:
            if item['q_lang'] != 'en':
                continue

            img_name = item['img_name']
            if img_name not in self.image_groups:
                self.image_groups[img_name] = {
                    'qa_pairs': [],
                    'img_path': os.path.join(img_dir, img_name)
                }
            self.image_groups[img_name]['qa_pairs'].append({
                'question': item['question'],
                'answer': item['answer']
            })

       
        self.unique_images = sorted(self.image_groups.keys())
        self.processor = processor
        self.max_length = max_length
        self.img_dir = img_dir

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        img_name = self.unique_images[idx]
        group = self.image_groups[img_name]
        image = Image.open(group['img_path']).convert('RGB')

        
        first_qa = group['qa_pairs'][0]
        text = f"Question: {first_qa['question']} Answer: {first_qa['answer']}"

       
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        return {
            "pixel_values": inputs.pixel_values.squeeze(),
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "image_id": img_name,
            "index": idx,
            "qa_pairs": group['qa_pairs']
        }

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'image_id': [item['image_id'] for item in batch],
        'index': [item['index'] for item in batch],
        'qa_pairs': [item['qa_pairs'] for item in batch]
    }

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device):
   model.train()
   total_loss = 0
   optimizer.zero_grad()
   gradient_accumulation_steps = 4

   for i, batch in enumerate(tqdm(train_loader, desc="Training")):
       pixel_values = batch['pixel_values'].to(device)
       input_ids = batch['input_ids'].to(device)
       attention_mask = batch['attention_mask'].to(device)

       with torch.amp.autocast('cuda'):
           loss = compute_loss(model, pixel_values, input_ids, attention_mask)
           loss = loss / gradient_accumulation_steps

       scaler.scale(loss).backward()

       if (i + 1) % gradient_accumulation_steps == 0:
           scaler.unscale_(optimizer)
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           scaler.step(optimizer)
           scaler.update()
           scheduler.step()
           optimizer.zero_grad()

       total_loss += loss.item() * gradient_accumulation_steps
       torch.cuda.empty_cache()

   return total_loss / len(train_loader)


def compute_loss(model, pixel_values, input_ids, attention_mask):
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids  
    )
    return outputs.loss

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device).half()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = compute_loss(model, pixel_values, input_ids, attention_mask)

            total_loss += loss.item()

    return total_loss / len(val_loader)

def generate_predictions(model, processor, test_loader, device, output_path):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            pixel_values = batch['pixel_values'].to(device).half()

            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                temperature=0.7
            )

            captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for idx in range(len(batch['image_id'])):
                results.append({
                    "index": batch['index'][idx],
                    "image_id": batch['image_id'][idx],
                    "caption": captions[idx],
                    "qa_pairs": batch['qa_pairs'][idx]
                })

    results.sort(key=lambda x: x['index'])
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float32,
        device_map="auto"
    ).to(device)
    model.config.use_cache = False  

   
    base_dir = "/content/drive/MyDrive/slakedataset/Slake1.0"
    train_json = os.path.join(base_dir, "train.json")
    val_json = os.path.join(base_dir, "validate.json")
    test_json = os.path.join(base_dir, "test.json")
    img_dir = os.path.join(base_dir, "imgs")

    
    train_dataset = SlakeDataset(train_json, img_dir, processor)
    val_dataset = SlakeDataset(val_json, img_dir, processor)
    test_dataset = SlakeDataset(test_json, img_dir, processor)

    
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*3)
    scaler = torch.amp.GradScaler()

   
    best_loss = float('inf')
    model_save_path = '/content/drive/MyDrive/modelparasave/blip2_slake_best.pth'

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_loss = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with val loss: {val_loss:.4f}")

    
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(model_save_path))

    output_path = '/content/drive/MyDrive/output/slake_test_results.json'
    results = generate_predictions(model, processor, test_loader, device, output_path)

    print("\nSample Predictions:")
    for item in results[:3]:
        print(f"\nImage: {item['image_id']}")
        print(f"Caption: {item['caption']}")
        print(f"Sample QA: {item['qa_pairs'][0]}")
        print("-" * 50)

if __name__ == "__main__":
    main()


#================================blip2 with slake=============================

from google.colab import drive
drive.mount('/content/drive')

import io
import sys
import os
import json
import torch
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm
import traceback

from transformers import Blip2Processor, Blip2ForConditionalGeneration

global_models = {}

def load_models():
    if 'blip2' not in global_models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        print("Loading BLIP-2 model...")
        
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        global_models['processor'] = processor
        global_models['model'] = model
        global_models['device'] = device

        print("BLIP-2 model loaded successfully")

def process_image_with_blip2(image_path, prompt=""):
    try:
        print(f"Loading image from: {image_path}")
       
        if not os.path.exists(image_path):
            print(f"ERROR: Image file not found: {image_path}")
            return "Error: Image file not found"

        
        image = Image.open(image_path).convert('RGB')
        print(f"Image successfully loaded. Size: {image.size}")

        
        print("Processing image with BLIP-2...")
        processor = global_models['processor']
        model = global_models['model']

        
        inputs = processor(images=image, return_tensors="pt").to(model.device)

        
        print("Generating description...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,    
                num_beams=5,           
                min_length=20,         
                top_p=0.9,             
                repetition_penalty=1.5, 
                length_penalty=1.0,     
                do_sample=True          
            )

        
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        
        if prompt:
            medical_caption = f"{prompt} {caption}"
            print(f"Generated caption: {medical_caption}")
            return medical_caption
        else:
            print(f"Generated caption: {caption}")
            return caption

    except Exception as e:
        print(f"Error processing image: {e}")
        print(traceback.format_exc())  
        return f"Error generating caption: {str(e)}"

def main():
    
    load_models()

    
    slake_base_dir = '/content/drive/MyDrive/slakedataset/Slake1.0'
    json_path = os.path.join(slake_base_dir, 'test.json')
    output_dir = '/content/drive/MyDrive/output'
    output_json = os.path.join(output_dir, 'slake_blip2_results.json')

    
    os.makedirs(output_dir, exist_ok=True)

    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    
    english_samples = [item for item in data if item.get('q_lang') == 'en']
    print(f"Loaded {len(english_samples)} English samples from test.json")

    
    modality_counts = {}
    for item in english_samples:
        modality = item.get('modality', 'Unknown')
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    print("Modality distribution:")
    for modality, count in modality_counts.items():
        print(f"  {modality}: {count} samples")

    results = []

    
    temp_output_json = os.path.join(output_dir, 'slake_blip2_results_temp.json')

    
    for idx, sample in tqdm(enumerate(english_samples), total=len(english_samples), desc="Processing images"):
        img_name = sample.get('img_name', '')
        img_id = sample.get('img_id', '')
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        modality = sample.get('modality', '')

        
        img_path = os.path.join(slake_base_dir, 'imgs', img_name)

        print(f"\nProcessing image {idx} (ID: {img_id})")
        print(f"Image path: {img_path}")
        print(f"Question: {question}")
        print(f"Modality: {modality}")

        try:
            
            if not os.path.exists(img_path):
                print(f"WARNING: Image file not found at {img_path}")
                
                alternative_paths = [
                    os.path.join(slake_base_dir, 'img', img_name),
                    os.path.join(slake_base_dir, 'images', img_name)
                ]
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        print(f"Found image at alternative path: {img_path}")
                        break

            
            prompt = ""
            caption_prefix = ""
            if modality == "CT":
                caption_prefix = "This CT scan shows"
            elif modality == "MRI":
                caption_prefix = "This MRI scan reveals"
            elif modality == "X-ray":
                caption_prefix = "This X-ray image displays"
            else:
                caption_prefix = "This medical image shows"

            
            blip2_caption = process_image_with_blip2(img_path)

            
            if not blip2_caption.lower().startswith("this"):
                blip2_caption = f"{caption_prefix} {blip2_caption}"

            
            result = {
                "id": idx,
                "img_id": img_id,
                "img_name": img_name,
                "question": question,
                "original_answer": answer,
                "modality": modality,
                "blip2_caption": blip2_caption
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing sample {idx}, image {img_name}: {e}")
            print(traceback.format_exc())
            
            results.append({
                "id": idx,
                "img_id": img_id,
                "img_name": img_name,
                "question": question,
                "original_answer": answer,
                "modality": modality,
                "blip2_caption": f"Error generating caption: {str(e)}"
            })

        
        if (idx + 1) % 10 == 0:
            try:
                with open(temp_output_json, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Temporary results saved to {temp_output_json} after processing {idx+1} samples")
            except Exception as save_error:
                print(f"Error saving temporary results: {save_error}")

        print("-" * 50)

    
    try:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Processing complete. Results saved to {output_json}")
    except Exception as save_error:
        print(f"Error saving final results: {save_error}")
        
        backup_output = os.path.join('/content', 'slake_blip2_results_backup.json')
        with open(backup_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to backup location: {backup_output}")


if __name__ == "__main__":
    main()





