# -*- coding: utf-8 -*-
"""image caption with tag2text
"""


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


sys.path.append('/content/drive/MyDrive/recognize-anything-main')

# Commented out IPython magic to ensure Python compatibility.
try:
    from ram.models import tag2text
    from ram import get_transform
    print("Successfully imported tag2text from Google Drive")
except ImportError as e:
    print(f"Error importing from Google Drive: {e}")
    print("Falling back to cloning the repository in Colab's local filesystem")
    !git clone https://github.com/xinyu1205/recognize-anything.git /content/recognize-anything
#     %cd /content/recognize-anything
    !pip install -e .
#     %cd /content
    sys.path.append('/content/recognize-anything')
    from ram.models import tag2text
    from ram import get_transform


global_models = {}

def load_models():
    if 'tag2text' not in global_models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        
        delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]
        global_models['tag2text'] = tag2text(pretrained="/content/drive/MyDrive/recognize-anything-main/pretrained/tag2text_swin_14m.pth",
                                             image_size=384,
                                             vit='swin_b',
                                             delete_tag_index=delete_tag_index)
        global_models['tag2text'].threshold = 0.68  
        global_models['tag2text'].eval()
        global_models['tag2text'] = global_models['tag2text'].to(device)
        print("Tag2Text model loaded successfully")

        
        global_models['transform'] = get_transform(image_size=384)

def process_image(image_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if isinstance(image_data, dict):
        image_bytes = image_data['bytes']
    elif isinstance(image_data, str):
        image_bytes = json.loads(image_data)['bytes']
    else:
        image_bytes = image_data
    image_bytes = bytes(image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    print(f"Image size: {image.size}")

    image_tensor = global_models['transform'](image).unsqueeze(0).to(device)

    try:
        
        with torch.no_grad():
            result = global_models['tag2text'].generate(image_tensor, return_tag_predict=True)

        if isinstance(result, tuple) and len(result) == 2:
            captions, tags = result
            caption = captions[0] if captions else "No caption generated"
        else:
            tags = result[0] if isinstance(result, list) else "No tags generated"
            caption = "No caption generated"

        
        if isinstance(tags, str):
            tags = tags.split('|')

        print(f"Tag2Text generated tags: {tags}")
        print(f"Tag2Text generated caption: {caption}")

        return tags, caption
    except Exception as e:
        print(f"Error processing image: {e}")
        return [], "Error generating caption"

def main():
    
    load_models()

    
    input_parquet = '/content/drive/MyDrive/RADdataset/test-00000-of-00001-e5bc3d208bb4deeb.parquet'
    output_json = '/content/drive/MyDrive/RADdataset/processed_data.json'

    
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df)} rows from the parquet file.")

    results = []

    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_data = row['image']
        question = row['question']
        answer = row['answer']

        print(f"\nProcessing image {index}")
        print(f"Question: {question}")

        try:
            
            tags, tag2text_caption = process_image(image_data)

            
            result = {
                "index": index,
                "question": question,
                "original_answer": answer,
                "tag2text_caption": tag2text_caption,
                "tags": tags,
                "blip3_caption": None  
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing image {index}: {e}")
            
            results.append({
                "index": index,
                "question": question,
                "original_answer": answer,
                "tag2text_caption": "Error generating caption",
                "tags": [],
                "blip3_caption": None
            })

        print("-" * 50)

    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Processing complete. Results saved to {output_json}")


if __name__ == "__main__":
    main()

#=======tag2text with slake========


from google.colab import drive
drive.mount('/content/drive')#
import io
import sys
import os
import json
import torch
from PIL import Image
import pandas as pd
from tqdm.notebook import tqdm


sys.path.append('/content/drive/MyDrive/recognize-anything-main')
try:
    
    import importlib
    if not importlib.util.find_spec("fairscale"):
        print("Installing fairscale...")
        !pip install fairscale
    if not importlib.util.find_spec("timm"):
        print("Installing timm...")
        !pip install timm
    if not importlib.util.find_spec("transformers"):
        print("Installing transformers...")
        !pip install transformers

    from ram.models import tag2text
    from ram import get_transform
    print("Successfully imported tag2text from Google Drive")
except ImportError as e:
    print(f"Error importing from Google Drive: {e}")
    print("Falling back to cloning the repository in Colab's local filesystem")
    !git clone https://github.com/xinyu1205/recognize-anything.git /content/recognize-anything
#     %cd /content/recognize-anything
   
    !pip install fairscale timm transformers
    !pip install -e .
#     %cd /content
    sys.path.append('/content/recognize-anything')
    from ram.models import tag2text
    from ram import get_transform


global_models = {}

def load_models():
    if 'tag2text' not in global_models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        
        delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]
        global_models['tag2text'] = tag2text(pretrained="/content/drive/MyDrive/recognize-anything-main/pretrained/tag2text_swin_14m.pth",
                                           image_size=384,
                                           vit='swin_b',
                                           delete_tag_index=delete_tag_index)
        global_models['tag2text'].threshold = 0.68  
        global_models['tag2text'].eval()
        global_models['tag2text'] = global_models['tag2text'].to(device)
        print("Tag2Text model loaded successfully")

        
        global_models['transform'] = get_transform(image_size=384)

def process_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        print(f"Loading image from: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"ERROR: Image file not found: {image_path}")
            return [], "Error: Image file not found"

        
        image = Image.open(image_path).convert('RGB')
        print(f"Image successfully loaded. Size: {image.size}")

        
        print("Applying image transformation...")
        image_tensor = global_models['transform'](image).unsqueeze(0).to(device)
        print(f"Image transformed to tensor of shape: {image_tensor.shape}")

        
        print("Running Tag2Text model inference...")
        with torch.no_grad():
            result = global_models['tag2text'].generate(image_tensor, return_tag_predict=True)
        print("Tag2Text inference completed")

        if isinstance(result, tuple) and len(result) == 2:
            captions, tags = result
            caption = captions[0] if captions else "No caption generated"
            print(f"Got caption: {caption}")
        else:
            tags = result[0] if isinstance(result, list) else "No tags generated"
            caption = "No caption generated"
            print("No caption in result, only tags")

       
        if isinstance(tags, str):
            tags = tags.split('|')

        print(f"Tag2Text generated tags: {tags}")
        print(f"Tag2Text generated caption: {caption}")

        return tags, caption
    except Exception as e:
        import traceback
        print(f"Error processing image: {e}")
        print(traceback.format_exc())  
        return [], f"Error generating caption: {str(e)}"

def main():
    
    load_models()

    
    slake_base_dir = '/content/drive/MyDrive/slakedataset/Slake1.0'
    json_path = os.path.join(slake_base_dir, 'test.json')
    output_dir = '/content/drive/Research1/output'
    output_json = os.path.join(output_dir, 'slake_tag2text_results.json')

   
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    
    english_samples = [item for item in data if item.get('q_lang') == 'en']
    print(f"Loaded {len(english_samples)} English samples from test.json")

    results = []

    
    temp_output_json = os.path.join(output_dir, 'slake_tag2text_results_temp.json')

    
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

            
            tags, tag2text_caption = process_image(img_path)

            
            result = {
                "id": idx,
                "img_id": img_id,
                "img_name": img_name,
                "question": question,
                "original_answer": answer,
                "modality": modality,
                "tag2text_caption": tag2text_caption,
                "tags": tags,
                "blip3_caption": None  
            }
            results.append(result)
        except Exception as e:
            import traceback
            print(f"Error processing sample {idx}, image {img_name}: {e}")
            print(traceback.format_exc())
            
            results.append({
                "id": idx,
                "img_id": img_id,
                "img_name": img_name,
                "question": question,
                "original_answer": answer,
                "modality": modality,
                "tag2text_caption": f"Error generating caption: {str(e)}",
                "tags": [],
                "blip3_caption": None
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
        
        backup_output = os.path.join('/content', 'slake_tag2text_results_backup.json')
        with open(backup_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to backup location: {backup_output}")


if __name__ == "__main__":
    main()







