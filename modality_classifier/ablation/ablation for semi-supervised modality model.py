# -*- coding: utf-8 -*-
"""semi-supervised modality modal-ablation study.ipynb
"""

#only fixmatch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random
import os
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def extract_modality(text):
    """extract modality from questions or answers"""
    if not isinstance(text, str):
        return None

    text = text.lower()
    modality_patterns = {
        'mri': r'\b(mri|magnetic resonance( imaging)?|mr imaging|nmr|magnetic resonance imaging)\b',
        'ct': r'\b(ct scan|ct|computed tomography|cat scan|computerized tomography|computed axial tomography)\b',
        'xray': r'\b(x-ray|xray|radiograph|radiography|chest x|cxr|roentgen|radiogram|plain film)\b'
    }

    for modality, pattern in modality_patterns.items():
        if re.search(pattern, text):
            return modality
    return None

# baseline-only fixmatch
class BaseModalityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.modality_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.modality_head(features)

# MedicalModalityAttention
class MedicalModalityAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.anatomy_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.texture_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        anatomy_weight = self.anatomy_attention(x)
        texture_weight = self.texture_attention(x)
        return x * anatomy_weight * texture_weight

# compaired model-fixmatch +  AttentionModalityClassifier
class AttentionModalityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.attention1 = MedicalModalityAttention(256)
        self.attention2 = MedicalModalityAttention(512)
        self.attention3 = MedicalModalityAttention(1024)
        self.attention4 = MedicalModalityAttention(2048)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.modality_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.attention1(self.backbone.layer1(x))
        x = self.attention2(self.backbone.layer2(x))
        x = self.attention3(self.backbone.layer3(x))
        x = self.attention4(self.backbone.layer4(x))

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return self.modality_head(features)


class MedicalImageAugmentation:
    def __init__(self):
        self.base_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        self.strong_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15)
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])


class RADDataset(Dataset):
    def __init__(self, df, is_labeled=True, mode='train'):
        self.df = df
        self.is_labeled = is_labeled
        self.mode = mode
        self.augmentation = MedicalImageAugmentation()

        self.valid_samples = []
        self.unlabeled_samples = []
        self.modality_to_idx = {'mri': 0, 'ct': 1, 'xray': 2}

        for idx, row in df.iterrows():
            question_modality = extract_modality(row['question'])
            answer_modality = extract_modality(row['answer'])
            modality = question_modality or answer_modality

            if modality in ['mri', 'ct', 'xray']:
                self.valid_samples.append({
                    'image': row['image'],
                    'modality': modality,
                    'index': idx
                })
            else:
                self.unlabeled_samples.append({
                    'image': row['image'],
                    'index': idx
                })

    def decode_image(self, image_data):
        try:
            if isinstance(image_data, dict):
                if 'bytes' in image_data:
                    bytes_data = image_data['bytes']
                    if isinstance(bytes_data, str):
                        bytes_data = bytes_data.encode('latin1')
                    img = Image.open(io.BytesIO(bytes_data))
                    return np.array(img)
                elif 'array' in image_data:
                    return np.array(image_data['array'])
            return np.array(image_data)
        except Exception as e:
            print(f"Error decoding image: {str(e)}")
            raise

    def __len__(self):
        if self.is_labeled:
            return len(self.valid_samples)
        return len(self.unlabeled_samples)

    def __getitem__(self, idx):
        if self.is_labeled:
            sample = self.valid_samples[idx]
            image_data = sample['image']
            label = self.modality_to_idx[sample['modality']]

            if self.mode == 'train':
                image_tensor = self._process_image(image_data, aug_type='base')
            else:
                image_tensor = self._process_image(image_data, aug_type='test')
            return image_tensor, label
        else:
            sample = self.unlabeled_samples[idx]
            image_data = sample['image']
            weak_aug = self._process_image(image_data, aug_type='base')
            strong_aug = self._process_image(image_data, aug_type='strong')
            return weak_aug, strong_aug

    def _process_image(self, image_data, aug_type='base'):
        try:
            image_array = self.decode_image(image_data)
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]

            image = Image.fromarray(image_array.astype('uint8'))

            if aug_type == 'strong':
                return self.augmentation.strong_aug(image)
            elif aug_type == 'test':
                return self.augmentation.test_transform(image)
            else:
                return self.augmentation.base_aug(image)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    modality_correct = {'mri': 0, 'ct': 0, 'xray': 0}
    modality_total = {'mri': 0, 'ct': 0, 'xray': 0}

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            
            for i, (pred, label) in enumerate(zip(predicted, labels)):
                modality = ['mri', 'ct', 'xray'][label.item()]
                modality_total[modality] += 1
                if pred == label:
                    modality_correct[modality] += 1

    avg_loss = total_loss / len(val_loader)
    overall_acc = correct / total

    modality_accuracies = {
        modality: modality_correct[modality]/modality_total[modality]
        for modality in ['mri', 'ct', 'xray']
        if modality_total[modality] > 0
    }

    return avg_loss, overall_acc, modality_accuracies

# FixMatch train
def train_fixmatch(model, train_loader, unlabeled_loader, val_loader, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # save
    save_dir = '/content/drive/MyDrive/output'
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 10
    threshold = 0.95  # threshold
    best_acc = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'modality_metrics': []
    }

    print(f"\nTraining {model_name}...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        modality_correct = {'mri': 0, 'ct': 0, 'xray': 0}
        modality_total = {'mri': 0, 'ct': 0, 'xray': 0}

        unlabeled_iter = iter(unlabeled_loader)
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)

            try:
                weak_aug, strong_aug = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                weak_aug, strong_aug = next(unlabeled_iter)

            weak_aug = weak_aug.to(device)
            strong_aug = strong_aug.to(device)

            optimizer.zero_grad()

            # labeledloss
            outputs = model(inputs)
            sup_loss = criterion(outputs, labels)

            # unlabeled loss
            with torch.no_grad():
                pseudo_outputs = model(weak_aug)
                pseudo_probs = F.softmax(pseudo_outputs, dim=1)
                max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
                mask = max_probs.ge(threshold)

            
            strong_outputs = model(strong_aug)
            unsup_loss = (F.cross_entropy(strong_outputs, pseudo_labels,
                                        reduction='none') * mask).mean()

            loss = sup_loss + unsup_loss
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

           
            for i, (pred, label) in enumerate(zip(predicted, labels)):
                modality = ['mri', 'ct', 'xray'][label.item()]
                modality_total[modality] += 1
                if pred == label:
                    modality_correct[modality] += 1

            total_loss += loss.item()

            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        
        train_acc = correct / total
        modality_accuracies = {
            modality: modality_correct[modality]/modality_total[modality]
            for modality in ['mri', 'ct', 'xray']
            if modality_total[modality] > 0
        }

        # val
        val_loss, val_acc, val_modality_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # print
        print(f'\nEpoch {epoch+1}:')
        print(f'Training Loss: {total_loss/(batch_idx+1):.4f}')
        print(f'Training Accuracy: {100.*train_acc:.2f}%')
        print('Training Modality Accuracies:')
        for modality, acc in modality_accuracies.items():
            print(f'{modality.upper()}: {100.*acc:.2f}%')

        print(f'\nValidation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {100.*val_acc:.2f}%')
        print('Validation Modality Accuracies:')
        for modality, acc in val_modality_acc.items():
            print(f'{modality.upper()}: {100.*acc:.2f}%')

        
        history['train_loss'].append(total_loss/(batch_idx+1))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['modality_metrics'].append(val_modality_acc)

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, f'{model_name}_best.pth')
            print(f'\nSaving new best model with validation accuracy: {100.*best_acc:.2f}% to {save_path}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, save_path)

    return history

def plot_training_history(base_history, attention_history):
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(base_history['train_acc'], label='Base Train')
    plt.plot(base_history['val_acc'], label='Base Val')
    plt.plot(attention_history['train_acc'], label='Attention Train')
    plt.plot(attention_history['val_acc'], label='Attention Val')
    plt.title('Overall Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    
    plt.subplot(1, 2, 2)
    plt.plot(base_history['train_loss'], label='Base Train')
    plt.plot(base_history['val_loss'], label='Base Val')
    plt.plot(attention_history['train_loss'], label='Attention Train')
    plt.plot(attention_history['val_loss'], label='Attention Val')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def run_ablation_study():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    print("\nLoading RAD dataset...")
    train_path = '/content/drive/MyDrive/RADdataset/train-00000-of-00001-eb8844602202be60.parquet'
    test_path = '/content/drive/MyDrive/RADdataset/test-00000-of-00001-e5bc3d208bb4deeb.parquet'

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    
    labeled_dataset = RADDataset(combined_df, is_labeled=True, mode='train')
    unlabeled_dataset = RADDataset(combined_df, is_labeled=False)
    test_dataset = RADDataset(combined_df, is_labeled=True, mode='test')

    
    train_size = int(0.8 * len(labeled_dataset))
    val_size = len(labeled_dataset) - train_size
    train_dataset, val_dataset = random_split(labeled_dataset, [train_size, val_size])

   
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size*2,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # 1. FixMatch
    print("\n=== Running Pure FixMatch ===")
    base_model = BaseModalityClassifier().to(device)
    base_history = train_fixmatch(
        base_model, train_loader, unlabeled_loader, val_loader,
        device, "base_fixmatch"
    )

    # 2. FixMatch+ AttentionModalityClassifier
    print("\n=== Running FixMatch + Medical Attention ===")
    attention_model = AttentionModalityClassifier().to(device)
    attention_history = train_fixmatch(
        attention_model, train_loader, unlabeled_loader, val_loader,
        device, "attention_fixmatch"
    )

    
    plot_training_history(base_history, attention_history)

    
    print("\n=== Final Results ===")
    print("Pure FixMatch:")
    print(f"Best Validation Accuracy: {100.*max(base_history['val_acc']):.2f}%")
    print("\nFixMatch + Medical Attention:")
    print(f"Best Validation Accuracy: {100.*max(attention_history['val_acc']):.2f}%")

if __name__ == "__main__":
    run_ablation_study()



#====================test slake=====================:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import json
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class BaseModalityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.modality_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.modality_head(features)


class MedicalModalityAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.anatomy_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.texture_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        anatomy_weight = self.anatomy_attention(x)
        texture_weight = self.texture_attention(x)
        return x * anatomy_weight * texture_weight


class AttentionModalityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(weights=None)

        self.attention1 = MedicalModalityAttention(256)
        self.attention2 = MedicalModalityAttention(512)
        self.attention3 = MedicalModalityAttention(1024)
        self.attention4 = MedicalModalityAttention(2048)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.modality_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.attention1(self.backbone.layer1(x))
        x = self.attention2(self.backbone.layer2(x))
        x = self.attention3(self.backbone.layer3(x))
        x = self.attention4(self.backbone.layer4(x))

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return self.modality_head(features)

class SLAKEDataset(Dataset):
    def __init__(self, json_path, base_dir):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        
        self.samples = [item for item in self.data if item['q_lang'] == 'en']

        self.base_dir = base_dir
        self.modality_to_idx = {'MRI': 0, 'CT': 1, 'X-ray': 2, 'Xray': 2, 'X-Ray': 2, 'XRAY': 2}

        
        modality_counts = {}
        for item in self.samples:
            modality = item.get('modality', 'Unknown')
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        print("modality distribution:")
        for modality, count in modality_counts.items():
            print(f"  {modality}: {count}samples")

        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"upload {len(self.samples)} english samples")

        
        self.img_dir_name = 'imgs'  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        
        img_path = os.path.join(self.base_dir, self.img_dir_name, sample['img_name'])

        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)

            
            modality = sample.get('modality', None)

            
            if modality:
                if modality.upper() in ['XRAY', 'X-RAY', 'X RAY']:
                    modality = 'X-ray'
                elif modality.upper() == 'MR':
                    modality = 'MRI'

            
            if modality in self.modality_to_idx:
                label = self.modality_to_idx[modality]
            else:
                
                label = -1

            return image_tensor, label, sample['img_id']

        except Exception as e:
            print(f"wrongly upload {img_path}: {e}")
            
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, -1, sample['img_id']

def test_model(model, test_loader, device, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    all_img_ids = []
    correct = 0
    total = 0

    
    modality_correct = {'MRI': 0, 'CT': 0, 'X-ray': 0}
    modality_total = {'MRI': 0, 'CT': 0, 'X-ray': 0}
    idx_to_modality = {0: 'MRI', 1: 'CT', 2: 'X-ray'}

    with torch.no_grad():
        for inputs, labels, img_ids in tqdm(test_loader, desc=f'Testing {model_name}'):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            
            valid_mask = (labels >= 0).cpu()

            
            if not valid_mask.any():
                continue

            
            valid_inputs = inputs[valid_mask].to(device)
            valid_labels = labels[valid_mask].to(device)
            valid_img_ids = img_ids[valid_mask]  # img_ids保持在CPU上


            if len(valid_inputs) == 0:
                continue

            outputs = model(valid_inputs)
            _, predicted = outputs.max(1)

            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            all_img_ids.extend(valid_img_ids.cpu().numpy())

            
            total += valid_labels.size(0)
            correct += predicted.eq(valid_labels).sum().item()

            
            for i, (pred, label) in enumerate(zip(predicted, valid_labels)):
                modality = idx_to_modality[label.item()]
                modality_total[modality] += 1
                if pred == label:
                    modality_correct[modality] += 1

    
    overall_acc = correct / total if total > 0 else 0
    print(f'\n{model_name} test results:')
    print(f'total acc: {100.*overall_acc:.2f}%')

    
    print('modality acc:')
    for modality in modality_total.keys():
        if modality_total[modality] > 0:
            acc = modality_correct[modality] / modality_total[modality]
            print(f'{modality}: {100.*acc:.2f}% ({modality_correct[modality]}/{modality_total[modality]})')

    # confused matrixed
    if len(all_labels) > 0:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=['MRI', 'CT', 'X-ray'],
                  yticklabels=['MRI', 'CT', 'X-ray'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(f'/content/drive/MyDrive/output/{model_name}_confusion_matrix.png')
        plt.show()

        
        report = classification_report(all_labels, all_preds,
                                      target_names=['MRI', 'CT', 'X-ray'],
                                      digits=4)
        print("\nclassification report:")
        print(report)

        
        error_indices = [i for i in range(len(all_preds)) if all_preds[i] != all_labels[i]]
        error_img_ids = [all_img_ids[i] for i in error_indices]
        error_true = [idx_to_modality[all_labels[i]] for i in error_indices]
        error_pred = [idx_to_modality[all_preds[i]] for i in error_indices]

        error_df = pd.DataFrame({
            'img_id': error_img_ids,
            'true_modality': error_true,
            'pred_modality': error_pred
        })

        print(f"\nwrong: {len(error_indices)}")
        if len(error_indices) > 0:
            print("wrong sample:")
            print(error_df.head(10))
            error_df.to_csv(f'/content/drive/MyDrive/output/{model_name}_errors.csv', index=False)

    return overall_acc, {modality: modality_correct[modality]/modality_total[modality]
                        if modality_total[modality] > 0 else 0
                        for modality in modality_total.keys()}

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    
    slake_base_dir = '/content/drive/MyDrive/slakedataset/Slake1.0'
    json_path = os.path.join(slake_base_dir, 'test.json')

    
    print(f"json: {json_path}")
    print(f"exsit: {os.path.exists(json_path)}")

    
    base_model_path = '/content/drive/MyDrive/output/base_fixmatch_best.pth'
    attention_model_path = '/content/drive/MyDrive/output/attention_fixmatch_best.pth'

    
    test_dataset = SLAKEDataset(json_path, slake_base_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    
    if len(test_dataset) == 0:
        print("warning: double check if empty")
        return

    results = {}

    
    if os.path.exists(base_model_path):
        print(f"\nupload: {base_model_path}")
        base_model = BaseModalityClassifier().to(device)
        checkpoint = torch.load(base_model_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_acc, base_modality_acc = test_model(base_model, test_loader, device, "Base")
        results['base'] = {'total acc': base_acc, 'modality acc': base_modality_acc}
    else:
        print(f"cannot find base modal: {base_model_path}")

    
    if os.path.exists(attention_model_path):
        print(f"\nupload attention model: {attention_model_path}")
        attention_model = AttentionModalityClassifier().to(device)
        checkpoint = torch.load(attention_model_path, map_location=device)
        attention_model.load_state_dict(checkpoint['model_state_dict'])
        attention_acc, attention_modality_acc = test_model(attention_model, test_loader, device, "Attention")
        results['attention model'] = {'total acc': attention_acc, 'modality acc': attention_modality_acc}
    else:
        print(f"cannot find attention model: {attention_model_path}")

    # results
    if results:
        print("\n=== comparison ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"total acc: {100.*metrics['total acc']:.2f}%")
            print("modality acc:")
            for modality, acc in metrics['modality acc'].items():
                print(f"  {modality}: {100.*acc:.2f}%")

        # 可视化对比
        if len(results) > 1:
            modalities = list(next(iter(results.values()))['modality acc'].keys())
            model_names = list(results.keys())

            # 准备数据
            overall_accs = [results[model]['total acc'] * 100 for model in model_names]
            modality_accs = {modality: [results[model]['modality acc'][modality] * 100
                                      for model in model_names]
                           for modality in modalities}

            
            fig, ax = plt.subplots(figsize=(12, 8))
            bar_width = 0.15
            index = np.arange(len(modalities) + 1)  # +1 for overall

            
            for i, model_name in enumerate(model_names):
                offset = (i - len(model_names)/2 + 0.5) * bar_width
                ax.bar(index[0] + offset, overall_accs[i], bar_width,
                     label=model_name)

            
            for i, modality in enumerate(modalities):
                for j, model_name in enumerate(model_names):
                    offset = (j - len(model_names)/2 + 0.5) * bar_width
                    ax.bar(index[i+1] + offset, modality_accs[modality][j],
                         bar_width)

            ax.set_ylabel('acc (%)')
            ax.set_title('comparsion on slake datasets')
            ax.set_xticks(index)
            ax.set_xticklabels(['total'] + list(modalities))
            ax.legend()

            plt.savefig('/content/drive/MyDrive/output/slake_comparison.png')
            plt.show()

if __name__ == "__main__":
    main()





#===========================================mri augmentation======================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import re
from google.colab import drive
drive.mount('/content/drive')

def find_mri_sample(df):
    """find a mri sample"""
    for idx, row in df.iterrows():
        if isinstance(row['answer'], str) and 'mri' in row['answer'].lower():
            return row
    return None

def decode_image(image_data):
    """decode"""
    try:
        if isinstance(image_data, dict):
            if 'bytes' in image_data:
                bytes_data = image_data['bytes']
                if isinstance(bytes_data, str):
                    bytes_data = bytes_data.encode('latin1')
                img = Image.open(io.BytesIO(bytes_data))
                return np.array(img)
            elif 'array' in image_data:
                return np.array(image_data['array'])
        return np.array(image_data)
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        raise

def process_attention_maps(image, attention_model):
    """attention figure"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attention_model = attention_model.to(device)
    attention_model.eval()

    # tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    
    attention_maps = []
    with torch.no_grad():
        x = attention_model.backbone.conv1(image_tensor)
        x = attention_model.backbone.bn1(x)
        x = attention_model.backbone.relu(x)
        x = attention_model.backbone.maxpool(x)

        
        attentions = [
            attention_model.attention1,
            attention_model.attention2,
            attention_model.attention3,
            attention_model.attention4
        ]

        for i, (layer, attention) in enumerate([
            (attention_model.backbone.layer1, attentions[0]),
            (attention_model.backbone.layer2, attentions[1]),
            (attention_model.backbone.layer3, attentions[2]),
            (attention_model.backbone.layer4, attentions[3])
        ]):
            x = layer(x)
             
            anatomy_attention = attention.anatomy_attention(x)
            texture_attention = attention.texture_attention(x)

           
            attention_map = (anatomy_attention * texture_attention).mean(1, keepdim=True) # Keep the channel dimension
            attention_map = nn.functional.interpolate(
                attention_map, size=(224, 224), mode='bilinear', align_corners=False
            )
            attention_maps.append(attention_map.squeeze().cpu().numpy())

            return attention_maps

def apply_mri_augmentations(image):
    """mri augmentation"""
   
    base_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    
    mri_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15)
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        )
    ])

    
    pil_image = Image.fromarray(image)
    base_augmented = base_aug(pil_image)
    mri_augmented = mri_aug(pil_image)

    return np.array(base_augmented), np.array(mri_augmented)

def visualize_mri_processing(df, attention_model):
    """visible samples"""
    
    mri_sample = find_mri_sample(df)
    if mri_sample is None:
        print("No MRI sample found!")
        return

    
    original_image = decode_image(mri_sample['image'])
    if len(original_image.shape) == 2:
        original_image = np.stack([original_image] * 3, axis=-1)

   
    attention_maps = process_attention_maps(original_image, attention_model)

    
    base_augmented, mri_augmented = apply_mri_augmentations(original_image)

   
    plt.figure(figsize=(15, 10))

   
    plt.subplot(3, 3, 1)
    plt.imshow(original_image)
    plt.title('Original MRI Image')
    plt.axis('off')

   
    for i, attention_map in enumerate(attention_maps):
        plt.subplot(3, 3, i + 2)
        plt.imshow(attention_map, cmap='jet')
        plt.title(f'Attention Layer {i+1}')
        plt.axis('off')

   
    plt.subplot(3, 3, 7)
    plt.imshow(original_image)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(base_augmented)
    plt.title('Base Augmentation')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(mri_augmented)
    plt.title('MRI Specific Augmentation')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

   
    print("\nQuestion:", mri_sample['question'])
    print("\nAnswer:", mri_sample['answer'])

def main():
    
    train_path = '/content/drive/MyDrive/RADdataset/train-00000-of-00001-eb8844602202be60.parquet'
    df = pd.read_parquet(train_path)

    
    attention_model = AttentionModalityClassifier()

    
    model_path = 'attention_fixmatch_best.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        attention_model.load_state_dict(checkpoint['model_state_dict'])

    
    visualize_mri_processing(df, attention_model)

if __name__ == "__main__":
    main()



#----

























