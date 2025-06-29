# -*- coding: utf-8 -*-
"""Pretrined Fixmatch with RAD ＋ medical attention
"""

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
import cv2
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
    """using question/answer infor to extract modality"""
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

def load_rad_data(train_path, test_path):
    """combine"""
    print("\n=== Loading RAD Dataset ===")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Combined dataset size: {len(combined_df)}")

    print("\n=== Text Analysis ===")
    question_lengths = combined_df['question'].str.len().describe()
    answer_lengths = combined_df['answer'].str.len().describe()

    print("\nQuestion length statistics:")
    print(question_lengths)
    print("\nAnswer length statistics:")
    print(answer_lengths)

    print("\n=== Data Quality Check ===")
    null_counts = combined_df.isnull().sum()
    print("\nNull values in each column:")
    print(null_counts)

    print("\n=== Initial Modality Analysis ===")
    question_modalities = combined_df['question'].apply(extract_modality)
    answer_modalities = combined_df['answer'].apply(extract_modality)

    combined_modalities = question_modalities.combine_first(answer_modalities)
    modality_counts = combined_modalities.value_counts()

    print("\nModality distribution in text:")
    print(modality_counts)
    print("\nPercentage of samples with modality information:")
    print(f"{(len(combined_modalities.dropna()) / len(combined_df)) * 100:.2f}%")

    return combined_df

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


        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1, dilation=1),
            nn.Conv2d(in_channels, in_channels//4, 3, padding=2, dilation=2),
            nn.Conv2d(in_channels, in_channels//4, 3, padding=4, dilation=4)
        ])

        self.channel_adjust = nn.Conv2d(in_channels//4*3, in_channels, 1)

    def forward(self, x):
        anatomy_weight = self.anatomy_attention(x)
        texture_weight = self.texture_attention(x)

        multi_scale_features = []
        for conv in self.multi_scale:
            multi_scale_features.append(conv(x))
        multi_scale_feat = torch.cat(multi_scale_features, dim=1)
        multi_scale_feat = self.channel_adjust(multi_scale_feat)

        enhanced = x * anatomy_weight * texture_weight
        return enhanced + multi_scale_feat

class ModalityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.attention1 = MedicalModalityAttention(256)
        self.attention2 = MedicalModalityAttention(512)
        self.attention3 = MedicalModalityAttention(1024)
        self.attention4 = MedicalModalityAttention(2048)

        in_features = self.backbone.fc.in_features
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

# data augmentation
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


        self.mri_aug = transforms.Compose([
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

        self.modality_to_idx = {
            'mri': 0,
            'ct': 1,
            'xray': 2
        }

        print("\n=== Dataset Statistics ===")
        total_samples = len(df)
        valid_count = 0
        modality_counts = {'mri': 0, 'ct': 0, 'xray': 0}

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            question_modality = extract_modality(row['question'])
            answer_modality = extract_modality(row['answer'])

            modality = question_modality or answer_modality

            if modality in ['mri', 'ct', 'xray']:
                self.valid_samples.append({
                    'image': row['image'],
                    'modality': modality,
                    'index': idx
                })
                modality_counts[modality] += 1
                valid_count += 1
            else:
                self.unlabeled_samples.append({
                    'image': row['image'],
                    'index': idx
                })

        if self.is_labeled:
            print(f"\nTotal samples: {total_samples}")
            print(f"Labeled samples: {valid_count} ({valid_count/total_samples*100:.2f}%)")
            print(f"Unlabeled samples: {total_samples - valid_count} ({(total_samples-valid_count)/total_samples*100:.2f}%)")

            print("\n=== Class Distribution ===")
            for modality, count in modality_counts.items():
                print(f"{modality.upper()}: {count} samples ({count/valid_count*100:.2f}%)")

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
            print(f"\nError decoding image:")
            print(f"Image data type: {type(image_data)}")
            if isinstance(image_data, dict):
                print(f"Dict keys: {image_data.keys()}")
            print(f"Error: {str(e)}")
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
                if sample['modality'] == 'mri':
                    image_tensor = self._process_image(image_data, aug_type='mri')
                else:
                    image_tensor = self._process_image(image_data, aug_type='base')
            else:
                image_tensor = self._process_image(image_data, aug_type='test')

            return image_tensor, label
        else:
            sample = self.unlabeled_samples[idx]
            image_data = sample['image']

            weak_aug = self._process_image(image_data, aug_type='base')
            strong_aug = self._process_image(image_data, aug_type='mri')

            return weak_aug, strong_aug

    def _process_image(self, image_data, aug_type='base'):
        try:
            image_array = self.decode_image(image_data)

            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]

            image = Image.fromarray(image_array.astype('uint8'))

            if aug_type == 'mri':
                return self.augmentation.mri_aug(image)
            elif aug_type == 'test':
                return self.augmentation.test_transform(image)
            else:
                return self.augmentation.base_aug(image)

        except Exception as e:
            print(f"\nError processing image:")
            print(f"Error: {str(e)}")
            raise

# train & val
def train_epoch(model, train_loader, unlabeled_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    unlabeled_iter = iter(unlabeled_loader)

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        try:
            unlabeled_weak, unlabeled_strong = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_weak, unlabeled_strong = next(unlabeled_iter)

        unlabeled_weak = unlabeled_weak.to(device)
        unlabeled_strong = unlabeled_strong.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        with torch.no_grad():
            pseudo_outputs = model(unlabeled_weak)
            pseudo_probs = F.softmax(pseudo_outputs, dim=1)
            max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
            mask = max_probs.ge(0.95)

        strong_outputs = model(unlabeled_strong)

        sup_loss = criterion(outputs, labels)
        unsup_loss = (F.cross_entropy(strong_outputs, pseudo_labels,
                                    reduction='none') * mask).mean()

        loss = sup_loss + 0.5 * unsup_loss

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()

        pbar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    classification_metrics = classification_report(
        all_labels,
        all_preds,
        target_names=['mri', 'ct', 'xray'],
        output_dict=True
    )

    return (total_loss / len(test_loader),
            correct / total,
            classification_metrics,
            all_preds,
            all_labels)

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['MRI', 'CT', 'X-ray'],
                yticklabels=['MRI', 'CT', 'X-ray'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def finetune_model():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading RAD dataset...")
    train_path = '/content/drive/MyDrive/RADdataset/train-00000-of-00001-eb8844602202be60.parquet'
    test_path = '/content/drive/MyDrive/RADdataset/test-00000-of-00001-e5bc3d208bb4deeb.parquet'
    rad_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    combined_df = pd.concat([rad_df, test_df], ignore_index=True)


    labeled_dataset = RADDataset(combined_df, is_labeled=True, mode='train')
    unlabeled_dataset = RADDataset(combined_df, is_labeled=False)
    test_dataset = RADDataset(combined_df, is_labeled=True, mode='test')

    # split train & val
    total_size = len(labeled_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
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


    model = ModalityClassifier(num_classes=3)
    model_path = '/content/drive/MyDrive/modelparasave/slake_modality_model_improved.pth'

    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained model loaded successfully")
    else:
        print("No pretrained model found, starting from scratch")

    model = model.to(device)


    class_weights = torch.FloatTensor([1.5, 1.0, 1.0]).to(device)  # MRI weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)


    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.attention1.parameters(), 'lr': 1e-4},
        {'params': model.attention2.parameters(), 'lr': 1e-4},
        {'params': model.attention3.parameters(), 'lr': 1e-4},
        {'params': model.attention4.parameters(), 'lr': 1e-4},
        {'params': model.modality_head.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'modality_metrics': []
    }

    # begin training
    num_epochs = 10
    best_acc = 0
    print("\nStarting training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # train
        train_loss, train_acc = train_epoch(
            model, train_loader, unlabeled_loader, criterion, optimizer, device
        )

        # val
        val_loss, val_acc, metrics, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        # update
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['modality_metrics'].append(metrics)

        # print epoches
        print(f"\nTraining Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print("\nDetailed metrics for each modality:")
        for modality in ['mri', 'ct', 'xray']:
            print(f"{modality}:")
            print(f"  Precision: {metrics[modality]['precision']:.4f}")
            print(f"  Recall: {metrics[modality]['recall']:.4f}")
            print(f"  F1-score: {metrics[modality]['f1-score']:.4f}")

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"\nSaving new best model with validation accuracy: {best_acc:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, model_path)


        plot_confusion_matrix(val_labels, val_preds)


    plot_training_history(history)

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    finetune_model()










