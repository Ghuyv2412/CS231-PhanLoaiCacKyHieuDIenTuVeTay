import random
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Thiết bị:', device)


DATA_ROOT = Path(
    "/kaggle/input/datasets/ayush02102001/circuit-component-analysis/CktComDb-v1.oa"
)

print("Dataset tồn tại:", DATA_ROOT.exists())

for path in DATA_ROOT.iterdir():
    print(path.name)

assert DATA_ROOT.exists(), 'Chưa Add Input dataset JUHCCR-v1-A vào notebook Kaggle.'

pairs = []
for folder in DATA_ROOT.rglob('*'):
    if folder.is_dir() and folder.name.lower().startswith('train'):
        tests = [p for p in folder.parent.iterdir()
                 if p.is_dir() and p.name.lower().startswith('test')]
        num_classes = sum(p.is_dir() for p in folder.iterdir())
        if tests and num_classes >= 10:
            pairs.append((folder, tests[0]))

for i, (tr, te) in enumerate(pairs):
    print(f'{i}: TRAIN = {tr}\n   TEST  = {te}')

assert pairs, 'Không tìm thấy cặp Train/Test. Hãy xem lại cấu trúc ở Cell 2.'
DATASET_ID = 0  # Đổi số này nếu bản .a không ở vị trí 0
TRAIN_DIR, TEST_DIR = pairs[DATASET_ID]
print('Đang sử dụng:', TRAIN_DIR, '\n', TEST_DIR)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


import re

# 1. Đọc dữ liệu
full_train = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

classes = full_train.classes

# 2. Tên 20 loại linh kiện theo JUHCCR-v1
class_names = {
    1: "AC Source",
    2: "Ammeter",
    3: "AND Gate",
    4: "Capacitor",
    5: "DC Source",
    6: "Ground",
    7: "Inductor",
    8: "NAND Gate",
    9: "NOR Gate",
    10: "NOT Gate",
    11: "NPN Transistor",
    12: "OR Gate",
    13: "PN Junction Diode",
    14: "PNP Transistor",
    15: "Power Supply",
    16: "Resistor",
    17: "Switch",
    18: "Transformer",
    19: "Voltmeter",
    20: "Zener Diode"
}

# 3. Kiểm tra số lớp
assert len(classes) == 20, "Dataset phải có 20 lớp."
assert full_train.class_to_idx == test_dataset.class_to_idx

# 4. Ánh xạ tên thư mục sang tên linh kiện
display_names = []

for folder in classes:
    match = re.match(r"^(\d+)", folder)

    if match is None:
        raise ValueError(f"Không tìm thấy mã lớp trong: {folder}")

    class_id = int(match.group(1))
    display_names.append(class_names[class_id])

# 5. Hiển thị kết quả
print("Số lớp:", len(classes))
print("Số ảnh Train:", len(full_train))
print("Số ảnh Test:", len(test_dataset))

print("\nDanh sách 20 classes:")
for i, name in enumerate(display_names):
    print(f"{i}: {name}")

plt.figure(figsize=(10, 6))
for i in range(6):
    image_path, label = full_train.samples[i * max(1, len(full_train) // 6)]
    image = Image.open(image_path).convert('RGB')
    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(classes[label], fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.show()

groups = []
for file_path, label in full_train.samples:
    stem = Path(file_path).stem
    match = re.search(r'(\d+)$', stem)
    source_id = match.group(1) if match else stem
    groups.append(f'{label}_{source_id}')

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(splitter.split(full_train, groups=groups))

train_dataset = Subset(full_train, train_idx)
val_dataset = Subset(full_train, val_idx)
print('Train:', len(train_dataset))
print('Validation:', len(val_dataset))
print('Test:', len(test_dataset))
print('Số group chung train/val:', len(set(np.array(groups)[train_idx]) &
                                   set(np.array(groups)[val_idx])))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)
print('Lớp phân loại mới:', model.fc)

EPOCHS = 8
LEARNING_RATE = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

def run_epoch(loader, training):
    model.train() if training else model.eval()
    if training:
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()

    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

history = {'train_acc': [], 'val_acc': []}
best_val_acc = 0.0
SAVE_PATH = '/kaggle/working/resnet18_juhccr_v1a.pth'

for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    print(f'Epoch {epoch+1}/{EPOCHS} | '
          f'Train loss: {train_loss:.4f}, acc: {train_acc:.2%} | '
          f'Val loss: {val_loss:.4f}, acc: {val_acc:.2%}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model': model.state_dict(), 'classes': classes}, SAVE_PATH)

print('Validation accuracy tốt nhất:', f'{best_val_acc:.2%}')
print('Đã lưu:', SAVE_PATH)

plt.figure(figsize=(7, 4))
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('ResNet-18: Training vs Validation')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


checkpoint = torch.load(SAVE_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model'])
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu().tolist()

        y_pred.extend(predictions)
        y_true.extend(labels.tolist())

accuracy = (np.array(y_true) == np.array(y_pred)).mean()

print('TEST ACCURACY:', f'{accuracy:.2%}')

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=display_names,
    zero_division=0,
    digits=4
))

fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    labels=list(range(len(classes))),
    display_labels=classes,
    xticks_rotation=90, cmap='Blues', ax=ax, colorbar=False
)
plt.title('Confusion Matrix - ResNet-18')
plt.tight_layout()
plt.show()

IMAGE_PATH = test_dataset.samples[0][0]  # Có thể thay bằng ảnh tự tải lên
image = Image.open(IMAGE_PATH).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    scores = model(image_tensor)
    probabilities = torch.softmax(scores, dim=1)[0]
    predicted_id = probabilities.argmax().item()

plt.imshow(image)
plt.axis('off')
plt.title(f'Dự đoán: {classes[predicted_id]} '
          f'({probabilities[predicted_id].item():.1%})')
plt.show()