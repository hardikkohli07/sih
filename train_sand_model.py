import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Step 1: Load CSV & Process
# --------------------------
csv_file = "usace_1024_aug_dry_set1_2_3_4_5_aug2021.csv"
df = pd.read_csv(csv_file)

# Quantile-based binning into 5 balanced categories
labels = [0, 1, 2, 3, 4]  # numeric labels
df['class'], bins = pd.qcut(df['mean'], q=5, labels=labels, retbins=True)

# Human-readable class names
class_names = ["Very Fine", "Fine", "Medium", "Coarse", "Very Coarse"]

print("✅ Class distribution:\n", df['class'].value_counts().sort_index())
print("✅ Bin edges (mm):", bins)

# --------------------------
# Step 2: Dataset Class
# --------------------------
class SandDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]['files']
        rel_path = rel_path.replace("sandsnap_images/", "")  # clean path
        img_path = os.path.join(self.img_dir, rel_path)

        image = Image.open(img_path).convert("RGB")
        label = int(self.df.iloc[idx]['class'])
        if self.transform:
            image = self.transform(image)
        return image, label

# --------------------------
# Step 3: Transforms
# --------------------------
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------
# Step 4: Split Dataset
# --------------------------
img_dir = "sand_images/"
dataset = SandDataset(df, img_dir=img_dir, transform=train_tfms)

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# Apply val_tfms to val/test
val_ds.dataset.transform = val_tfms
test_ds.dataset.transform = val_tfms

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# --------------------------
# Step 5: Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # 5 classes
model = model.to(device)

# Compute class weights
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(df['class']),
    y=df['class']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("✅ Class Weights:", class_weights)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# --------------------------
# Step 6: Training Loop
# --------------------------
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_acc = correct / total
    
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
    
    scheduler.step()

# --------------------------
# Step 7: Test & Report
# --------------------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved as confusion_matrix.png")

# --------------------------
# Step 8: Save Model
# --------------------------
torch.save(model.state_dict(), "sand_classifier_resnet18.pth")
print("✅ Model saved as sand_classifier_resnet18.pth")
