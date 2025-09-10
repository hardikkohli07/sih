import torch
from torchvision import transforms, models
from PIL import Image
import sys
import os

# --------------------------
# Step 1: Config
# --------------------------
MODEL_PATH = "sand_classifier_resnet18.pth"
CLASS_NAMES = ["Very Fine", "Fine", "Medium", "Coarse", "Very Coarse"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Step 2: Define transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------
# Step 3: Load model
# --------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------
# Step 4: Predict function
# --------------------------
def predict(image_file):
    try:
        # If Flask FileStorage object, read stream
        if hasattr(image_file, "read"):
            # Reset file pointer to beginning
            image_file.seek(0)
            image = Image.open(image_file).convert("RGB")
        else:
            if not os.path.exists(image_file):
                raise FileNotFoundError(f"File not found: {image_file}")
            image = Image.open(image_file).convert("RGB")
        
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
        return pred.item(), CLASS_NAMES[pred.item()]
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None, None


# --------------------------
# Step 5: CLI Usage
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
    
    image_file = sys.argv[1]
    class_idx, label = predict(image_file)
    print(f"✅ Predicted class: {label}")
