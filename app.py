# Reloading app to detect new model 
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, render_template, request, url_for

# ─────────────────────────────────────────────
# App & Config
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB limit
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}

# ─────────────────────────────────────────────
# Load Pre-trained ResNet-18 Model
# ─────────────────────────────────────────────
print("[INFO] Loading pre-trained ResNet-18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
model.eval()
print("[INFO] Pre-trained ResNet-18 model loaded successfully.")

# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict(image_path: str) -> str:
    """Return 'Cat' or 'Dog' for the given image path."""
    transform = get_transform()

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(tensor)                        # (1, 1000) logits
        probs   = torch.softmax(outputs, dim=1)        # (1, 1000) probabilities

    # Dogs: classes 151 to 268 in ImageNet
    # Cats: classes 281 to 287 in ImageNet
    dog_prob = probs[0, 151:269].sum().item()
    cat_prob = probs[0, 281:288].sum().item()

    if dog_prob > cat_prob:
        return "Dog"
    else:
        return "Cat"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    prediction  = None
    image_path  = None
    error       = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "No file selected. Please upload an image."
        elif not allowed_file(file.filename):
            error = "Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP."
        else:
            filename   = file.filename
            save_path  = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            image_path = url_for("static", filename=f"uploads/{filename}")

            try:
                prediction = predict(save_path)
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path,
        error=error,
    )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
