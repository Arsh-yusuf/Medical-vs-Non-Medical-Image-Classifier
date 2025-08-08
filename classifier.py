import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import fitz  # PyMuPDF
import io, os, requests
from bs4 import BeautifulSoup
import pandas as pd


# --------- Load model ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("medical_vs_nonmedical_efficientnet_finetuned.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ["medical", "non-medical"]

# --------- Predict single image ---------
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    return class_names[pred]

# --------- Extract images from PDF ---------
def extract_images_from_pdf(pdf_path, output_folder="pdf_images"):
    os.makedirs(output_folder, exist_ok=True)
    pdf = fitz.open(pdf_path)
    img_paths = []
    for page_num, page in enumerate(pdf):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_path = f"{output_folder}/page{page_num+1}_img{img_index+1}.jpg"
            img.save(img_path)
            img_paths.append(img_path)
    return img_paths

# --------- Extract images from URL ---------
def extract_images_from_url(url, output_folder="web_images"):
    os.makedirs(output_folder, exist_ok=True)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")
    img_paths = []
    for i, tag in enumerate(img_tags):
        img_url = tag.get("src")
        if img_url and img_url.startswith("http"):
            try:
                img_data = requests.get(img_url).content
                img_path = f"{output_folder}/image_{i+1}.jpg"
                with open(img_path, "wb") as f:
                    f.write(img_data)
                img_paths.append(img_path)
            except:
                continue
    return img_paths

# --------- Classify images ---------
def classify_images(image_paths, output_csv="results.csv"):
    results = []
    for img_path in image_paths:
        label = predict_image(img_path)
        results.append({"image": img_path, "prediction": label})
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(df)
    return df

def classify_from_input(input_path_or_url, is_pdf=True):
    if is_pdf:
        imgs = extract_images_from_pdf(input_path_or_url)
    else:
        try:
            # Try downloading as an image first
            response = requests.get(input_path_or_url, stream=True)
            content_type = response.headers.get("Content-Type", "")
            
            if "image" in content_type:  # It's a direct image
                img_path = "downloaded_image.jpg"
                with open(img_path, "wb") as f:
                    f.write(response.content)
                imgs = [img_path]
            else:
                # Otherwise treat as a webpage
                imgs = extract_images_from_url(input_path_or_url)
        except Exception as e:
            print(f"Error fetching URL: {e}")
            imgs = []
    
    return classify_images(imgs) if imgs else pd.DataFrame(columns=["image", "prediction"])



# --------- Example Usage ---------
if __name__ == "__main__":
    # PDF example
    # classify_from_input(r"pdf_images\animal.pdf", is_pdf=True)
    
    # URL example
    classify_from_input("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ9_dxLutYcexRnviiHxITKirn2-u4PAV53Fg&s", is_pdf=False)
