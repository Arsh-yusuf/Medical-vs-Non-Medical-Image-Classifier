import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import fitz  # PyMuPDF
import io, os, requests
from bs4 import BeautifulSoup
import pandas as pd
import torch.nn.functional as F

# -----------------------
# Load Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

modelA = load_model("model_1.pth")
modelB = load_model("model_2.pth")
class_names = ["medical", "non-medical"]

# -----------------------
# Prediction Function
# -----------------------
def predict_image_ensemble(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outA = F.softmax(modelA(img_t), dim=1)
        outB = F.softmax(modelB(img_t), dim=1)
        avg_out = (outA + outB) / 2
        _, pred = torch.max(avg_out, 1)
    
    return class_names[pred], avg_out.cpu().numpy()

# -----------------------
# PDF Image Extraction
# -----------------------
def extract_images_from_pdf(pdf_file):
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    img_paths = []
    for page_num, page in enumerate(pdf):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_path = f"pdf_img_{page_num+1}_{img_index+1}.jpg"
            img.save(img_path)
            img_paths.append(img_path)
    return img_paths
# -----------------------
# URL Image Extraction
# -----------------------
def extract_images_from_url(url):
    img_paths = []
    try:
        # Check if it's a direct image
        response = requests.get(url, stream=True)
        content_type = response.headers.get("Content-Type", "")
        if "image" in content_type:
            img_path = "downloaded_image.jpg"
            with open(img_path, "wb") as f:
                f.write(response.content)
            img_paths.append(img_path)
        else:
            # Treat as webpage
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            for i, tag in enumerate(soup.find_all("img")):
                img_url = tag.get("src")
                if img_url and img_url.startswith("http"):
                    try:
                        img_data = requests.get(img_url).content
                        img_path = f"web_image_{i+1}.jpg"
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        img_paths.append(img_path)
                    except:
                        continue
    except:
        pass
    return img_paths

# -----------------------
# Streamlit UI
# -----------------------
st.title("Medical vs Non-Medical Image Classifier")
st.write("Upload a PDF or enter a website URL to classify extracted images.")

option = st.radio("Select Input Type", ["PDF", "URL"])

if option == "PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        images = extract_images_from_pdf(uploaded_pdf)
        if images:
            results = []
            for img_path in images:
                label, conf = predict_image_ensemble(img_path)
                st.image(img_path, caption=f"Prediction: {label}", width=300)
                results.append({"image": img_path, "prediction": label})
        else:
            st.warning("No images found in PDF.")

elif option == "URL":
    url = st.text_input("Enter website or image URL")
    if url:
        images = extract_images_from_url(url)
        if images:
            results = []
            for img_path in images:
                label, conf = predict_image_ensemble(img_path)
                st.image(img_path, caption=f"Prediction: {label}", width=300)
                results.append({"image": img_path, "prediction": label})
        else:
            st.warning("No images found at this URL.")