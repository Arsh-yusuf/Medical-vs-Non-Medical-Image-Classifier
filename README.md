# Medical vs Non-Medical Image Classifier

## 📌 Overview
This project is a **deep learning pipeline** to classify images as either:
- **Medical** (X-rays, MRIs, CT scans, ultrasound, etc.)
- **Non-Medical** (landscapes, vehicles, architecture, animals, etc.)

The system can take input from:
- A **PDF file** (extracts all embedded images)
- A **website or image URL** (downloads and classifies all images)

The app uses an **ensemble of two EfficientNet-B0 models**:
- **Model 1**: Original model trained on diverse medical and non-medical images, strong on ultrasound and general medical imagery.
- **Model 2**: Fine-tuned model trained with additional MRI and X-ray images, plus challenging non-medical examples (cars, bikes).

Both models contribute to predictions via **probability averaging** for more robust classification.

---

## 📂 Dataset
**Medical images:**
- Brain MRIs (colored and grayscale)
- Knee X-rays
- Other X-rays and ultrasound images (original training)

**Non-medical images:**
- Landscapes, architecture
- Cats, dogs, animals
- Cars and bikes (additional fine-tuning data)

**Training data size:**
- ~7,400 images in the original training set  
- Fine-tuning data: 3,600 images (1,800 medical + 1,800 non-medical)

---

## ⚙️ Approach and Reasoning
1. **Base Model**: EfficientNet-B0 was chosen for its balance between accuracy and speed.
2. **Original Training**: Trained on a broad set of medical/non-medical images for generalization.
3. **Fine-Tuning**: Updated the model using new medical and non-medical examples where the original struggled (e.g., knee MRIs, vehicles).
4. **Ensemble Strategy**: Kept both the original and fine-tuned model and combined their predictions:
   - Extract softmax probabilities from both models
   - Average them
   - Take the class with the highest average probability
5. **Deployment**: Integrated into a **Streamlit app** for easy user interaction.  
   - Automatically extracts all images from PDFs or URLs
   - Classifies each image and displays the result

---

## 📊 Accuracy on Validation/Test Set
A small test set (~200 images, 100 medical + 100 non-medical) was used.

| Model                  | Accuracy | Notes |
|------------------------|----------|-------|
| Original Model         | ~99.78%     | Strong on ultrasound, some misses on knee MRIs |
| Fine-Tuned Model       | ~98%     | Strong on knee MRIs, weaker on ultrasound |
| **Ensemble (Final)**   | **98.89%**| Balanced performance across all modalities |

---

## ⚡ Performance & Efficiency
- **Model Size**: ~20 MB each (EfficientNet-B0)
- **Inference Time**: ~0.05–0.1s per image on GPU (slightly higher on CPU)
- **Memory Usage**: ~500MB RAM with both models loaded
- **Scalability**: Can process multiple images from a single PDF/URL in sequence

**Optimizations:**
- Used image resizing to `128x128` for faster inference
- Cached models in Streamlit to avoid reload delays
- Batched extraction and classification steps

---

## 🚀 Usage
### Install Dependencies
```bash
pip install streamlit torch torchvision pillow pymupdf requests beautifulsoup4 pandas

