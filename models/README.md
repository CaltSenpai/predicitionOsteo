# OsteoAI Model Directory

## Required Model File
Place your trained osteoarthritis detection model here with the exact filename:
- `osteoarthritis_model.h5`

## Model Requirements
- **Input Shape:** (224, 224, 3) - RGB images resized to 224x224 pixels
- **Output:** Single probability value between 0 and 1
- **Format:** Keras/TensorFlow H5 file
- **Architecture:** CNN-based classification model

## Model Training Notes
Ensure your model was trained with:
- Similar image preprocessing (resize to 224x224, normalize 0-1)
- Binary classification output
- Medical image data (X-rays, MRIs, CT scans)

## Updating Model Requirements
If your model has different input requirements, update the `preprocess_image()` function in `app.py`:
- Change image dimensions in `image.resize((width, height))`
- Adjust normalization if needed
- Modify output interpretation logic

## Model Performance
Document your model's performance metrics here:
- Accuracy: ____%
- Precision: ____%
- Recall: ____%
- Training Dataset Size: ____
- Validation Method: ____

---
Last Updated: 2025-07-03