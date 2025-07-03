# 🦴 OsteoAI - Osteoarthritis Detection Web App

A Flask-based web application that uses artificial intelligence to analyze medical images for osteoarthritis detection.

## 🌟 Features

- **📚 Educational Home Page** - Comprehensive information about osteoarthritis
- **🤖 AI-Powered Analysis** - Upload medical images for instant AI assessment
- **📊 Detailed Results** - Confidence levels and analysis timing
- **📱 Responsive Design** - Works seamlessly on desktop and mobile devices
- **🎯 H5 Model Support** - Direct integration with Keras/TensorFlow models
- **📄 Report Generation** - Download analysis reports
- **🔒 Secure Upload** - File validation and size limits

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd predicitionOsteo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your AI model**
   - Place your trained `.h5` model file in the `models/` directory
   - Rename it to `osteoarthritis_model.h5`

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
predicitionOsteo/
├── app.py                    # Flask application (main file)
├── requirements.txt          # Python dependencies
├── templates/               # HTML templates
│   ├── home.html           # Educational home page
│   └── predict.html        # Prediction interface
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css       # Main stylesheet
│   └── js/
│       └── prediction.js   # JavaScript functionality
├── models/                 # AI model storage
│   ├── osteoarthritis_model.h5  # Your trained model
│   └── README.md          # Model documentation
├── uploads/               # Temporary file storage
│   └── .gitkeep          # Git tracking file
└── README.md             # This documentation
```

## 🔧 Configuration

### Model Requirements
- **Input Size:** 224x224 pixels (RGB)
- **Output:** Single probability value (0-1)
- **Format:** Keras H5 file
- **Training:** Medical images (X-rays, MRIs, etc.)

### Customization Options
- **Image Size:** Modify `preprocess_image()` in `app.py`
- **File Size Limit:** Change `MAX_CONTENT_LENGTH` in `app.py`
- **Styling:** Edit `static/css/style.css`
- **Functionality:** Update `static/js/prediction.js`

## 🩺 Usage Instructions

### For Healthcare Professionals
1. Navigate to the Prediction page
2. Upload high-quality medical images (X-rays, MRI, CT scans)
3. Review AI analysis results
4. Download reports for documentation
5. **Always combine with clinical evaluation**

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Maximum size: 16MB

### Best Results Tips
- Use clear, high-resolution medical images
- Focus on joint areas (knees, hips, hands, spine)
- Ensure proper image orientation
- Avoid blurry or low-quality images

## ⚠️ Important Disclaimers

### Medical Disclaimer
This application is designed for **educational and research purposes only**. It should never be used as:
- A substitute for professional medical diagnosis
- The sole basis for treatment decisions
- A replacement for clinical examination

Always consult qualified healthcare providers for proper medical evaluation and treatment.

### AI Limitations
- Model accuracy depends on training data quality
- Results may vary with different image types
- False positives and negatives are possible
- Continuous validation against clinical standards is required

## 🛠️ Development

### Dependencies
- **Flask 2.3.3** - Web framework
- **TensorFlow 2.13.0** - AI model inference
- **Pillow 10.0.0** - Image processing
- **NumPy 1.24.3** - Numerical operations
- **Werkzeug 2.3.7** - WSGI utilities

### Adding New Features
1. **New routes:** Add to `app.py`
2. **UI changes:** Modify templates in `templates/`
3. **Styling:** Update `static/css/style.css`
4. **Client-side logic:** Edit `static/js/prediction.js`

### Testing
- Test with various image formats and sizes
- Validate model predictions against known cases
- Check responsive design on different devices
- Verify error handling with invalid inputs

## 📈 Performance Optimization

### Model Optimization
- Use model quantization for faster inference
- Implement model caching for repeated predictions
- Consider GPU acceleration for larger models

### Web App Optimization
- Implement image compression before upload
- Add progress indicators for long operations
- Use CDN for static assets in production
- Enable gzip compression

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
1. **Use production WSGI server** (Gunicorn, uWSGI)
2. **Set up reverse proxy** (Nginx, Apache)
3. **Configure SSL/HTTPS**
4. **Set environment variables**
5. **Implement proper logging**

### Environment Variables
```bash
export FLASK_ENV=production
export MODEL_PATH=/path/to/models/
export UPLOAD_FOLDER=/path/to/uploads/
```

## 📊 Model Performance

Update this section with your model's metrics:
- **Accuracy:** ____%
- **Precision:** ____%
- **Recall:** ____%
- **F1-Score:** ____%
- **Dataset Size:** _____ images
- **Validation Method:** _____

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📜 License

This project is for educational purposes. Ensure compliance with medical software regulations before clinical use.

## 📞 Support

For technical support or questions:
- Review documentation in each directory
- Check model requirements in `models/README.md`
- Validate file formats and sizes
- Ensure proper model placement

---

**Built with ❤️ for advancing healthcare technology**

*Last updated: July 3, 2025*