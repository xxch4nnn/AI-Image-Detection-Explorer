# 🔍 AI-Generated Image Detection Explorer

## Project Description

This interactive Streamlit web application provides a comprehensive exploration of AI-generated images using the **DiffusionDB** dataset from Hugging Face. The application enables users to visualize, analyze, and understand AI-generated imagery through multiple interactive features including dataset exploration, statistical analysis, and machine learning classification.

### What the App Does

The AI-Generated Image Detection Explorer offers five main functionalities:

1. **Home Dashboard**: Provides an overview of the dataset, quick statistics, and sample images
2. **Dataset Explorer**: Interactive browsing with filters for sampling methods, generation steps, and metadata viewing
3. **Statistics & Insights**: Comprehensive visualizations including distribution charts, pie charts, and key metrics
4. **Random Gallery**: Dynamic image gallery that displays random selections from the dataset
5. **ML Classification**: Pre-trained ResNet18 model for image classification with file upload capability

### How to Use the App

#### Navigation
- Use the tab menu to switch between different sections
- Each tab offers unique functionality for dataset exploration

#### Dataset Explorer
1. Select filters from the dropdown menus (Sampler, Generation Steps)
2. Adjust the number of images to display using the slider
3. View detailed metadata by clicking "View Details" under each image
4. Download the complete metadata as CSV

#### Statistics Section
- Review key metrics displayed in colorful cards
- Explore distribution charts showing sampler methods and generation steps
- Read insights about dataset characteristics

#### Random Gallery
- Click "Show Random Images" to view a fresh random selection
- Expand metadata sections to see generation parameters
- Refresh multiple times to explore dataset diversity

#### ML Classification
- Upload your own images (JPG, PNG format)
- View classification results with confidence scores
- Try classification on random dataset images
- Explore top-5 predictions with probability percentages

### Key Insights Users Can Gain

1. **Understanding AI Image Generation**:
   - Learn how different sampling methods affect image quality
   - Understand the role of generation steps in image creation
   - Explore the relationship between prompts and generated outputs

2. **Dataset Characteristics**:
   - Distribution of sampling methods (DDIM, PLMS, K-LMS, etc.)
   - Common generation parameters and their frequencies
   - Diversity of prompts and artistic styles

3. **Machine Learning Application**:
   - Practical demonstration of pre-trained model usage
   - Understanding confidence scores and classification results
   - Hands-on experience with image classification

4. **Data Analysis Skills**:
   - Interactive filtering and data exploration
   - Statistical visualization interpretation
   - Metadata analysis and pattern recognition

## Technical Requirements Met

### ✅ Dataset Selection & Loading
- **Dataset**: DiffusionDB (2m_random_1k split) from Hugging Face
- **Loading**: Uses `datasets.load_dataset()` with caching
- **Metadata**: Cleaned table with prompts, seeds, samplers, steps, CFG, dimensions
- **Visualization**: Sample grid and interactive filters

### ✅ Summary Statistics & Insights
- **Counts**: Total images, unique samplers, average steps
- **Charts**: Pie chart for sampler distribution, histogram for generation steps
- **Insights**: Key findings about dataset composition and quality

### ✅ Interactive Features
- **Image Gallery**: Updates based on sampler and step filters
- **Random Button**: "Show Random Images" displays 12 random examples
- **Metadata Display**: Expanders showing detailed information
- **Download**: CSV export functionality

### ✅ Pre-trained ML Model (Bonus)
- **Model**: ResNet18 from TorchVision (ImageNet pre-trained)
- **Functionality**: Image classification with confidence scores
- **File Upload**: Custom image testing capability
- **Display**: Top-5 predictions with probability bars

### ✅ Organization
- **Tabs**: 5 organized sections (Home, Dataset, Statistics, Gallery, Model)
- **Layout**: Wide layout with responsive columns
- **Styling**: Custom CSS for enhanced visual appeal

## Deployment Instructions

### Option 1: Hugging Face Spaces (Recommended)

1. **Create a Hugging Face Account**
   - Visit https://huggingface.co/
   - Sign up for a free account

2. **Create a New Space**
   - Click on your profile → "New Space"
   - Name: "ai-image-detection-explorer"
   - License: Select appropriate license
   - SDK: Choose "Streamlit"
   - Hardware: CPU Basic (free tier)

3. **Upload Files**
   - Upload `app.py` (the main Streamlit application)
   - Upload `requirements.txt`
   - Commit the files

4. **Wait for Build**
   - Space will automatically build and deploy
   - Usually takes 5-10 minutes
   - App will be live at: `https://huggingface.co/spaces/YOUR-USERNAME/ai-image-detection-explorer`

### Option 2: Streamlit Cloud

1. **Prepare GitHub Repository**
   ```bash
   git init
   git add app.py requirements.txt
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin YOUR-GITHUB-REPO-URL
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Visit https://share.streamlit.io/
   - Click "New app"
   - Select your GitHub repository
   - Main file path: `app.py`
   - Click "Deploy"

3. **Access Your App**
   - App will be available at: `https://YOUR-APP-NAME.streamlit.app/`

## Local Development

### Setup
```bash
# Clone or download the project
git clone YOUR-REPO-URL
cd your-project-folder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## File Structure

```
project/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore file (optional)
```

## Dataset Information

- **Name**: DiffusionDB (2m_random_1k split)
- **Source**: Hugging Face Datasets
- **Type**: AI-Generated Images (Stable Diffusion)
- **Size**: 1,000 images with complete metadata
- **Features**:
  - Prompts used for generation
  - Sampling methods (DDIM, PLMS, K-LMS, etc.)
  - Generation steps
  - CFG scale values
  - Image dimensions
  - Random seeds

## Technologies Used

- **Streamlit**: Web application framework
- **Hugging Face Datasets**: Dataset loading and management
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **PyTorch & TorchVision**: Deep learning and pre-trained models
- **Pillow (PIL)**: Image processing
- **NumPy**: Numerical operations

## Features Checklist

- [x] Dataset loaded from Hugging Face
- [x] Cleaned metadata table with download option
- [x] Image grid with interactive filters
- [x] Multiple filtering options (sampler, steps, count)
- [x] Summary statistics with metrics
- [x] Bar/pie charts for distributions
- [x] Interactive gallery with filters
- [x] Random image button functionality
- [x] Metadata expanders
- [x] Tab-based organization (5 tabs)
- [x] Pre-trained ML model (ResNet18)
- [x] File uploader for custom images
- [x] Classification with confidence scores
- [x] Top-5 predictions display

## Troubleshooting

### Common Issues

1. **Dataset Loading Slow**
   - First load caches the dataset
   - Subsequent loads are much faster
   - Ensure stable internet connection

2. **Model Loading Errors**
   - PyTorch installation required
   - Check CUDA compatibility for GPU
   - Use CPU version if GPU unavailable

3. **Memory Issues**
   - Reduce number of images displayed
   - Use smaller dataset split
   - Close unnecessary applications

### Performance Optimization

- Dataset is cached after first load
- Metadata preparation limited to 1000 samples
- Efficient filtering with pandas
- Image lazy loading

## Future Enhancements

Possible improvements for future versions:

1. Real vs. AI-generated classification model
2. Multiple dataset support
3. Advanced filtering options
4. Image similarity search
5. Batch image processing
6. Export filtered results
7. Custom model fine-tuning
8. Real-time generation monitoring

## Credits & Acknowledgments

- **Dataset**: DiffusionDB by Hugging Face
- **Framework**: Streamlit
- **Models**: PyTorch and TorchVision
- **Visualization**: Plotly

## License

This project is for educational purposes. Please check individual component licenses:
- DiffusionDB dataset license
- PyTorch license
- Streamlit license

## Contact & Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Contact via Hugging Face Spaces
- Check Streamlit documentation: https://docs.streamlit.io/

---

**Built with ❤️ for AI and Machine Learning Education**
