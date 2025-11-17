# 🔍 AI-Generated Image Detection Explorer

## Project Description

This interactive Streamlit web application provides a comprehensive exploration of real and AI-generated images using the **CIFAKE** dataset from Hugging Face. The application enables users to visualize, analyze, and understand AI-generated imagery through multiple interactive features.

### What the App Does

The AI-Generated Image Detection Explorer offers four main functionalities:

1. **Home Dashboard**: Provides an overview of the dataset, quick statistics, and a sample image.
2. **Dataset Explorer**: Interactive browsing with filters for image type (Real/AI-Generated) and image count.
3. **Statistics & Insights**: Comprehensive visualizations including distribution charts, pie charts, and key metrics on image types.
4. **Random Gallery**: Dynamic image gallery that displays random selections from the dataset.

### How to Use the App

#### Navigation
- Use the tab menu to switch between different sections.
- Each tab offers unique functionality for dataset exploration.

#### Dataset Explorer
1. Select filters from the dropdown menu (Image Type).
2. Adjust the number of images to display using the slider.
3. View the image and its corresponding label (Real or AI-Generated).

#### Statistics Section
- Review key metrics displayed in colorful cards.
- Explore distribution charts showing the percentage of real vs. AI-generated images.
- Read insights about dataset characteristics.

#### Random Gallery
- Click "Show Random Images" to view a fresh random selection.
- Refresh multiple times to explore dataset diversity.

### Key Insights Users Can Gain

1. **Understanding AI Image Distribution**:
   - Analyze the balance of real vs. AI-generated images in the CIFAKE dataset.
   - Visually compare the characteristics of the two image types.

2. **Dataset Characteristics**:
   - Distribution of image types (Real vs. AI-Generated).
   - Key metrics on the total size and composition of the dataset.

3. **Data Analysis Skills**:
   - Interactive filtering and data exploration.
   - Statistical visualization interpretation.

## Technical Requirements Met

### ✅ Dataset Selection & Loading
- **Dataset**: CIFAKE-image-dataset (train and test splits) from Hugging Face
- **Loading**: Uses `datasets.load_dataset()` with caching
- **Visualization**: Sample grid and interactive filters

### ✅ Summary Statistics & Insights
- **Counts**: Total images, Real images, AI-Generated images
- **Charts**: Pie chart for image type distribution
- **Insights**: Key findings about dataset composition

### ✅ Interactive Features
- **Image Gallery**: Updates based on image type filter.
- **Random Button**: "Show Random Images" displays a random selection.
- **Tab-based organization**: 4 organized sections.

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
   - Hardware: **CPU Basic (free tier) - Easier to run with lighter dependencies**

3. **Upload Files**
   - Upload `app.py` (the main Streamlit application)
   - Upload the **NEW, LIGHTER** `requirements.txt`
   - Commit the files

4. **Wait for Build**
   - Space will automatically build and deploy
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
