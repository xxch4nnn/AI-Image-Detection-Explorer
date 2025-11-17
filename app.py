import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import numpy as np
import random
from collections import Counter
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AI Image Detection Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache dataset loading
@st.cache_resource
def load_image_dataset():
    """Load the dataset from Hugging Face"""
    try:
        # Using a lightweight AI-generated image detection dataset
        # This dataset contains real and AI-generated images
        train_dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="train")
        test_dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="test")
        dataset = concatenate_datasets([train_dataset, test_dataset])
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Fallback to a simpler dataset structure
        return None

@st.cache_resource
def load_classification_model():
    """Load a pre-trained ResNet18 model for image classification"""
    try:
        model = models.resnet18(pretrained=True)
        model.eval()
        
        # Image preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        return model, preprocess
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 AI-Generated Image Detection Explorer</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load dataset
    with st.spinner("Loading dataset... This may take a moment..."):
        dataset = load_image_dataset()
    
    if dataset is None:
        st.error("Failed to load dataset. Please check your internet connection.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home", 
        "📊 Dataset Explorer", 
        "📈 Statistics", 
        "🎲 Random Gallery",
        "🤖 ML Classification"
    ])
    
    # TAB 1: HOME
    with tab1:
        st.header("Welcome to AI-Generated Image Detection Explorer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### About This Application
            
            This interactive web application explores the **CIFAKE** dataset, a collection of
            real and AI-generated images. The dataset is designed for image classification tasks.
            
            - 📊 **Distribution Analysis**: Statistics on real vs. AI-generated images.
            - 🔍 **Visual Exploration**: Interactive image browsing and filtering.
            - 🤖 **ML Classification**: Pre-trained model predictions.
            
            ### Dataset Overview
            
            - **Source**: Hugging Face (dragonintelligence/CIFAKE-image-dataset)
            - **Type**: Real and AI-Generated Images
            - **Size**: 120,000 images
            - **Features**: Images and corresponding labels (real/AI-generated).
            
            ### How to Use
            
            1. **Dataset Explorer**: Browse images and filter by type (Real/AI-Generated).
            2. **Statistics**: View distribution charts.
            3. **Random Gallery**: Discover random images from the dataset.
            4. **ML Classification**: Upload your own images for analysis.
            """)
        
        with col2:
            st.info("### Quick Stats")
            st.metric("Total Images", len(dataset))
            st.metric("Data Source", "CIFAKE")
            st.metric("Image Types", "Real & AI-Generated")
            
            # Sample image
            if len(dataset) > 0:
                st.markdown("### Sample Image")
                sample_idx = random.randint(0, len(dataset)-1)
                sample_image = dataset[sample_idx]['image']
                st.image(sample_image, use_column_width=True)
    
    # TAB 2: DATASET EXPLORER
    with tab2:
        st.header("📊 Dataset Explorer")
        
        st.subheader("Dataset Overview")
        st.write("The CIFAKE dataset contains real and AI-generated images. You can filter them below.")

        # Filters
        st.subheader("🔍 Interactive Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Image type filter
            selected_type = st.selectbox(
                "Select Image Type",
                ["All", "Real", "AI-Generated"]
            )
        
        with col2:
            # Number of images to display
            num_images = st.slider("Images to Display", 1, 20, 8)

        # Apply filters
        if selected_type == "All":
            filtered_dataset = dataset
        elif selected_type == "Real":
            filtered_dataset = dataset.filter(lambda example: example['label'] == 0)
        else: # AI-Generated
            filtered_dataset = dataset.filter(lambda example: example['label'] == 1)

        st.success(f"Found {len(filtered_dataset)} images matching filters")
        
        # Display filtered images
        st.subheader("🖼️ Image Gallery")
        
        cols = st.columns(4)
        for i in range(min(num_images, len(filtered_dataset))):
            with cols[i % 4]:
                try:
                    item = filtered_dataset[i]
                    img = item['image']
                    label = "AI-Generated" if item['label'] == 1 else "Real"
                    st.image(img, use_column_width=True, caption=label)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    
    # TAB 3: STATISTICS
    with tab3:
        st.header("📈 Dataset Statistics & Insights")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Images", len(dataset))
            st.markdown('</div>', unsafe_allow_html=True)
        
        label_counts = Counter(dataset['label'])

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Real Images", label_counts[0])
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("AI-Generated Images", label_counts[1])
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        st.subheader("📊 Image Type Distribution")
        
        fig = px.pie(
            values=label_counts.values(),
            names=['Real', 'AI-Generated'],
            title="Distribution of Image Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: RANDOM GALLERY
    with tab4:
        st.header("🎲 Random Image Gallery")
        
        st.markdown("""
        Click the button below to view a random selection of images from the dataset.
        Each refresh shows different images!
        """)
        
        if st.button("🔄 Show Random Images", type="primary"):
            st.subheader("Random Image Selection")
            
            cols = st.columns(4)
            random_indices = random.sample(range(len(dataset)), min(12, len(dataset)))
            
            for idx, rand_idx in enumerate(random_indices):
                with cols[idx % 4]:
                    try:
                        item = dataset[rand_idx]
                        img = item['image']
                        label = "AI-Generated" if item['label'] == 1 else "Real"
                        st.image(img, use_column_width=True, caption=label)
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # TAB 5: ML CLASSIFICATION
    with tab5:
        st.header("🤖 Machine Learning Classification")
        
        st.markdown("""
        ### Pre-trained Model Analysis
        
        Upload your own image to analyze it using a pre-trained ResNet18 model.
        This demonstrates basic image classification capabilities.
        
        **Note**: This model is trained on ImageNet and provides general object classification,
        not specifically for AI-generated vs. real image detection.
        """)
        
        # Load model
        model, preprocess = load_classification_model()
        
        if model is None:
            st.warning("Model loading failed. This feature requires PyTorch.")
            return
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Classification Results")
                
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess
                        input_tensor = preprocess(image)
                        input_batch = input_tensor.unsqueeze(0)
                        
                        # Prediction
                        with torch.no_grad():
                            output = model(input_batch)
                        
                        # Get probabilities
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        top5_prob, top5_catid = torch.topk(probabilities, 5)
                        
                        # Load ImageNet labels
                        LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
                        import requests
                        labels = requests.get(LABELS_URL).json()
                        
                        # Display results
                        st.success("Analysis Complete!")
                        
                        for i in range(5):
                            label = labels[top5_catid[i]]
                            prob = top5_prob[i].item() * 100
                            
                            st.write(f"**{i+1}. {label.title()}**")
                            st.progress(prob / 100)
                            st.write(f"Confidence: {prob:.2f}%")
                            st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"Classification error: {e}")
        
        else:
            st.info("👆 Upload an image to begin classification")
            
            # Show example from dataset
            st.markdown("### Or Try with a Dataset Image")
            if st.button("Classify Random Dataset Image"):
                rand_idx = random.randint(0, len(dataset)-1)
                img = dataset[rand_idx]['image']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img, use_column_width=True)
                
                with col2:
                    with st.spinner("Analyzing..."):
                        try:
                            input_tensor = preprocess(img)
                            input_batch = input_tensor.unsqueeze(0)
                            
                            with torch.no_grad():
                                output = model(input_batch)
                            
                            probabilities = torch.nn.functional.softmax(output[0], dim=0)
                            top5_prob, top5_catid = torch.topk(probabilities, 5)
                            
                            import requests
                            LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
                            labels = requests.get(LABELS_URL).json()
                            
                            for i in range(5):
                                label = labels[top5_catid[i]]
                                prob = top5_prob[i].item() * 100
                                st.write(f"**{label.title()}**: {prob:.1f}%")
                        
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit 🎈 | Dataset from Hugging Face 🤗 | Powered by PyTorch 🔥</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
