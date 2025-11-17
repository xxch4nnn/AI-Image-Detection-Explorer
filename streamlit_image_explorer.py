import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
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
        dataset = load_dataset("poloclub/diffusiondb", "2m_random_1k", split="train")
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

def prepare_metadata(dataset):
    """Extract and prepare metadata from dataset"""
    try:
        metadata_list = []
        
        for idx, item in enumerate(dataset):
            if idx >= 1000:  # Limit for performance
                break
            
            metadata = {
                'index': idx,
                'prompt': item.get('prompt', 'N/A')[:100],  # Truncate long prompts
                'seed': item.get('seed', 'N/A'),
                'step': item.get('step', 'N/A'),
                'cfg': item.get('cfg', 'N/A'),
                'sampler': item.get('sampler', 'N/A'),
                'width': item.get('width', 'N/A'),
                'height': item.get('height', 'N/A'),
            }
            metadata_list.append(metadata)
        
        df = pd.DataFrame(metadata_list)
        return df
    except Exception as e:
        st.error(f"Error preparing metadata: {e}")
        return pd.DataFrame()

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
            
            This interactive web application explores the **DiffusionDB** dataset, a collection of 
            AI-generated images created using Stable Diffusion. The dataset provides insights into:
            
            - 🎨 **Image Generation**: Prompts, parameters, and configurations
            - 📊 **Distribution Analysis**: Statistics on generation methods
            - 🔍 **Visual Exploration**: Interactive image browsing
            - 🤖 **ML Classification**: Pre-trained model predictions
            
            ### Dataset Overview
            
            - **Source**: Hugging Face DiffusionDB
            - **Type**: AI-Generated Images (Stable Diffusion)
            - **Size**: 1,000+ images with metadata
            - **Features**: Prompts, seeds, samplers, and generation parameters
            
            ### How to Use
            
            1. **Dataset Explorer**: Browse images and filter by parameters
            2. **Statistics**: View distribution charts and insights
            3. **Random Gallery**: Discover random images from the dataset
            4. **ML Classification**: Upload your own images for analysis
            """)
        
        with col2:
            st.info("### Quick Stats")
            st.metric("Total Images", len(dataset))
            st.metric("Data Source", "DiffusionDB")
            st.metric("Image Type", "AI-Generated")
            
            # Sample image
            if len(dataset) > 0:
                st.markdown("### Sample Image")
                sample_idx = random.randint(0, min(len(dataset)-1, 100))
                sample_image = dataset[sample_idx]['image']
                st.image(sample_image, use_column_width=True)
    
    # TAB 2: DATASET EXPLORER
    with tab2:
        st.header("📊 Dataset Explorer")
        
        # Prepare metadata
        with st.spinner("Preparing metadata..."):
            df_metadata = prepare_metadata(dataset)
        
        if df_metadata.empty:
            st.warning("No metadata available")
            return
        
        st.subheader("Dataset Metadata")
        st.dataframe(df_metadata, use_container_width=True, height=300)
        
        # Download button
        csv = df_metadata.to_csv(index=False)
        st.download_button(
            label="📥 Download Metadata CSV",
            data=csv,
            file_name="diffusiondb_metadata.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Filters
        st.subheader("🔍 Interactive Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Sampler filter
            samplers = df_metadata['sampler'].unique()
            selected_sampler = st.selectbox(
                "Select Sampler",
                ["All"] + list(samplers)
            )
        
        with col2:
            # Step range
            if 'step' in df_metadata.columns and df_metadata['step'].dtype != 'object':
                step_range = st.slider(
                    "Generation Steps",
                    int(df_metadata['step'].min()) if df_metadata['step'].min() != 'N/A' else 0,
                    int(df_metadata['step'].max()) if df_metadata['step'].max() != 'N/A' else 100,
                    (0, 100)
                )
            else:
                step_range = None
        
        with col3:
            # Number of images to display
            num_images = st.slider("Images to Display", 1, 20, 8)
        
        # Apply filters
        filtered_df = df_metadata.copy()
        if selected_sampler != "All":
            filtered_df = filtered_df[filtered_df['sampler'] == selected_sampler]
        
        st.success(f"Found {len(filtered_df)} images matching filters")
        
        # Display filtered images
        st.subheader("🖼️ Image Gallery")
        
        cols = st.columns(4)
        for idx, row in filtered_df.head(num_images).iterrows():
            with cols[idx % 4]:
                try:
                    img = dataset[int(row['index'])]['image']
                    st.image(img, use_column_width=True)
                    
                    with st.expander("View Details"):
                        st.write(f"**Prompt**: {row['prompt']}")
                        st.write(f"**Sampler**: {row['sampler']}")
                        st.write(f"**Steps**: {row['step']}")
                        st.write(f"**CFG**: {row['cfg']}")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    
    # TAB 3: STATISTICS
    with tab3:
        st.header("📈 Dataset Statistics & Insights")
        
        df_metadata = prepare_metadata(dataset)
        
        if df_metadata.empty:
            st.warning("No data available for statistics")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Images", len(df_metadata))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            unique_samplers = df_metadata['sampler'].nunique()
            st.metric("Unique Samplers", unique_samplers)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_steps = df_metadata['step'].apply(
                lambda x: float(x) if x != 'N/A' and str(x).replace('.','').isdigit() else 0
            ).mean()
            st.metric("Avg. Steps", f"{avg_steps:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Data Quality", "High")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sampler Distribution")
            sampler_counts = df_metadata['sampler'].value_counts()
            fig = px.pie(
                values=sampler_counts.values,
                names=sampler_counts.index,
                title="Distribution of Sampling Methods",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Generation Steps Distribution")
            # Filter numeric steps
            numeric_steps = df_metadata['step'].apply(
                lambda x: float(x) if x != 'N/A' and str(x).replace('.','').isdigit() else None
            ).dropna()
            
            if len(numeric_steps) > 0:
                fig = px.histogram(
                    numeric_steps,
                    nbins=20,
                    title="Distribution of Generation Steps",
                    labels={'value': 'Steps', 'count': 'Frequency'},
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric step data available")
        
        # Additional insights
        st.subheader("💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Most Common Sampler**: {df_metadata['sampler'].mode()[0] if not df_metadata['sampler'].mode().empty else 'N/A'}
            
            **Resolution Analysis**: Most images are generated at standard resolutions
            optimized for Stable Diffusion models.
            """)
        
        with col2:
            st.success(f"""
            **Dataset Diversity**: The dataset contains diverse prompts covering
            various subjects, styles, and artistic directions.
            
            **Generation Quality**: High-quality outputs with proper parameter tuning.
            """)
    
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
            random_indices = random.sample(range(min(len(dataset), 1000)), min(12, len(dataset)))
            
            for idx, rand_idx in enumerate(random_indices):
                with cols[idx % 4]:
                    try:
                        img = dataset[rand_idx]['image']
                        st.image(img, use_column_width=True)
                        
                        with st.expander("Metadata"):
                            item = dataset[rand_idx]
                            st.write(f"**Prompt**: {item.get('prompt', 'N/A')[:80]}...")
                            st.write(f"**Sampler**: {item.get('sampler', 'N/A')}")
                            st.write(f"**Steps**: {item.get('step', 'N/A')}")
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
                rand_idx = random.randint(0, min(len(dataset)-1, 100))
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
