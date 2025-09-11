import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="DermAI - Professional Skin Lesion Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(240, 147, 251, 0.4);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
        margin: 1rem 0;
    }
    
    .confidence-high { color: #28a745; font-weight: 600; }
    .confidence-medium { color: #ffc107; font-weight: 600; }
    .confidence-low { color: #dc3545; font-weight: 600; }
    
    .upload-section {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(45deg, #f8f9ff, #f0f4ff);
        margin: 1rem 0;
    }
    
    .sidebar-content {
        padding: 1rem 0;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .analysis-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 2rem 0;
        color: #856404;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model(model_path):
    """Load the trained model with caching for better performance"""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # EfficientNet-B3
        model = timm.create_model("tf_efficientnet_b3.ns_jft_in1k", pretrained=False, num_classes=len(checkpoint["label_map"]))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        label_map = checkpoint["label_map"]
        idx2label = checkpoint["idx2label"]
        return model, label_map, idx2label, True
    except Exception as e:
        return None, None, None, False

# ========== LESION INFORMATION DATABASE ==========
LESION_INFO = {
    "akiec": {
        "name": "Actinic Keratoses",
        "description": "Pre-cancerous lesions caused by sun damage",
        "severity": "Medium",
        "color": "#ff9f43"
    },
    "bcc": {
        "name": "Basal Cell Carcinoma", 
        "description": "Most common form of skin cancer",
        "severity": "High",
        "color": "#e74c3c"
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": "Non-cancerous skin growth",
        "severity": "Low",
        "color": "#2ecc71"
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "Benign skin tumor",
        "severity": "Low", 
        "color": "#3498db"
    },
    "mel": {
        "name": "Melanoma",
        "description": "Serious form of skin cancer",
        "severity": "Critical",
        "color": "#8e44ad"
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "description": "Common mole",
        "severity": "Low",
        "color": "#27ae60"
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": "Blood vessel related skin lesion",
        "severity": "Low",
        "color": "#f39c12"
    }
}

# ========== MAIN APP ==========
def main():
    # Header Section
    st.markdown("""
    <div class="main-header">
        <h1>🏥 DermAI</h1>
        <p>Advanced AI-Powered Dermatological Analysis System</p>
        <p>Professional Skin Lesion Classification using HAM10000 Dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔬 Analysis Dashboard")
        
        # Model Status
        model_path = "model/ham10000_best_streamlit.pth"
        
        if os.path.exists(model_path):
            model, label_map, idx2label, model_loaded = load_model(model_path)
            
            if model_loaded:
                st.success("✅ Model loaded successfully")
                st.info(f"📊 Classes: {len(label_map)}")
            else:
                st.error("❌ Error loading model")
                return
        else:
            st.error("❌ Model file not found")
            st.write("Please ensure 'ham10000_best_streamlit.pth' is in the same directory.")
            return
        
        st.markdown("---")
        
        # Feature Information
        st.markdown("### 🎯 Model Features")
        st.markdown("""
        <div class="feature-highlight">
            <strong>Architecture:</strong> EfficientNet-B3<br>
            <strong>Dataset:</strong> HAM10000<br>
            <strong>Input Size:</strong> 224×224<br>
            <strong>Classes:</strong> 7 lesion types
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis Statistics (placeholder)
        st.markdown("### 📈 Session Stats")
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        
        st.metric("Analyses Performed", st.session_state.analysis_count)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Dermatoscopic Image")
        
        st.markdown("""
        <div class="info-card">
            <h4>🔍 Image Requirements</h4>
            <ul>
                <li>High-quality dermatoscopic images</li>
                <li>Supported formats: JPG, JPEG, PNG</li>
                <li>Clear, well-lit lesion images</li>
                <li>Minimum resolution: 224×224 pixels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a dermatoscopic image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📸 Uploaded Image", use_column_width=True)
            
            # Image information
            st.markdown(f"""
            <div class="info-card">
                <strong>Image Info:</strong><br>
                📏 Size: {image.size[0]}×{image.size[1]} pixels<br>
                💾 File: {uploaded_file.name}<br>
                📅 Uploaded: {datetime.now().strftime("%H:%M:%S")}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🧠 AI Analysis Results")
            
            # Processing animation
            with st.spinner("🔬 Analyzing image..."):
                # Preprocessing
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(image).unsqueeze(0)
                
                # Simulate processing time for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Predict
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_idx].item()
                
                pred_label = idx2label[pred_idx]
                
                # Update session stats
                st.session_state.analysis_count += 1
            
            # Main Prediction Card
            lesion_info = LESION_INFO.get(pred_label, {"name": pred_label, "description": "Unknown lesion type", "severity": "Unknown", "color": "#95a5a6"})
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Primary Diagnosis</h2>
                <h1>{lesion_info['name']}</h1>
                <p style="font-size: 1.1rem; opacity: 0.9;">{lesion_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Metrics
            confidence_color = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
            confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            
            col_conf1, col_conf2, col_conf3 = st.columns(3)
            
            with col_conf1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Confidence</h3>
                    <h1>{confidence:.1%}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_conf2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚠️ Severity</h3>
                    <h1>{lesion_info['severity']}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_conf3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔬 Analysis</h3>
                    <h1>Complete</h1>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.markdown("### 👈 Upload an image to begin analysis")
            st.markdown("""
            <div class="analysis-section">
                <h4>🚀 How it works:</h4>
                <ol>
                    <li><strong>Upload</strong> a dermatoscopic image</li>
                    <li><strong>AI processes</strong> the image using EfficientNet-B3</li>
                    <li><strong>Get results</strong> with confidence scores</li>
                    <li><strong>View detailed</strong> analysis and recommendations</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Analysis Section (only when image is uploaded)
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("## 📊 Comprehensive Analysis Report")
        
        # Top-3 Predictions Visualization
        col_chart1, col_chart2 = st.columns([1, 1])
        
        with col_chart1:
            st.markdown("### 🏆 Top 3 Predictions")
            
            # Get top-3 predictions
            top3_probs, top3_idx = torch.topk(probs, 3)
            
            # Create dataframe for chart
            top3_data = []
            for i in range(3):
                lbl_key = idx2label[top3_idx[0][i].item()]
                lbl_info = LESION_INFO.get(lbl_key, {"name": lbl_key})
                top3_data.append({
                    "Lesion": lbl_info["name"],
                    "Confidence": top3_probs[0][i].item(),
                    "Percentage": f"{top3_probs[0][i].item():.1%}"
                })
            
            # Create horizontal bar chart
            df_top3 = pd.DataFrame(top3_data)
            fig = px.bar(
                df_top3, 
                x="Confidence", 
                y="Lesion",
                orientation='h',
                color="Confidence",
                color_continuous_scale="viridis",
                text="Percentage"
            )
            fig.update_layout(
                title="Prediction Confidence Levels",
                xaxis_title="Confidence Score",
                yaxis_title="",
                showlegend=False,
                height=300
            )
            fig.update_traces(textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            st.markdown("### 🎯 Confidence Distribution")
            
            # Create pie chart for all predictions
            all_probs = probs[0].numpy()
            all_labels = [LESION_INFO.get(idx2label[i], {"name": idx2label[i]})["name"] for i in range(len(all_probs))]
            
            fig_pie = px.pie(
                values=all_probs,
                names=all_labels,
                title="Complete Classification Results"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed Information Table
        st.markdown("### 📋 Detailed Classification Results")
        
        detailed_results = []
        for i in range(len(all_probs)):
            lbl_key = idx2label[i]
            lbl_info = LESION_INFO.get(lbl_key, {"name": lbl_key, "description": "Unknown", "severity": "Unknown"})
            detailed_results.append({
                "Rank": i + 1,
                "Lesion Type": lbl_info["name"],
                "Confidence": f"{all_probs[i]:.1%}",
                "Severity": lbl_info["severity"],
                "Description": lbl_info["description"]
            })
        
        # Sort by confidence
        detailed_results.sort(key=lambda x: float(x["Confidence"].rstrip('%')), reverse=True)
        for i, result in enumerate(detailed_results):
            result["Rank"] = i + 1
        
        df_results = pd.DataFrame(detailed_results)
        st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Medical Disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ Important Medical Disclaimer</h4>
        <p><strong>This AI system is designed for educational and research purposes only.</strong> The results provided by this application should not be considered as medical advice, diagnosis, or treatment recommendations. Always consult with qualified healthcare professionals for proper medical evaluation and treatment of any skin conditions.</p>
        
        <p><strong>For Medical Emergencies:</strong> If you notice rapid changes in a lesion, bleeding, or other concerning symptoms, please seek immediate medical attention.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>DermAI Professional</strong> | AI-Powered Dermatological Analysis | Built with Streamlit & PyTorch</p>
        <p>Developed for Advanced Medical AI Research | HAM10000 Dataset Implementation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()