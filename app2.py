"""
Hugging Face Gradio App for Best Hybrid Model (Adapter-based)
==============================================================

This app hosts the adapter-based hybrid model for plant species identification.
It uses a frozen DINOv2 backbone with trainable adapter layers.

Deploy to Hugging Face Spaces:
    1. Create a new Space on huggingface.co (select Gradio SDK)
    2. Upload: app2.py, requirements.txt, best_hybrid_model.pth, species_mapping.csv
    3. Space will auto-deploy
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
from PIL import Image
import pandas as pd
import os


# ============================================================================
# MODEL ARCHITECTURE - Adapter Model
# ============================================================================

class AdapterModel(nn.Module):
    """
    Adapter-based model with frozen DINOv2 backbone and trainable adapters.
    """
    def __init__(self, num_classes=100, input_dim=768, adapter_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Frozen DINOv2 backbone
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=True,
            num_classes=0,
        )
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Adapter layers (trainable)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, adapter_dim),
            nn.ReLU(inplace=True),
        )
        
        # Classifier head
        self.classifier = nn.Linear(adapter_dim, num_classes)
        
        # Prototypes (for prototype-based inference)
        self.prototypes = None
    
    def forward(self, x):
        # Extract features from frozen backbone
        with torch.no_grad():
            if hasattr(self.backbone, "forward_features"):
                feat = self.backbone.forward_features(x)
            else:
                feat = self.backbone(x)
            
            # Handle different output types
            if hasattr(feat, "last_hidden_state"):
                feat = feat.last_hidden_state
            elif isinstance(feat, dict):
                if "x_norm_clstoken" in feat:
                    feat = feat["x_norm_clstoken"]
                elif "pool" in feat:
                    feat = feat["pool"]
                else:
                    feat = list(feat.values())[0]
            
            if feat.ndim == 3:
                feat = feat[:, 0, :]
            if feat.ndim == 4:
                feat = feat.mean(dim=[2, 3])
        
        # Pass through adapter (trainable)
        adapted_feat = self.adapter(feat)
        
        # Classify
        logits = self.classifier(adapted_feat)
        
        return {
            "logits_cls": logits,
            "features": adapted_feat,
        }


# ============================================================================
# MODEL LOADING & INFERENCE
# ============================================================================

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
idx2species = {}
transform = None

MODEL_PATH = "best_hybrid_model.pth"


def load_model():
    """Load the hybrid adapter model and species mapping."""
    global model, idx2species, transform
    
    # Load species mapping
    if os.path.exists("species_mapping.csv"):
        species_df = pd.read_csv("species_mapping.csv")
        idx2species = species_df.set_index("label_idx")["species_name"].to_dict()
        num_classes = len(idx2species)
    else:
        # Fallback: create dummy mapping
        num_classes = 100
        idx2species = {i: f"Species_{i}" for i in range(num_classes)}
    
    print(f"📦 Initializing Adapter Model...")
    model = AdapterModel(num_classes=num_classes, input_dim=768, adapter_dim=256)
    
    # Load checkpoint
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            # Load adapter weights
            if "adapter_state_dict" in checkpoint:
                model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
                print(f"✅ Loaded adapter layers")
            
            # Load classifier weights
            if "classifier_state_dict" in checkpoint:
                model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
                print(f"✅ Loaded classifier")
            
            # Load prototypes if available
            if "prototypes" in checkpoint:
                model.prototypes = checkpoint["prototypes"].to(device)
                print(f"✅ Loaded prototypes")
            
            print(f"✅ Model loaded from {MODEL_PATH}")
                
        except Exception as e:
            print(f"❌ Error loading model: {str(e)[:200]}")
    else:
        print(f"⚠️ {MODEL_PATH} not found - using random initialization")
    
    model = model.to(device)
    model.eval()
    
    # Define preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    
    print(f"✅ Model ready on {device}")


def predict_image(image):
    """
    Predict plant species from an input image.
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        dict: Gradio-formatted results with species predictions
    """
    if image is None:
        return {}
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs["logits_cls"]
        probs = torch.softmax(logits, dim=1)[0]
    
    # Get top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, k=5)
    
    # Format results
    predictions = {}
    for prob, idx in zip(top5_probs.cpu().numpy(), top5_indices.cpu().numpy()):
        species_name = idx2species.get(int(idx), f"Unknown_{idx}")
        predictions[species_name] = float(prob)
    
    return predictions


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create Gradio UI for the hybrid model."""
    
    # Load model
    load_model()
    
    # Create interface
    with gr.Blocks() as demo:
        
        # Custom CSS
        gr.HTML("""
        <style>
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
            max-width: 1000px;
            margin: auto;
        }
        .gr-button {
            background: #10b981 !important;
            border-color: #10b981 !important;
            color: white !important;
        }
        .gr-button:hover {
            background: #059669 !important;
        }
        h1, h2, h3 {
            color: #1a1a1a;
        }
        </style>
        """)
        
        gr.Markdown(
            """
            # 🌿 Plant Species Identifier
            
            Upload an image of a plant to identify its species.
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Upload Image")
                
                image_input = gr.Image(
                    type="pil",
                    label="Plant Image",
                    height=400
                )
                
                submit_btn = gr.Button("🔍 Identify Species", variant="primary", size="lg")
                
                gr.Markdown("---")
                gr.Markdown(
                    """
                    **Tips for best results:**
                    - Use clear, well-lit images
                    - Include distinctive features (leaves, flowers, bark)
                    - Avoid heavy shadows or extreme angles
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Top-5 Predictions")
                
                output = gr.Label(
                    num_top_classes=5,
                    label="Species Predictions"
                )
                
                gr.Markdown("---")
                gr.Markdown(
                    """
                    ### 📊 About
                    
                    - **Recognized Species:** 100 plant species
                    - **Works with:** Herbarium specimens & field photos
                    - **Provides:** Confidence scores for predictions
                    """
                )
        
        # Connect prediction function
        submit_btn.click(
            fn=predict_image,
            inputs=[image_input],
            outputs=output
        )
        
        gr.Markdown(
            """
            ---
            ### ℹ️ Information
            
            Built for botanical research, supporting plant species identification 
            from both herbarium specimens and field photographs.
            
            ---
            """
        )
    
    return demo


# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )

