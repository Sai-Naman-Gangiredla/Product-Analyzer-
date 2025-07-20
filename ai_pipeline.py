# ai_pipeline.py

print("Initializing AI Pipeline...")

# --- Core Python and Data Handling ---
import os
import pandas as pd
import numpy as np
from PIL import Image
import base64
import io
import traceback # For error logging
import time # Added for processing time

# --- PyTorch Core ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# --- PyTorch Image Processing & Models ---
try:
    import timm
    print("timm library loaded.")
except ImportError:
    print("ERROR: timm library not found. Install using 'pip install timm'")
    timm = None

# --- PyTorch Text Processing (Sentence Transformers) ---
try:
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers library loaded.")
except ImportError:
    print("ERROR: sentence-transformers library not found. Install using 'pip install sentence-transformers'")
    SentenceTransformer = None

# --- XAI Libraries ---
try:
    from captum.attr import LayerGradCam
    from captum.attr import visualization as viz
    import matplotlib.pyplot as plt # Needed by viz helper
    print("Captum library loaded.")
except ImportError:
    print("WARNING: Captum library not found. Image XAI (Grad-CAM) will not work.")
    LayerGradCam = None
    viz = None
    plt = None
try:
    import lime
    import lime.lime_text
    print("LIME library loaded.")
except ImportError:
    print("WARNING: LIME library not found. Text XAI (LIME) will not work.")
    lime = None

# --- Global Configurations & Constants ---
print("Defining global configurations...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch using CPU.")

NUM_CLASSES = 21
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
MODEL_IMG_SIZE = 224
TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TEXT_EMBEDDING_DIM = 384
IMAGE_MODEL_NAME = 'efficientnet_lite0'
IMAGE_FEATURE_DIM = 1280
BEST_MODEL_PATH = 'best_multimodal_model.pth'

idx_to_category_map = {
    0: 'All Beauty', 1: 'All Electronics', 2: 'Appliances', 3: 'Arts, Crafts & Sewing',
    4: 'Automotive', 5: 'Baby', 6: 'Baby Products', 7: 'Beauty',
    8: 'Cell Phones & Accessories', 9: 'Clothing, Shoes & Jewelry', 10: 'Electronics',
    11: 'Grocery & Gourmet Food', 12: 'Health & Personal Care', 13: 'Industrial & Scientific',
    14: 'Musical Instruments', 15: 'Office Products', 16: 'Patio, Lawn & Garden',
    17: 'Pet Supplies', 18: 'Sports & Outdoors', 19: 'Tools & Home Improvement',
    20: 'Toys & Games'
}
if len(idx_to_category_map) != NUM_CLASSES:
     print(f"WARNING: NUM_CLASSES ({NUM_CLASSES}) does not match length of idx_to_category_map ({len(idx_to_category_map)})!")
     NUM_CLASSES = len(idx_to_category_map)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
image_transforms = transforms.Compose([
    transforms.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
print("Image transformations defined.")

class MultimodalProductClassifier(nn.Module):
    def __init__(self, image_backbone_model, text_backbone_model, mlp_classifier_head):
        super(MultimodalProductClassifier, self).__init__()
        self.image_backbone = image_backbone_model
        self.text_backbone = text_backbone_model
        self.mlp_head = mlp_classifier_head

    def forward(self, image_tensors, text_list):
        image_tensors = image_tensors.to(device)
        image_features = self.image_backbone(image_tensors)
        text_features = self.text_backbone.encode(
            text_list, convert_to_tensor=True, device=device, show_progress_bar=False
        )
        text_features = text_features.to(device)
        combined_features = torch.cat((image_features, text_features), dim=1)
        logits = self.mlp_head(combined_features)
        return logits
print("MultimodalProductClassifier class defined.")

loaded_image_backbone = None
loaded_text_backbone = None
loaded_mlp_head = None
model = None
target_layer = None
explainer = None

try:
    if timm:
        loaded_image_backbone = timm.create_model(IMAGE_MODEL_NAME, pretrained=False, num_classes=0)
        print(f"Image backbone structure '{IMAGE_MODEL_NAME}' loaded.")
    else:
        print("ERROR: timm not loaded, cannot create image backbone.")

    if SentenceTransformer:
        loaded_text_backbone = SentenceTransformer(TEXT_MODEL_NAME)
        print(f"Text backbone structure '{TEXT_MODEL_NAME}' loaded.")
    else:
        print("ERROR: SentenceTransformer not loaded, cannot create text backbone.")

    loaded_mlp_head = nn.Sequential(
        nn.Linear(IMAGE_FEATURE_DIM + TEXT_EMBEDDING_DIM, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, NUM_CLASSES)
    )
    print("MLP head structure defined.")

    if loaded_image_backbone and loaded_text_backbone and loaded_mlp_head:
        model = MultimodalProductClassifier(
            image_backbone_model=loaded_image_backbone,
            text_backbone_model=loaded_text_backbone,
            mlp_classifier_head=loaded_mlp_head
        )
        print("Combined model architecture instantiated.")

        if os.path.exists(BEST_MODEL_PATH):
            print(f"Loading trained weights from {BEST_MODEL_PATH}...")
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
            print("Trained weights loaded successfully.")
        else:
            print(f"WARNING: Saved model weights not found at {BEST_MODEL_PATH}. Model is UNTRAINED.")

        model.to(device)
        model.eval()
        print(f"Model moved to {device} and set to evaluation mode.")

        try:
            # Attempt to access a common final layer in many timm models.
            # For efficientnet_lite0, 'conv_head' is usually the last conv layer before the classifier.
            # If you use a different EfficientNet or another model, this might need adjustment.
            # You can inspect model.image_backbone to find the correct layer name.
            if hasattr(model.image_backbone, 'conv_head'):
                target_layer = model.image_backbone.conv_head
            elif hasattr(model.image_backbone, 'features') and isinstance(model.image_backbone.features, nn.Sequential):
                 # Fallback: try to get the last conv layer from a 'features' block
                for layer in reversed(list(model.image_backbone.features.children())):
                    if isinstance(layer, nn.Conv2d):
                        target_layer = layer
                        break
            if target_layer:
                 print(f"XAI target layer set: {type(target_layer).__name__}")
            else:
                print("WARNING: Could not automatically determine XAI target_layer. Grad-CAM may not work as expected.")

        except Exception as e:
            print(f"WARNING: Could not set XAI target_layer: {e}")

        if lime and idx_to_category_map and NUM_CLASSES > 0:
            class_names_list = [idx_to_category_map[i] for i in range(NUM_CLASSES)]
            explainer = lime.lime_text.LimeTextExplainer(class_names=class_names_list)
            print("LIME explainer created.")
        else:
            print("Could not create LIME explainer (library or mapping missing).")
    else:
         print("ERROR: Could not instantiate combined model due to missing components.")

except Exception as e:
     print(f"ERROR during model loading or setup: {e}")
     traceback.print_exc()
     model = None

def generate_grad_cam(model_to_explain, input_image_tensor, input_text_list, target_class_index, target_cnn_layer):
    if LayerGradCam is None or target_cnn_layer is None: # Check if target_cnn_layer is set
        print("Grad-CAM prerequisites not met (Captum or target_layer missing).")
        return None
    model_to_explain.eval()
    if input_image_tensor.ndim == 3: input_image_tensor = input_image_tensor.unsqueeze(0)
    
    # Wrapper for Captum: needs a function that takes only image input for image attribution
    # The text input is fixed for this specific Grad-CAM call.
    def model_wrapper_for_image_attribution(img_inp):
        return model_to_explain(img_inp, input_text_list) # input_text_list is from the outer scope

    layer_gc = LayerGradCam(model_wrapper_for_image_attribution, target_cnn_layer)
    attribution = layer_gc.attribute(input_image_tensor, target=target_class_index, relu_attributions=True)
    return attribution

def create_grad_cam_overlay_base64(original_img_tensor_cpu, attribution_tensor, title="Grad-CAM"):
    if viz is None or plt is None or attribution_tensor is None: return None # Check attribution_tensor
    try:
        img_np = original_img_tensor_cpu.numpy().transpose(1, 2, 0)
        mean = np.array(imagenet_mean); std = np.array(imagenet_std)
        img_np = std * img_np + mean; img_np = np.clip(img_np, 0, 1)
        attribution_np = attribution_tensor[0].permute(1,2,0).cpu().detach().numpy()
        fig, ax = viz.visualize_image_attr(
            attr=attribution_np, original_image=img_np, method='blended_heat_map',
            sign='positive', show_colorbar=False, title=title, use_pyplot=False
        )
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
        buf.seek(0); image_base64 = base64.b64encode(buf.read()).decode('utf-8'); buf.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e: print(f"Error creating Grad-CAM overlay image: {e}"); return None

print("Helper functions defined.")

def get_prediction_and_explanation(image_path_or_bytes, input_text):
    if model is None:
         print("ERROR: Model not loaded in ai_pipeline. Cannot predict.")
         return None

    start_time = time.time() # Start timing
    results = {
        'processing_time': None,
        'image_features_extracted': False,
        'text_features_extracted': False
    } # Initialize with new metric placeholders

    model.eval()
    try:
        # 1. Load/Preprocess Image
        if isinstance(image_path_or_bytes, str): img = Image.open(image_path_or_bytes).convert('RGB')
        elif isinstance(image_path_or_bytes, bytes): img = Image.open(io.BytesIO(image_path_or_bytes)).convert('RGB')
        else: raise ValueError("Input image must be file path or bytes.")
        
        img_tensor = image_transforms(img)
        img_tensor_gpu = img_tensor.to(device)
        results['input_text'] = input_text # Store input text for display
        
        try:
             img_resized_for_display = img.resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE))
             buf = io.BytesIO(); img_resized_for_display.save(buf, format='PNG'); buf.seek(0)
             results['original_image_b64'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
             buf.close()
        except Exception as e_img_save:
            print(f"Error saving original image for display: {e_img_save}")
            results['original_image_b64'] = None

        # 2. Prediction
        with torch.no_grad(): # Ensure no gradients are computed for standard prediction path
            logits = model(img_tensor_gpu.unsqueeze(0), [input_text]) # Pass text as a list
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            results['predicted_class'] = idx_to_category_map.get(predicted_idx.item(), "Unknown Category")
            results['confidence'] = confidence.item()
            current_predicted_idx = predicted_idx.item()
        
        # Update feature extraction status (simple boolean for now)
        results['image_features_extracted'] = True # Assuming if prediction happens, features were extracted
        results['text_features_extracted'] = True

        # 3. Grad-CAM
        if target_layer is not None and LayerGradCam is not None:
            try:
                # For Grad-CAM, we need to allow gradients through the part of the model we are explaining
                # The model.eval() is set, but Captum handles enabling gradients as needed for its operations.
                attribution = generate_grad_cam(model, img_tensor_gpu, [input_text], current_predicted_idx, target_layer)
                if attribution is not None:
                    results['grad_cam_overlay_b64'] = create_grad_cam_overlay_base64(img_tensor.cpu().detach(), attribution.cpu().detach())
                else:
                    results['grad_cam_overlay_b64'] = None
            except Exception as e_gradcam:
                print(f"Error during Grad-CAM: {e_gradcam}")
                traceback.print_exc()
                results['grad_cam_overlay_b64'] = None
        else:
            print("Skipping Grad-CAM: target_layer or Captum not available.")
            results['grad_cam_overlay_b64'] = None

        # 4. LIME
        if explainer is not None:
            try:
                # LIME predictor function
                def lime_predictor_local(texts_for_lime): # Renamed to avoid conflict
                    all_probas = []
                    batch_size_lime = 64 # Process in batches if many perturbed texts
                    model.eval() # Ensure model is in eval mode
                    with torch.no_grad(): # LIME perturbations don't require gradient tracking for the main model
                         for i in range(0, len(texts_for_lime), batch_size_lime):
                              batch_texts = texts_for_lime[i : i + batch_size_lime]
                              num_texts_in_batch = len(batch_texts)
                              # Repeat the single image tensor for each text in the batch
                              batch_images = img_tensor_gpu.unsqueeze(0).repeat(num_texts_in_batch, 1, 1, 1)
                              
                              logits_lime = model(batch_images, batch_texts)
                              probas = F.softmax(logits_lime, dim=1).cpu().numpy()
                              all_probas.append(probas)
                    return np.vstack(all_probas) if all_probas else np.array([])

                explanation = explainer.explain_instance(
                    input_text,
                    lime_predictor_local,
                    num_features=10, # How many words to highlight
                    top_labels=1,    # Explain for the top predicted label
                    num_samples=1000 # Number of perturbed samples LIME generates
                )
                # Get explanation for the predicted class
                results['lime_explanation'] = explanation.as_list(label=current_predicted_idx)
            except Exception as e_lime:
                print(f"Error during LIME: {e_lime}")
                traceback.print_exc()
                results['lime_explanation'] = []
        else:
            results['lime_explanation'] = []

        end_time = time.time() # End timing
        results['processing_time'] = (end_time - start_time) * 1000 # Convert to milliseconds

        return results

    except Exception as e:
        print(f"ERROR in get_prediction_and_explanation: {e}")
        traceback.print_exc()
        # Ensure all expected keys are present even in case of error, with None values
        results['predicted_class'] = "Error"
        results['confidence'] = 0.0
        results['original_image_b64'] = None
        results['grad_cam_overlay_b64'] = None
        results['lime_explanation'] = []
        if results.get('processing_time') is None: # Check if already set
            end_time = time.time()
            results['processing_time'] = (end_time - start_time) * 1000 if 'start_time' in locals() else -1
        return results


print("Integrated prediction function defined.")
print("\n--- AI Pipeline Initialization Complete ---")

# Example test call (ensure you have a test image)
# if __name__ == '__main__':
#     print("\nRunning a test prediction using ai_pipeline directly...")
#     # Create a dummy image for testing if you don't have one
#     try:
#         from PIL import Image
#         dummy_image = Image.new('RGB', (MODEL_IMG_SIZE, MODEL_IMG_SIZE), color = 'red')
#         dummy_image_bytes = io.BytesIO()
#         dummy_image.save(dummy_image_bytes, format='PNG')
#         dummy_image_bytes = dummy_image_bytes.getvalue()
#         test_text = "This is a test description for a red square."
#         test_results = get_prediction_and_explanation(dummy_image_bytes, test_text)
#     except Exception as e:
#         print(f"Error creating dummy image for test: {e}")
#         test_results = None

#     if test_results:
#         print("\nTest Results:")
#         print(f"  Predicted Class: {test_results.get('predicted_class')}")
#         print(f"  Confidence: {test_results.get('confidence', 0.0):.4f}")
#         print(f"  Processing Time: {test_results.get('processing_time', -1):.2f} ms")
#         print(f"  Image Features Extracted: {test_results.get('image_features_extracted')}")
#         print(f"  Text Features Extracted: {test_results.get('text_features_extracted')}")
#         print(f"  LIME (Top 3 for predicted): {test_results.get('lime_explanation', [])[:3]}")
#         print(f"  Grad-CAM data exists: {test_results.get('grad_cam_overlay_b64') is not None}")
#         if test_results.get('grad_cam_overlay_b64'):
#             print("  (To view Grad-CAM, copy the base64 string into a data URL: data:image/png;base64,YOUR_STRING)")
#     else:
#         print("Test prediction failed or could not be run.")

