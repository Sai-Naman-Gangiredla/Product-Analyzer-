#Product Analyzer
An AI-powered multimodal classification project that analyzes both images and text to predict product categories.

This project was developed as a final year submission.

##📦 Project Structure:
app.py — Main Flask web app
ai_pipeline.py — AI/ML pipeline and model code
requirements.txt — List of required Python packages
best_multimodal_model.zip — Zipped model file (must be unzipped before use)
saved_models/ — Additional model files
static/ — CSS and images for the web app
templates/ — HTML templates for the web app
21BCE8082_AP2024254001601_RV4.pdf — Final report

##🚀 Setup Instructions:
1. Install Anaconda (if not already installed)
Download from: https://www.anaconda.com/products/distribution
Install and open Anaconda Prompt
2. Clone the Repository
Apply: git clone https://github.com/Sai-Naman-Gangiredla/Product-Analyzer-.git
cd Product-Analyzer-
3. Create and Activate the Environment
Apply: conda create -n sdp python=3.10
       conda activate sdp
4. Install Requirements
Apply: pip install -r requirements.txt

##📦 How to Handle the Zipped Model File:
Find best_multimodal_model.zip in the project folder.
Right-click > Extract All...
Make sure best_multimodal_model.pth is in the same folder as app.py after unzipping.

##▶️ How to Run the App:
Apply: python app.py
Open your browser and go to: http://localhost:5000/

##📝 What to Change:
If you rename the model file, update the path in ai_pipeline.py:
Apply:   BEST_MODEL_PATH = 'best_multimodal_model.pth'
If you move the model file, update the path accordingly.
If you want to use a different model, place it in the project folder and update the code as above.

##💡 How to Use:
Open the web app in your browser.
Upload an image and enter a product description.
Click the button to get predictions and explanations.
View the predicted category, confidence, and explanation visuals.

##📄 Credits / Report:
See 21BCE8082_AP2024254001601_RV4.pdf for the full project report and technical details.

##👤 Author:
Sai Naman Gangiredla
