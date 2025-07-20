import os
from flask import Flask, render_template, request, jsonify
try:
    from ai_pipeline import get_prediction_and_explanation, device
    print("Successfully imported AI pipeline function.")
except ImportError as e:
    print("---------------------------------------------------------")
    print(f"ERROR: Could not import from ai_pipeline.py: {e}")
    print("Please ensure ai_pipeline.py exists and has no errors.")
    print("The /predict endpoint will return an error.")
    print("---------------------------------------------------------")
    get_prediction_and_explanation = None
except Exception as e_import:
     print(f"An unexpected error occurred during import: {e_import}")
     get_prediction_and_explanation = None


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analysis')
def analysis_tool():
    return render_template('index.html')



@app.route('/explanation')
def explanation():
    return render_template('explanation.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request...")
    if get_prediction_and_explanation is None:
         return jsonify({'error': 'AI pipeline could not be loaded.'}), 500

    try:
        if 'imageFile' not in request.files or 'description' not in request.form:
            print("Error: Missing 'imageFile' or 'description'.")
            return jsonify({'error': 'Missing image file or description'}), 400

        image_file = request.files['imageFile']
        description_text = request.form['description']

        if image_file.filename == '' or description_text.strip() == '':
             print("Error: Empty file or description.")
             return jsonify({'error': 'Missing image file or description'}), 400

        image_bytes = image_file.read()
        print(f"Read {len(image_bytes)} bytes from image file.")
        print(f"Received description: {description_text[:50]}...")

        print("Calling AI prediction function...")
        results = get_prediction_and_explanation(image_bytes, description_text)
        print("AI prediction function finished.")

        if results is None:
            print("Error: AI function returned None.")
            return jsonify({'error': 'AI prediction failed'}), 500

        print("Sending results back to frontend.")
        return jsonify(results)

    except Exception as e:
        print(f"Error during prediction request processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)