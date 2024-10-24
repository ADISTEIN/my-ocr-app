from flask import Flask, request, jsonify
import os
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2
from flask_cors import CORS
# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# OCR API route
@app.route('/api/extract-text', methods=['POST'])
def extract_text():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Save the uploaded image
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        # Perform OCR
        result = ocr_model.ocr(filepath, cls=True)

        extracted_info = {
            'results': []
        }

        if result[0]:
            for line in result[0]:
                bbox, (text, confidence) = line
                extracted_info['results'].append({
                    'text': text,
                    'confidence': float(confidence),
                    'box': bbox
                })

        # Clean up: delete the saved image file
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(extracted_info)

    except Exception as e:
        # Log the error message
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing the image.'}), 500

# Health check route to check server status
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
