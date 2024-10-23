from flask import Flask, request, jsonify
import os
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2

# Initialize Flask app
app = Flask(__name__)

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# OCR API route
@app.route('/api/extract-text', methods=['POST'])
def extract_text():
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

# Health check route to check server status
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'}), 200

if __name__ == '__main__':
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
