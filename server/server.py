from flask import Flask, request, jsonify, render_template, send_from_directory, Response
import os
import time
import base64
import uuid
import json
from io import BytesIO
from PIL import Image
import threading
import datetime

app = Flask(__name__, template_folder='templates')

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Create templates directory if it doesn't exist
TEMPLATES_FOLDER = 'templates'
if not os.path.exists(TEMPLATES_FOLDER):
    os.makedirs(TEMPLATES_FOLDER)

# Request log for dashboard - store last 10 requests
request_history = []
MAX_HISTORY = 10
history_lock = threading.Lock()


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route('/process', methods=['POST'])
def process_request():
    """
    Process a request containing both image and text

    Expected JSON format:
    {
        "prompt": "text prompt here",
        "image": "base64 encoded image data"
    }
    """
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Extract prompt and image from request
    prompt = request.json.get('prompt')
    image_data = request.json.get('image')

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    if not image_data:
        return jsonify({"error": "Missing image data"}), 400

    try:
        # Decode and save the image
        img_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(img_data))

        # Determine image format from the image itself
        img_format = image.format if image.format else "JPEG"
        extension = img_format.lower()

        # Generate a unique filename with appropriate extension
        filename = f"{uuid.uuid4()}.{extension}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath, format=img_format)

        # Mock VLM processing - in reality, this would call your model
        response = mock_vlm_process(prompt, filepath)

        # Create request record for dashboard
        request_record = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "image_path": filename,
            "status": "completed",
            "result": response,
            "processing_time": response["processing_time"]
        }

        # Add to history with thread safety
        with history_lock:
            request_history.append(request_record)
            if len(request_history) > MAX_HISTORY:
                request_history.pop(0)

        return jsonify({
            "success": True,
            "prompt": prompt,
            "response": response,
            "image_saved_as": filename
        })

    except Exception as e:
        error_message = str(e)
        # Log error to history
        with history_lock:
            request_history.append({
                "id": str(uuid.uuid4())[:8],
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "status": "error",
                "error": error_message
            })
        return jsonify({"error": f"Error processing request: {error_message}"}), 500


def mock_vlm_process(prompt, image_path):
    """Mock function for VLM processing"""
    # In a real implementation, this would call your VLM model
    time.sleep(1)  # Simulate processing time
    return {
        "response_text": f"Processed prompt: '{prompt}' with image at {image_path}",
        "confidence": 0.95,
        "processing_time": 8.0
    }


@app.route('/')
def dashboard():
    """Render the dashboard page"""
    return render_template('dashboard.html')


@app.route('/data')
def get_data():
    """API endpoint to get the current request history"""
    with history_lock:
        return jsonify(request_history)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True)
    print("Server started on port 7777")
