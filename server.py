from utils.load_models import load_molmo
from utils.model_utils import get_molmo_output
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
import sys

# Add the parent directory to sys.path to access the utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions from the main app

app = Flask(__name__, template_folder='server/templates')

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'server/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Create templates directory if it doesn't exist
TEMPLATES_FOLDER = 'server/templates'
if not os.path.exists(TEMPLATES_FOLDER):
    os.makedirs(TEMPLATES_FOLDER)

# Request log for dashboard - store last 10 requests
request_history = []
MAX_HISTORY = 10
history_lock = threading.Lock()

# Global variables to store loaded models
processor = None
molmo_model = None
model_loading_lock = threading.Lock()


def load_models():
    """Load the MOLMO model and processor"""
    global processor, molmo_model
    try:
        print("Loading MOLMO model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'allenai/MolmoE-1B-0924'  # Default model
        processor, molmo_model = load_molmo(
            model_name=model_name, device=device)
        print(f"MOLMO model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    global processor, molmo_model
    status = "healthy" if processor is not None and molmo_model is not None else "model not loaded"
    return jsonify({"status": status, "timestamp": time.time()})


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
    print("Received request to /process")
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Extract prompt and image from request
    prompt = request.json.get('prompt')
    image_data = request.json.get('image')

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    if not image_data:
        return jsonify({"error": "Missing image data"}), 400

    # Ensure model is loaded
    global processor, molmo_model
    if processor is None or molmo_model is None:
        with model_loading_lock:
            if processor is None or molmo_model is None:
                success = load_models()
                if not success:
                    return jsonify({"error": "Failed to load VLM model"}), 500

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

        # Process with real VLM
        response = real_vlm_process(prompt, image, filepath)

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


def real_vlm_process(prompt, image, image_path):
    """Real function for VLM processing using MOLMO"""
    global processor, molmo_model

    start_time = time.time()

    try:
        # Get raw output from MOLMO model
        output_text = get_molmo_output(
            image=image,
            processor=processor,
            model=molmo_model,
            prompt=prompt
        )

        processing_time = time.time() - start_time

        return {
            "response_text": output_text,
            "raw_html": True,  # Flag to indicate raw HTML content
            "processing_time": round(processing_time, 2)
        }
    except Exception as e:
        raise Exception(f"Error in VLM processing: {str(e)}")


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
    # Import torch here to avoid loading it unnecessarily when importing this module
    import torch

    # Try to load models at startup
    print("Loading models...")
    load_models()
    print("Models loaded successfully")

    app.run(host='0.0.0.0', port=7777, debug=True)
    print("Server started on port 7777")
