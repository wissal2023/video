from flask import Flask, request, jsonify, send_file
import os
import tempfile
import subprocess
from werkzeug.utils import secure_filename
from main_version_3 import summarize_with_mistral, extract_text_and_images_from_pptx, chunk_text, VideoConfig

app = Flask(__name__)

UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/convert', methods=['POST'])
def convert_pptx():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Extract text and images
        text, images = extract_text_and_images_from_pptx(file_path)
        chunks = chunk_text(text)
        config = VideoConfig()  # default config
        summarize_with_mistral(text, chunks, config, images)

        # Assuming the video is saved as final_video.mp4 in current directory
        video_path = os.path.join(os.getcwd(), 'final_video.mp4')
        if os.path.exists(video_path):
            return send_file(video_path, as_attachment=True)
        else:
            return jsonify({'error': 'Video file not found after processing'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(port=5001)
