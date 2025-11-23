
import logging
from flask import Flask, request, jsonify
from PIL import Image
import stag
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download the model
pretrained = hf_hub_download(
    repo_id="xinyu1s/recognize-anything-plus-model",
    filename="ram_plus_swin_large_14m.pth"
)

# Create and run the tagger
tagger = stag.SKTagger(
    model_path=pretrained,
    image_size=384,
    force_tagging=False,
    test_mode=False,
    prefer_exact_filenames=False,
    tag_prefix="st"
)

@app.route('/tag_image', methods=['POST'])
def tag_image():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    if file:
        try:
            image = Image.open(file.stream)
            tags = tagger.get_tags_for_image(image)
            return jsonify({'tags': tags.split('|')})
        except Exception as e:
            logging.error(f"Error processing image: {e}", exc_info=True)
            return jsonify({'error': 'An internal error occurred while processing the image.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
