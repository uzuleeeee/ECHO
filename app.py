from flask import Flask, request, send_file
from PIL import Image
import io

# Import both the analysis function and the model loader from your final script
try:
    from analyze_image import run_full_analysis, import_model
except ImportError:
    print("Error: Could not import from analyze_image.py.")
    print("Please ensure analyze_image.py is in the same directory.")
    exit()

# --- Load the model into a global variable when the server starts ---
print("Initializing server and loading AI model...")
MODEL = import_model()
print("Model loaded. Server is ready for requests.")
# --------------------------------

app = Flask(__name__)


@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    if "image" not in request.files:
        return "Error: No image file provided.", 400

    if MODEL is None:
        return "Error: Model is not loaded on the server.", 500

    image_file = request.files["image"]
    input_image = Image.open(image_file.stream).convert("RGB")

    # Pass the globally-loaded model into the analysis function
    processed_template_image = run_full_analysis(input_image, MODEL)

    # Save the processed PNG (with transparency) to an in-memory buffer
    buffer = io.BytesIO()
    processed_template_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Send the processed image back to the iOS app
    return send_file(buffer, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
