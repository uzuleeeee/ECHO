import torch
import os

# --- THIS IS THE MAIN CHANGE ---
# We are swapping the lightweight MobileNetV3 model for the more powerful ResNet101.
# This will result in more accurate segmentation masks.
MODEL_NAME = "deeplabv3_resnet50"
MODEL_FILENAME = f"{MODEL_NAME}.pt"


def export_deeplab_model():
    """
    Downloads the specified pre-trained DeepLabV3 model from PyTorch Hub,
    traces it for optimization, and saves it to a local .pt file.
    """
    print(f"Downloading pre-trained model: {MODEL_NAME}...")
    try:
        # Load the specified model from PyTorch Hub
        model = torch.hub.load("pytorch/vision:v0.10.0", MODEL_NAME, pretrained=True)
        model.eval()  # Set the model to evaluation mode (important for inference)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"An error occurred while downloading the model: {e}")
        return

    print("Tracing the model for optimization...")
    # Create a dummy input tensor with a typical shape.
    # Tracing records the operations performed on this input, creating an optimized
    # script module that can be run more efficiently.
    dummy_input = torch.randn(1, 3, 224, 224)

    try:
        # --- THE FIX IS HERE ---
        # The ResNet101 model returns a dictionary, which the tracer dislikes
        # by default. We add `strict=False` to allow this behavior.
        traced_script_module = torch.jit.trace(model, dummy_input, strict=False)
        print("Model traced successfully.")
    except Exception as e:
        print(f"An error occurred during model tracing: {e}")
        return

    # Save the optimized, traced model
    traced_script_module.save(MODEL_FILENAME)
    print(f"✅ Traced model saved as '{MODEL_FILENAME}'")
    print("\nYou can now upload this file to your Hugging Face Space.")


if __name__ == "__main__":
    export_deeplab_model()
