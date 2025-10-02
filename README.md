# ECHO - AI Composition Guide

*Recreate any photo, perfectly.*

ECHO is an iOS camera app that helps you replicate the composition of any photo. Upload a reference image, and our AI-powered server analyzes its core compositional elements such as the outline of the person, the main horizon line, and key perspective lines. The app then generates a clean, semi-transparent "template" and overlays it on your live camera feed, allowing you to perfectly align your subject and scene to match the original shot.

## Demo

The app takes any input image and generates a clean, aesthetic compositional guide.

| Input Image | Generated Template | 
| --- | --- | 
| ![Image 2](/ExampleImages/image2.png) | ![Outline 2](/ExampleImages/outline2.png) | 


## Features

### iOS Application (Client)

* **Live Camera Feed:** Full-screen camera preview with manual controls for zoom and exposure.
* **Interactive Overlay:** The generated template can be moved, scaled, and rotated using intuitive gestures.
* **Server Communication:** Seamlessly uploads images to the backend and displays the returned compositional guide.
* **Clean UI:** A minimal, modern interface built entirely in SwiftUI.

### AI Backend (Server)

* **Person Segmentation:** Accurately isolates human subjects in any photo using Google's **MediaPipe Selfie Segmentation**. The final output is a smoothed, aesthetically pleasing outline.
* **Horizon Detection:** A custom algorithm analyzes image gradients to find the dominant tilted or horizontal line in the composition.
* **One-Point Perspective Analysis:** Robustly detects the primary vanishing point (including off-screen) by using OpenCV's Line Segment Detector (LSD) and MeanShift clustering to find the point of maximum line intersection.
* **Aesthetic Template Generation:** The server combines all analysis results into a final PNG image with a transparent background, featuring semi-transparent outlines and compositional lines.

## How It Works

The project follows a client-server architecture to offload the heavy computer vision processing from the mobile device.

1. **Image Upload (iOS Client):** The user selects a reference photo from their library. The native SwiftUI app sends this image as a `multipart/form-data` POST request to the server's `/analyze` endpoint. This process is seamless and includes a loading state to provide user feedback.

2. **Server-Side Analysis (AI Backend):** A Flask server, containerized with Docker and hosted on Hugging Face Spaces, receives the image. It is then processed through a sophisticated, multi-stage computer vision pipeline using Python libraries like OpenCV and MediaPipe to identify the person, horizon, and perspective lines.

3. **Display Overlay (iOS Client):** The server returns a final PNG template with a transparent background. The iOS app receives this image and displays it as an interactive overlay on the live camera feed. The user can then move, scale, and rotate the template with intuitive gestures to align their shot.

## Tech Stack

* **Frontend:** Swift, SwiftUI, AVFoundation, Combine
* **Backend:** Python, Flask
* **AI / Computer Vision:** OpenCV, MediaPipe, Scikit-learn, NumPy
* **Deployment:** Docker, Hugging Face Spaces
