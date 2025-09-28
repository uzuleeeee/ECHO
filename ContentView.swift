import SwiftUI
import AVFoundation
import Photos
import Combine
import WebKit // --- NEW: Import WebKit for the GIF view ---

// Main ContentView that holds the camera interface
struct ContentView: View {
    @StateObject private var cameraController = CameraController()
    @State private var overlayImage: UIImage?
    @State private var showImagePicker = false

    // State for server communication
    @State private var selectedImage: UIImage?
    @State private var isLoading = false
    @State private var errorMessage: String?
    
    // State for showing/hiding camera settings
    @State private var showCameraSettings = false
    
    // --- IMPORTANT: Use your actual Hugging Face URL ---
    let serverURL = URL(string: "https://uzulee-photo-guide-api.hf.space/analyze")!

    // State for overlay transformations
    @State private var overlayScale: CGFloat = 1.0
    @State private var overlayRotation = Angle.zero
    @State private var overlayPosition = CGSize.zero

    // Gesture state for live updates
    @GestureState private var gestureScale: CGFloat = 1.0
    @GestureState private var gestureRotation = Angle.zero
    @GestureState private var gesturePosition = CGSize.zero

    // MARK: - Gestures
    
    var dragGesture: some Gesture {
        DragGesture()
            .updating($gesturePosition) { value, state, _ in
                state = value.translation
            }
            .onEnded { value in
                overlayPosition.width += value.translation.width
                overlayPosition.height += value.translation.height
            }
    }
    
    var magnificationAndRotationGesture: some Gesture {
        MagnificationGesture()
            .updating($gestureScale) { value, state, _ in
                state = value
            }
            .onEnded { value in
                overlayScale *= value
            }
            .simultaneously(with: RotationGesture()
                .updating($gestureRotation) { value, state, _ in
                    state = value
                }
                .onEnded { value in
                    overlayRotation += value
                }
            )
    }

    var body: some View {
        ZStack {
            // Main Camera UI
            ZStack {
                CameraPreview(session: cameraController.session, isSessionRunning: cameraController.isSessionRunning)
                    .ignoresSafeArea()
                    .overlay(
                        Group {
                            if let overlayImage = overlayImage {
                                Image(uiImage: overlayImage)
                                    .resizable()
                                    .scaledToFill()
                                    .scaleEffect(overlayScale * gestureScale)
                                    .rotationEffect(overlayRotation + gestureRotation)
                                    .offset(x: overlayPosition.width + gesturePosition.width, y: overlayPosition.height + gesturePosition.height)
                                    .opacity(0.8)
                                    .gesture(dragGesture)
                                    .gesture(magnificationAndRotationGesture)
                                    .ignoresSafeArea()
                            }
                        }
                    )
                
                // This view detects taps to dismiss the settings panel
                if showCameraSettings {
                    Color.clear
                        .contentShape(Rectangle())
                        .ignoresSafeArea()
                        .onTapGesture {
                            showCameraSettings = false
                        }
                }
                
                // UI Controls overlayed on top of the camera preview
                VStack {
                    // Top controls
                    HStack {
                        Button(action: {
                            cameraController.stopSession()
                            showImagePicker = true
                        }) {
                            Image("uploadIcon")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 40, height: 40)
                                .foregroundColor(.white)
                        }
                        
                        if overlayImage != nil {
                            Button(action: resetOverlayTransforms) {
                                Image(systemName: "arrow.counterclockwise")
                                    .font(.system(size: 28, weight: .bold))
                                    .foregroundColor(.white)
                                    .padding()
                            }
                            .padding(.leading, 5)
                        }
                        
                        Spacer()
                        
                        Button(action: {
                            showCameraSettings.toggle()
                        }) {
                            Image("settingsIcon")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 40, height: 40)
                                .foregroundColor(.white)
                        }

                        Button(action: {
                            cameraController.switchCamera()
                        }) {
                            Image("flipIcon")
                                .resizable()
                                .scaledToFit()
                                .frame(width: 40, height: 40)
                                .foregroundColor(.white)
                        }
                    }.padding(.horizontal)
                    
                    if let errorMessage = errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.red.opacity(0.8))
                            .cornerRadius(10)
                            .padding(.top)
                    }
                    
                    Spacer()

                    if showCameraSettings {
                        VStack(spacing: 20) {
                            HStack {
                                Image(systemName: "minus.magnifyingglass")
                                    .foregroundColor(.white)
                                Slider(value: $cameraController.zoomFactor, in: 1...10)
                                    .accentColor(.white)
                                Image(systemName: "plus.magnifyingglass")
                                    .foregroundColor(.white)
                            }
                            .padding(.horizontal)
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(15)
                            
                            HStack {
                                Image(systemName: "sun.min")
                                    .foregroundColor(.white)
                                Slider(value: $cameraController.exposureValue, in: -2...2)
                                    .accentColor(.white)
                                Image(systemName: "sun.max.fill")
                                    .foregroundColor(.white)
                            }
                            .padding(.horizontal)
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(15)
                        }
                        .padding()
                        .onTapGesture {}
                    }

                    // Bottom controls: Capture button
                    HStack(alignment: .center) {
                        Spacer()
                        Button(action: {
                            cameraController.capturePhoto()
                        }) {
                            ZStack {
                                Circle()
                                    .fill(Color.white)
                                    .frame(width: 70, height: 70)
                                Circle()
                                    .stroke(Color(red: 147/255, green: 112/255, blue: 219/255), lineWidth: 4)
                                    .frame(width: 80, height: 80)
                            }
                        }
                        Spacer()
                    }
                    .padding()
                }
            }
            
            // Loading overlay
            if isLoading {
                Color.black.opacity(0.5)
                    .ignoresSafeArea()
                VStack {
                    // --- THESE ARE THE CHANGES ---
                    GIFView("loadingGif")
                        .frame(width: 130, height: 130) // Made the GIF larger
                        .cornerRadius(20)              // Added rounded corners
                    Text("Analyzing Composition...")
                        .foregroundColor(.white)
                        .font(.headline)
                        .padding(.top)
                }
            }
        }
        .onAppear {
            cameraController.checkPermissions()
            cameraController.startSession()
        }
        .alert(isPresented: $cameraController.showPermissionAlert) {
            Alert(
                title: Text("Camera Access Denied"),
                message: Text("Please enable camera access in Settings to use this feature."),
                primaryButton: .default(Text("Settings"), action: {
                    UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!)
                }),
                secondaryButton: .cancel()
            )
        }
        .sheet(isPresented: $showImagePicker, onDismiss: {
            cameraController.startSession()
        }) {
            ImagePicker(image: $selectedImage, cameraController: cameraController)
        }
        .onChange(of: selectedImage) { newImage in
            guard let image = newImage else { return }
            overlayImage = nil
            errorMessage = nil
            resetOverlayTransforms()
            sendImageToServer(image)
        }
    }
    
    private func resetOverlayTransforms() {
        overlayScale = 1.0
        overlayRotation = .zero
        overlayPosition = .zero
    }
    
    // Networking Logic
    func sendImageToServer(_ image: UIImage) {
        isLoading = true
        
        guard let imageData = image.pngData() else {
            errorMessage = "Could not convert image to PNG data."
            isLoading = false
            return
        }
        
        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLoading = false
                
                if let error = error {
                    self.errorMessage = "Network request failed: \(error.localizedDescription)"
                    return
                }
                
                guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                    self.errorMessage = "Server returned an error. Status code: \((response as? HTTPURLResponse)?.statusCode ?? 0)"
                    return
                }
                
                guard let data = data, let returnedImage = UIImage(data: data) else {
                    self.errorMessage = "Could not decode template from server."
                    return
                }
                
                self.overlayImage = returnedImage
            }
        }.resume()
    }
}


// A helper view to display animated GIFs
struct GIFView: UIViewRepresentable {
    private let name: String

    init(_ name: String) {
        self.name = name
    }

    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView()
        
        // Find the URL for the GIF in the app's bundle
        guard let url = Bundle.main.url(forResource: name, withExtension: "gif") else {
            print("Error: GIF not found")
            return webView
        }
        
        do {
            let data = try Data(contentsOf: url)
            webView.load(
                data,
                mimeType: "image/gif",
                characterEncodingName: "UTF-8",
                baseURL: url.deletingLastPathComponent()
            )
            // Make the web view transparent so the GIF appears to float
            webView.isOpaque = false
            webView.backgroundColor = .clear
            webView.scrollView.backgroundColor = .clear
        } catch {
            print("Error loading GIF data: \(error)")
        }
        
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        // Reload the GIF if the view updates
        uiView.reload()
    }
}


// Manages all camera-related logic (AVFoundation)
class CameraController: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
    @Published var session = AVCaptureSession()
    @Published var output = AVCapturePhotoOutput()
    @Published var previewLayer: AVCaptureVideoPreviewLayer!
    
    @Published var lastCapturedImage: UIImage?
    @Published var showPermissionAlert = false
    @Published var isSessionRunning = false
    
    private var currentDevice: AVCaptureDevice?
    @Published var zoomFactor: CGFloat = 1.0 {
        didSet { updateZoom() }
    }
    @Published var exposureValue: Float = 0.0 {
        didSet { updateExposure() }
    }
    
    private var backCamera: AVCaptureDevice?
    private var frontCamera: AVCaptureDevice?
    
    override init() {
        super.init()
        setupSession()
    }
    
    func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if !granted {
                        self?.showPermissionAlert = true
                    }
                }
            }
        default:
            showPermissionAlert = true
        }
    }

    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .photo
        
        let discoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .unspecified)
        for device in discoverySession.devices {
            if device.position == .back { backCamera = device }
            else if device.position == .front { frontCamera = device }
        }
        
        currentDevice = backCamera
        guard let device = currentDevice else {
            print("Error: No back camera found.")
            session.commitConfiguration()
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input) }
            if session.canAddOutput(output) { session.addOutput(output) }
            session.commitConfiguration()
        } catch {
            print("Error setting up camera session: \(error.localizedDescription)")
        }
    }
    
    func startSession() {
        if !session.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.session.startRunning()
                DispatchQueue.main.async {
                    self?.isSessionRunning = true
                }
            }
        }
    }
    
    func stopSession() {
        if session.isRunning {
            session.stopRunning()
            isSessionRunning = false
        }
    }
    
    func switchCamera() {
        session.beginConfiguration()
        guard let currentInput = session.inputs.first as? AVCaptureDeviceInput else { return }
        
        session.removeInput(currentInput)
        
        let newDevice = (currentInput.device.position == .back) ? frontCamera : backCamera
        
        guard let device = newDevice else {
            print("Error: Could not find other camera.")
            session.commitConfiguration()
            return
        }
        
        do {
            let newInput = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(newInput) { session.addInput(newInput) }
            currentDevice = device
            zoomFactor = 1.0
            exposureValue = 0.0
        } catch {
            print("Error switching camera: \(error.localizedDescription)")
            session.addInput(currentInput)
        }
        
        session.commitConfiguration()
    }

    func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        output.capturePhoto(with: settings, delegate: self)
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("Error capturing photo: \(error.localizedDescription)")
            return
        }
        
        guard let imageData = photo.fileDataRepresentation(), let image = UIImage(data: imageData) else {
            print("Error: Could not get image data.")
            return
        }
        
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized else { return }
            PHPhotoLibrary.shared().performChanges({
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: imageData, options: nil)
            }) { [weak self] success, error in
                if success {
                    DispatchQueue.main.async {
                        self?.lastCapturedImage = image
                    }
                } else if let error = error {
                    print("Error saving photo: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func updateZoom() {
        guard let device = currentDevice else { return }
        do {
            try device.lockForConfiguration()
            let newZoom = max(1.0, min(zoomFactor, device.activeFormat.videoMaxZoomFactor))
            device.videoZoomFactor = newZoom
            device.unlockForConfiguration()
        } catch {
            print("Error locking device for zoom configuration: \(error)")
        }
    }
    
    private func updateExposure() {
        guard let device = currentDevice else { return }
        do {
            try device.lockForConfiguration()
            let minEV = device.minExposureTargetBias
            let maxEV = device.maxExposureTargetBias
            let newExposure = max(minEV, min(exposureValue, maxEV))
            device.setExposureTargetBias(newExposure, completionHandler: nil)
            device.unlockForConfiguration()
        } catch {
            print("Error locking device for exposure configuration: \(error)")
        }
    }
}

// A SwiftUI UIViewRepresentable to wrap the AVCaptureVideoPreviewLayer
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let isSessionRunning: Bool

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.frame = view.frame
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        if let previewLayer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            previewLayer.frame = uiView.bounds
        }
    }
}

// A UIViewControllerRepresentable to wrap UIImagePickerController for use in SwiftUI
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    let cameraController: CameraController
    @Environment(\.presentationMode) private var presentationMode

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.presentationMode.wrappedValue.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

// Preview Provider
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

