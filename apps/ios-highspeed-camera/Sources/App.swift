import AVFoundation
import SwiftUI

@main
struct GamstHighSpeedCameraApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var camera = HighSpeedCamera()
    @AppStorage("gamstHost") private var host = "192.168.0.3"
    @AppStorage("gamstPort") private var port = "9123"

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            CameraPreview(camera: camera).ignoresSafeArea()
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    TextField("Mac IP", text: $host).textFieldStyle(.roundedBorder).frame(width: 160)
                        .keyboardType(.numbersAndPunctuation).disabled(camera.running)
                    TextField("port", text: $port).textFieldStyle(.roundedBorder).frame(width: 70)
                        .keyboardType(.numberPad).disabled(camera.running)
                    Button(camera.running ? "Stop" : "Start 240 fps") {
                        if camera.running { camera.stop() } else { camera.start(host: host, port: UInt16(port) ?? 9123) }
                    }
                    .buttonStyle(.borderedProminent)
                    Button(camera.position == .back ? "Front camera" : "Back camera") { camera.switchCamera() }
                        .buttonStyle(.bordered)
                    Button("\(Int(camera.zoom))x") { camera.cycleZoom() }
                        .buttonStyle(.bordered)
                }
                Text(camera.status).font(.system(.caption, design: .monospaced)).foregroundColor(.white)
            }
            .padding(12)
            .background(Color.black.opacity(0.55))
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let camera: HighSpeedCamera

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.preview.session = camera.session
        view.preview.videoGravity = .resizeAspect
        camera.previewLayer = view.preview
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}
}
