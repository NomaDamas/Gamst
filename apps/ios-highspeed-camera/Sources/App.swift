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
        ZStack {
            CameraPreview(camera: camera).ignoresSafeArea()
            VStack {
                HStack(spacing: 10) {
                    Circle().fill(camera.macConnected ? Color.green : Color.orange).frame(width: 10, height: 10)
                    Text(camera.macConnected ? "GAMST connected" : "connecting to \(host):\(port)")
                        .font(.system(.footnote, design: .rounded).weight(.semibold))
                    Spacer()
                    if let started = camera.recordingStartedAt {
                        TimelineView(.periodic(from: started, by: 0.1)) { context in
                            Text(String(format: "● REC %.1fs", context.date.timeIntervalSince(started)))
                                .font(.system(.footnote, design: .monospaced).weight(.bold))
                                .foregroundColor(.red)
                        }
                    }
                }
                .padding(10)
                .background(.ultraThinMaterial, in: Capsule())
                Spacer()
                HStack(alignment: .bottom, spacing: 14) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            TextField("Mac IP", text: $host).textFieldStyle(.roundedBorder).frame(width: 150)
                                .keyboardType(.numbersAndPunctuation)
                                .onSubmit { camera.connect(host: host, port: UInt16(port) ?? 9123) }
                            TextField("port", text: $port).textFieldStyle(.roundedBorder).frame(width: 64)
                                .keyboardType(.numberPad)
                            Button(camera.position == .back ? "Front" : "Back") { camera.switchCamera() }
                                .buttonStyle(.bordered)
                            Button("\(Int(camera.zoom))x") { camera.cycleZoom() }.buttonStyle(.bordered)
                        }
                        Text(camera.status).font(.system(.caption2, design: .monospaced)).foregroundColor(.white)
                    }
                    Spacer()
                    Button(action: camera.toggleRecording) {
                        ZStack {
                            Circle().stroke(Color.white, lineWidth: 4).frame(width: 74, height: 74)
                            RoundedRectangle(cornerRadius: camera.recording ? 8 : 30)
                                .fill(Color.red)
                                .frame(width: camera.recording ? 30 : 60, height: camera.recording ? 30 : 60)
                        }
                    }
                    .accessibilityLabel(camera.recording ? "Stop" : "Start")
                }
                .padding(12)
                .background(Color.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 16))
            }
            .padding()
            .foregroundColor(.white)
        }
        .onAppear { camera.activate(host: host, port: UInt16(port) ?? 9123) }
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
