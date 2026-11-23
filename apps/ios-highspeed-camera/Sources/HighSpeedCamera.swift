import AVFoundation
import Foundation
import Network
import VideoToolbox

/// Captures at the camera's highest frame rate (240 fps on the back camera) and streams
/// upright, downscaled colour JPEG frames with their sensor presentation timestamps to GAMST
/// on a Mac. The phone is only a sensor: tracking, rotation counting, records and alarms all
/// run in GAMST. Scaling, rotation and JPEG encoding run on the VideoToolbox hardware path.
///
/// Wire format per frame (little endian): "GMST" | u32 sequence | f64 capture seconds |
/// u16 width | u16 height | u32 jpeg length | jpeg bytes. Frames are already upright.
final class HighSpeedCamera: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var status = "idle"
    @Published var running = false
    @Published private(set) var position: AVCaptureDevice.Position = .back

    let session = AVCaptureSession()
    private let captureQueue = DispatchQueue(label: "gamst.capture", qos: .userInteractive)
    private var connection: NWConnection?
    private let lock = NSLock()
    private var pending = 0
    private var sequence: UInt32 = 0
    private var captured = 0, sent = 0, droppedSend = 0, droppedLate = 0
    private var formatDescription = ""
    private var statusTimer: Timer?
    private var lastStatusAt = Date()
    private var lastCaptured = 0, lastSent = 0

    private let outWidth = 640, outHeight = 360
    private let maxPending = 8
    private var transfer: VTPixelTransferSession?
    private var rotator: VTPixelRotationSession?
    private var encoder: VTCompressionSession?
    private var encoderSize = (width: 0, height: 0)
    private var pools: [String: CVPixelBufferPool] = [:]
    private var appliedAngle = -1
    private var captureAngle = 0
    private var coordinator: AVCaptureDevice.RotationCoordinator?
    private var observations: [NSKeyValueObservation] = []
    weak var previewLayer: AVCaptureVideoPreviewLayer?

    func start(host: String, port: UInt16) {
        guard !running else { return }
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { DispatchQueue.main.async { self.status = "camera permission denied" }; return }
            self.captureQueue.async {
                do {
                    try self.configure()
                    self.connect(host: host, port: port)
                    self.session.startRunning()
                    DispatchQueue.main.async {
                        self.running = true
                        self.startStatusTimer()
                    }
                } catch {
                    DispatchQueue.main.async { self.status = "setup failed: \(error.localizedDescription)" }
                }
            }
        }
    }

    /// Front cameras usually top out below 240 fps; the status line shows the rate actually set.
    func switchCamera() {
        let next: AVCaptureDevice.Position = position == .back ? .front : .back
        position = next
        guard running else { return }
        captureQueue.async {
            self.session.stopRunning()
            do {
                try self.configure()
                self.session.startRunning()
            } catch {
                DispatchQueue.main.async { self.status = "camera switch failed: \(error.localizedDescription)" }
            }
        }
    }

    @Published private(set) var zoom: CGFloat = 1

    /// Cycles 1x, 2x, 3x so a small object fills more of the streamed 640x360 frame.
    func cycleZoom() {
        let next: CGFloat = zoom >= 3 ? 1 : zoom + 1
        zoom = next
        captureQueue.async {
            guard let device = (self.session.inputs.first as? AVCaptureDeviceInput)?.device,
                  (try? device.lockForConfiguration()) != nil else { return }
            device.videoZoomFactor = min(next, device.activeFormat.videoMaxZoomFactor)
            device.unlockForConfiguration()
        }
    }

    func stop() {
        captureQueue.async {
            self.session.stopRunning()
            self.connection?.cancel()
            self.connection = nil
            DispatchQueue.main.async {
                self.running = false
                self.statusTimer?.invalidate()
                self.status = "stopped"
            }
        }
    }

    private func configure() throws {
        let (position, zoom) = DispatchQueue.main.sync { (self.position, self.zoom) }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            throw NSError(domain: "gamst", code: 1, userInfo: [NSLocalizedDescriptionKey: "no \(position == .back ? "back" : "front") camera"])
        }
        // Highest frame rate the camera offers (240 on the back camera), then the smallest format at that
        // rate; capture rate, not resolution, decides countability.
        func maxRate(_ format: AVCaptureDevice.Format) -> Double {
            format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        }
        let rate = device.formats.map(maxRate).max() ?? 0
        let candidates = device.formats.filter { maxRate($0) >= rate - 1 }.sorted {
            let a = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
            let b = CMVideoFormatDescriptionGetDimensions($1.formatDescription)
            return Int(a.width) * Int(a.height) < Int(b.width) * Int(b.height)
        }
        guard let format = candidates.first(where: {
            CMVideoFormatDescriptionGetDimensions($0.formatDescription).width >= 1280
        }) ?? candidates.first, rate > 0 else {
            throw NSError(domain: "gamst", code: 2, userInfo: [NSLocalizedDescriptionKey: "no usable video format"])
        }
        let fps = Int32(rate.rounded())
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw NSError(domain: "gamst", code: 3, userInfo: [NSLocalizedDescriptionKey: "cannot add camera input"])
        }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: captureQueue)
        guard session.canAddOutput(output) else {
            throw NSError(domain: "gamst", code: 4, userInfo: [NSLocalizedDescriptionKey: "cannot add video output"])
        }
        session.addOutput(output)
        try device.lockForConfiguration()
        device.activeFormat = format
        let frame = CMTime(value: 1, timescale: fps)
        device.activeVideoMinFrameDuration = frame
        device.activeVideoMaxFrameDuration = frame
        // Auto exposure keeps the picture bright; capping the shutter keeps a fast coin sharp.
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
            device.activeMaxExposureDuration = CMTimeMaximum(format.minExposureDuration, CMTime(value: 1, timescale: 1000))
        }
        if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
        device.videoZoomFactor = min(zoom, format.videoMaxZoomFactor)
        device.unlockForConfiguration()
        session.commitConfiguration()
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let description = "\(position == .back ? "back" : "front") \(dims.width)x\(dims.height)@\(fps) -> \(outWidth)x\(outHeight)"
        lock.lock(); formatDescription = description; lock.unlock()
        DispatchQueue.main.async { self.followRotation(of: device) }
    }

    /// Keeps the preview and the streamed frames level with the horizon however the phone is held.
    private func followRotation(of device: AVCaptureDevice) {
        observations.removeAll()
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        self.coordinator = coordinator
        observations.append(coordinator.observe(\.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]) { [weak self] coordinator, _ in
            DispatchQueue.main.async {
                self?.previewLayer?.connection?.videoRotationAngle = coordinator.videoRotationAngleForHorizonLevelPreview
            }
        })
        observations.append(coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: [.initial, .new]) { [weak self] coordinator, _ in
            guard let self else { return }
            let angle = (Int(coordinator.videoRotationAngleForHorizonLevelCapture.rounded()) % 360 + 360) % 360
            self.lock.lock(); self.captureAngle = angle; self.lock.unlock()
        })
    }

    private func connect(host: String, port: UInt16) {
        let params = NWParameters.tcp
        if let tcp = params.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options { tcp.noDelay = true }
        let connection = NWConnection(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!, using: params)
        connection.stateUpdateHandler = { state in
            DispatchQueue.main.async {
                switch state {
                case .failed(let error): self.status = "network failed: \(error)"
                case .waiting(let error): self.status = "waiting for Mac: \(error)"
                default: break
                }
            }
        }
        connection.start(queue: DispatchQueue(label: "gamst.network"))
        self.connection = connection
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lock.lock(); droppedLate += 1; lock.unlock()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lock.lock()
        captured += 1
        let busy = pending >= maxPending
        if busy { droppedSend += 1 } else { pending += 1 }
        let angle = captureAngle
        lock.unlock()
        if busy { return }
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer), let network = self.connection,
              let upright = uprightFrame(pixels, angle: angle) else { finish(sent: false); return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let width = CVPixelBufferGetWidth(upright), height = CVPixelBufferGetHeight(upright)
        guard let encoder = jpegEncoder(width: width, height: height) else { finish(sent: false); return }
        let status = VTCompressionSessionEncodeFrame(encoder, imageBuffer: upright, presentationTimeStamp: pts,
                                                     duration: .invalid, frameProperties: nil, infoFlagsOut: nil) { status, _, sample in
            guard status == noErr, let sample, let block = CMSampleBufferGetDataBuffer(sample) else { self.finish(sent: false); return }
            var jpeg = Data(count: CMBlockBufferGetDataLength(block))
            let copied = jpeg.withUnsafeMutableBytes {
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: $0.count, destination: $0.baseAddress!)
            }
            guard copied == kCMBlockBufferNoErr else { self.finish(sent: false); return }
            self.lock.lock(); self.sequence &+= 1; let sequence = self.sequence; self.lock.unlock()
            var header = Data("GMST".utf8)
            self.append(&header, sequence.littleEndian)
            self.append(&header, pts.seconds.bitPattern.littleEndian)
            self.append(&header, UInt16(width).littleEndian)
            self.append(&header, UInt16(height).littleEndian)
            self.append(&header, UInt32(jpeg.count).littleEndian)
            network.send(content: header + jpeg, completion: .contentProcessed { error in self.finish(sent: error == nil) })
        }
        if status != noErr { finish(sent: false) }
    }

    private func finish(sent ok: Bool) {
        lock.lock()
        pending -= 1
        if ok { sent += 1 }
        lock.unlock()
    }

    private func append<T>(_ data: inout Data, _ value: T) {
        withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
    }

    private func buffer(width: Int, height: Int) -> CVPixelBuffer? {
        let key = "\(width)x\(height)"
        if pools[key] == nil {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
            pools[key] = pool
        }
        guard let pool = pools[key] else { return nil }
        var pixels: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixels)
        return pixels
    }

    /// Hardware downscale, then hardware rotation to the horizon-level angle.
    private func uprightFrame(_ pixels: CVPixelBuffer, angle: Int) -> CVPixelBuffer? {
        if transfer == nil { VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer) }
        guard let transfer, let scaled = buffer(width: outWidth, height: outHeight),
              VTPixelTransferSessionTransferImage(transfer, from: pixels, to: scaled) == noErr else { return nil }
        if angle == 0 { return scaled }
        if rotator == nil { VTPixelRotationSessionCreate(nil, &rotator) }
        guard let rotator else { return nil }
        if angle != appliedAngle {
            let rotation: CFString = angle == 90 ? kVTRotation_CW90 : angle == 180 ? kVTRotation_180 : kVTRotation_CCW90
            VTSessionSetProperty(rotator, key: kVTPixelRotationPropertyKey_Rotation, value: rotation)
            appliedAngle = angle
        }
        let quarter = angle == 90 || angle == 270
        guard let rotated = buffer(width: quarter ? outHeight : outWidth, height: quarter ? outWidth : outHeight),
              VTPixelRotationSessionRotateImage(rotator, scaled, rotated) == noErr else { return nil }
        return rotated
    }

    private func jpegEncoder(width: Int, height: Int) -> VTCompressionSession? {
        if let encoder, encoderSize == (width, height) { return encoder }
        if let encoder { VTCompressionSessionInvalidate(encoder) }
        encoder = nil
        var session: VTCompressionSession?
        guard VTCompressionSessionCreate(allocator: nil, width: Int32(width), height: Int32(height),
                                         codecType: kCMVideoCodecType_JPEG, encoderSpecification: nil,
                                         imageBufferAttributes: nil, compressedDataAllocator: nil,
                                         outputCallback: nil, refcon: nil, compressionSessionOut: &session) == noErr,
              let session else { return nil }
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_Quality, value: 0.5 as CFNumber)
        encoder = session
        encoderSize = (width, height)
        return session
    }

    private func startStatusTimer() {
        statusTimer?.invalidate()
        lastStatusAt = Date()
        statusTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            self.lock.lock()
            let (c, s, ds, dl, format) = (self.captured, self.sent, self.droppedSend, self.droppedLate, self.formatDescription)
            self.lock.unlock()
            let now = Date(), dt = now.timeIntervalSince(self.lastStatusAt)
            let captureFps = Double(c - self.lastCaptured) / dt, sendFps = Double(s - self.lastSent) / dt
            self.lastCaptured = c; self.lastSent = s; self.lastStatusAt = now
            self.status = String(format: "%@\ncapture %.0f fps · sent %.0f fps\nsent %d · dropped send %d · late %d",
                                 format, captureFps, sendFps, s, ds, dl)
        }
    }
}
