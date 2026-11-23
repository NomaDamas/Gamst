import AVFoundation
import Foundation
import Network
import VideoToolbox

/// The phone is a sensor for GAMST on a Mac. The camera runs from launch (the Mac shows the
/// live view for framing); Start/Stop marks a take, and every frame carries that flag.
/// Capture runs at the camera's highest frame rate (240 fps on the back camera); scaling,
/// horizon-level rotation and JPEG encoding run on the VideoToolbox hardware path.
///
/// Wire format per frame (little endian, 28-byte header): "GMS2" | u32 sequence |
/// f64 capture seconds | u16 width | u16 height | u8 flags (bit 0 = recording) | 3 bytes reserved |
/// u32 jpeg length | jpeg bytes. Frames are already upright.
final class HighSpeedCamera: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var status = "starting camera"
    @Published private(set) var macConnected = false
    @Published private(set) var recording = false
    @Published private(set) var recordingStartedAt: Date?
    @Published private(set) var position: AVCaptureDevice.Position = .back
    @Published private(set) var zoom: CGFloat = 1

    let session = AVCaptureSession()
    weak var previewLayer: AVCaptureVideoPreviewLayer?

    private let captureQueue = DispatchQueue(label: "gamst.capture", qos: .userInteractive)
    private let networkQueue = DispatchQueue(label: "gamst.network")
    private let lock = NSLock()
    private var connection: NWConnection?
    private var endpoint: (host: String, port: UInt16)?
    private var connected = false
    private var recordingFlag = false
    private var pending = 0
    private var sequence: UInt32 = 0
    private var captured = 0, sent = 0, droppedSend = 0, droppedLate = 0
    private var formatDescription = ""
    private var statusTimer: Timer?
    private var lastStatusAt = Date()
    private var lastCaptured = 0, lastSent = 0

    private let outWidth = 960, outHeight = 540
    // Half a second of in-flight frames absorbs Wi-Fi jitter; a cap of 8 dropped ~25% of 240 fps frames.
    private let maxPending = 120
    private var transfer: VTPixelTransferSession?
    private var rotator: VTPixelRotationSession?
    private var encoder: VTCompressionSession?
    private var encoderSize = (width: 0, height: 0)
    private var pools: [String: CVPixelBufferPool] = [:]
    private var appliedAngle = -1
    private var captureAngle = 0
    private var coordinator: AVCaptureDevice.RotationCoordinator?
    private var observations: [NSKeyValueObservation] = []

    /// Starts the camera and the stream to the Mac; called once when the app appears.
    func activate(host: String, port: UInt16) {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { DispatchQueue.main.async { self.status = "camera permission denied" }; return }
            self.captureQueue.async {
                do {
                    try self.configure()
                    self.session.startRunning()
                } catch {
                    DispatchQueue.main.async { self.status = "camera setup failed: \(error.localizedDescription)" }
                }
            }
            DispatchQueue.main.async {
                self.connect(host: host, port: port)
                self.startStatusTimer()
            }
        }
    }

    func toggleRecording() {
        let next = !recording
        lock.lock(); recordingFlag = next; lock.unlock()
        recording = next
        recordingStartedAt = next ? Date() : nil
    }

    /// Front cameras usually top out below 240 fps; the status line shows the rate actually set.
    func switchCamera() {
        position = position == .back ? .front : .back
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

    /// Cycles 1x, 2x, 3x so a small object fills more of the streamed frame.
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

    // MARK: Network

    /// Connects (or reconnects) to GAMST on the Mac; a refused or dropped connection retries every second.
    func connect(host: String, port: UInt16) {
        endpoint = (host, port)
        connection?.cancel()
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { status = "invalid port"; return }
        let params = NWParameters.tcp
        if let tcp = params.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options { tcp.noDelay = true }
        let connection = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: params)
        connection.stateUpdateHandler = { [weak self, weak connection] state in
            guard let self, let connection else { return }
            switch state {
            case .ready:
                self.setConnected(true)
            case .failed, .waiting:
                self.setConnected(false)
                connection.cancel()
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                    guard self.connection === connection, let endpoint = self.endpoint else { return }
                    self.connect(host: endpoint.host, port: endpoint.port)
                }
            case .cancelled:
                self.setConnected(false)
            default:
                break
            }
        }
        lock.lock(); self.connection = connection; lock.unlock()
        connection.start(queue: networkQueue)
    }

    private func setConnected(_ value: Bool) {
        lock.lock(); connected = value; lock.unlock()
        DispatchQueue.main.async { self.macConnected = value }
    }

    // MARK: Camera

    private func configure() throws {
        let (position, zoom) = DispatchQueue.main.sync { (self.position, self.zoom) }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            throw NSError(domain: "gamst", code: 1, userInfo: [NSLocalizedDescriptionKey: "no \(position == .back ? "back" : "front") camera"])
        }
        // Highest frame rate the camera offers, then the smallest format at that rate that still
        // covers the streamed size; capture rate, not resolution, decides countability.
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

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lock.lock(); droppedLate += 1; lock.unlock()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        lock.lock()
        captured += 1
        let network = connected ? self.connection : nil
        let busy = network != nil && pending >= maxPending
        if busy { droppedSend += 1 } else if network != nil { pending += 1 }
        let angle = captureAngle, flags: UInt8 = recordingFlag ? 1 : 0
        lock.unlock()
        guard let network, !busy else { return }
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer),
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
            var header = Data("GMS2".utf8)
            self.append(&header, sequence.littleEndian)
            self.append(&header, pts.seconds.bitPattern.littleEndian)
            self.append(&header, UInt16(width).littleEndian)
            self.append(&header, UInt16(height).littleEndian)
            header.append(contentsOf: [flags, 0, 0, 0])
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
            self.status = String(format: "%@\ncapture %.0f fps · sent %.0f fps · dropped %d · late %d",
                                 format, captureFps, sendFps, ds, dl)
        }
    }
}
