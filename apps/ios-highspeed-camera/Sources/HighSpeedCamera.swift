import AVFoundation
import CoreMotion
import Foundation
import Network
import UIKit
import VideoToolbox

/// The phone is a sensor for GAMST on a Mac. The camera runs from launch (the Mac shows the
/// live view for framing); Start/Stop marks a take, and every frame carries that flag.
///
/// Two capture modes, neither of which scales pixels on the phone:
/// * full: the camera's smallest format at its highest frame rate (720p240), sent whole for framing;
/// * crop: the largest format at that rate (1080p240) and only a requested sensor rectangle sent at
///   native resolution, so a small fast object keeps its pixels while the frame keeps its height.
/// The phone never rotates pixels: each frame carries the angle that makes it upright, and the Mac
/// rotates. Frames are JPEG-encoded on the VideoToolbox path.
///
/// The Mac drives the same controls with newline-delimited JSON written back on the socket:
/// {"cmd":"record","on":true}, {"cmd":"camera","position":"front"}, {"cmd":"zoom","factor":2},
/// {"cmd":"crop","rect":[x,y,w,h]} (sensor-normalized; [0,0,1,1] returns to full mode).
///
/// Wire format per frame (little endian, 36-byte header): "GMS3" | u32 sequence | f64 capture seconds |
/// u16 width | u16 height | u8 flags (bit 0 recording, bit 1 front camera) | u8 zoom x10 |
/// u16 clockwise degrees to upright | u16 x4 crop rect (sensor-normalized x65535) | u32 jpeg length.
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
    private var frontFlag = false
    private var zoomTenths: UInt8 = 10
    private var commandBuffer = Data()
    private var pending = 0
    /// Bumped on every new connection: sends still in flight on a dropped connection may never
    /// complete, so they must not hold the new connection's backlog full (frames would all drop).
    private var generation = 0
    private var sequence: UInt32 = 0
    private var captured = 0, sent = 0, droppedSend = 0, droppedLate = 0
    private var formatDescription = ""
    private var statusTimer: Timer?
    private var lastStatusAt = Date()
    private var lastCaptured = 0, lastSent = 0

    /// Native pixels a crop may send per frame; ~0.6 MP keeps 240 fps within Wi-Fi bandwidth.
    private let cropPixelBudget = 600_000
    private let maxPending = 120
    private var crop = CGRect(x: 0, y: 0, width: 1, height: 1)  // sensor-normalized; capture queue only
    private var encoder: VTCompressionSession?
    private var encoderSize = (width: 0, height: 0)
    private var pools: [String: CVPixelBufferPool] = [:]
    private var captureAngle = 0
    private var coordinatorBox: AnyObject?
    private var observations: [NSKeyValueObservation] = []
    private let motion = CMMotionManager()

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

    func toggleRecording() { setRecording(!recording) }

    func setRecording(_ on: Bool) {
        guard on != recording else { return }
        lock.lock(); recordingFlag = on; lock.unlock()
        recording = on
        recordingStartedAt = on ? Date() : nil
    }

    /// Front cameras usually top out below 240 fps; the status line shows the rate actually set.
    func switchCamera() { setCamera(position == .back ? .front : .back) }

    func setCamera(_ next: AVCaptureDevice.Position) {
        guard next != position else { return }
        position = next
        lock.lock(); frontFlag = next == .front; lock.unlock()
        reconfigure()
    }

    /// Cycles 1x, 2x, 3x so a small object fills more of the streamed frame.
    func cycleZoom() { setZoom(zoom >= 3 ? 1 : zoom + 1) }

    func setZoom(_ factor: CGFloat) {
        let next = min(max(factor, 1), 3)
        zoom = next
        lock.lock(); zoomTenths = UInt8((next * 10).rounded()); lock.unlock()
        captureQueue.async {
            guard let device = (self.session.inputs.first as? AVCaptureDeviceInput)?.device,
                  (try? device.lockForConfiguration()) != nil else { return }
            device.videoZoomFactor = min(next, device.activeFormat.videoMaxZoomFactor)
            device.unlockForConfiguration()
        }
    }

    /// Sensor-normalized rectangle to stream at native resolution; the full rectangle returns to full mode.
    func setCrop(_ rect: CGRect) {
        let clipped = rect.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        captureQueue.async {
            let wasFull = self.isFull(self.crop), isFull = clipped.isNull || self.isFull(clipped)
            self.crop = isFull ? CGRect(x: 0, y: 0, width: 1, height: 1) : clipped
            if wasFull != isFull { self.reconfigureNow() }
        }
    }

    private func isFull(_ rect: CGRect) -> Bool { rect.width >= 0.999 && rect.height >= 0.999 }

    private func reconfigure() { captureQueue.async { self.reconfigureNow() } }

    private func reconfigureNow() {
        session.stopRunning()
        do {
            try configure()
            session.startRunning()
        } catch {
            DispatchQueue.main.async { self.status = "camera reconfigure failed: \(error.localizedDescription)" }
        }
    }

    /// Commands from GAMST on the Mac, applied exactly like the on-screen buttons.
    private func receiveCommands(on connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self, weak connection] data, _, done, error in
            guard let self, let connection else { return }
            if let data {
                self.commandBuffer.append(data)
                while let newline = self.commandBuffer.firstIndex(of: 0x0A) {
                    let line = self.commandBuffer[self.commandBuffer.startIndex..<newline]
                    self.commandBuffer.removeSubrange(self.commandBuffer.startIndex...newline)
                    guard let command = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
                    DispatchQueue.main.async { self.apply(command) }
                }
            }
            if error == nil && !done { self.receiveCommands(on: connection) }
        }
    }

    private func apply(_ command: [String: Any]) {
        switch command["cmd"] as? String {
        case "record": setRecording(command["on"] as? Bool ?? !recording)
        case "camera": setCamera((command["position"] as? String) == "front" ? .front : .back)
        case "zoom": setZoom(CGFloat((command["factor"] as? NSNumber)?.doubleValue ?? 1))
        case "crop":
            if let rect = command["rect"] as? [NSNumber], rect.count == 4 {
                setCrop(CGRect(x: rect[0].doubleValue, y: rect[1].doubleValue,
                               width: rect[2].doubleValue, height: rect[3].doubleValue))
            }
        default: break
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
                self.lock.lock(); self.generation &+= 1; self.pending = 0; self.lock.unlock()
                self.setConnected(true)
                self.networkQueue.async { self.commandBuffer.removeAll() }
                self.receiveCommands(on: connection)
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
        func maxRate(_ format: AVCaptureDevice.Format) -> Double {
            format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        }
        func pixels(_ format: AVCaptureDevice.Format) -> Int {
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return Int(d.width) * Int(d.height)
        }
        let rate = device.formats.map(maxRate).max() ?? 0
        let candidates = device.formats.filter { maxRate($0) >= rate - 1 }.sorted { pixels($0) < pixels($1) }
        // Full mode: smallest format at the top rate (whole field of view, affordable to send).
        // Crop mode: largest format at the top rate, so the cropped strip keeps native detail.
        let wantsCrop = !isFull(crop)
        guard rate > 0, let format = (wantsCrop ? candidates.last : candidates.first(where: {
            CMVideoFormatDescriptionGetDimensions($0.formatDescription).width >= 1280
        }) ?? candidates.first) else {
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
        let description = "\(position == .back ? "back" : "front") \(dims.width)x\(dims.height)@\(fps) \(wantsCrop ? "crop" : "full")"
        lock.lock(); formatDescription = description; lock.unlock()
        DispatchQueue.main.async { self.followRotation(of: device) }
    }

    /// Tracks the clockwise angle that makes frames upright (sent to the Mac) and keeps the preview level.
    private func followRotation(of device: AVCaptureDevice) {
        observations.removeAll()
        motion.stopDeviceMotionUpdates()
        if #available(iOS 17.0, *) {
            let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
            coordinatorBox = coordinator
            observations.append(coordinator.observe(\.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]) { [weak self] coordinator, _ in
                DispatchQueue.main.async {
                    self?.previewLayer?.connection?.videoRotationAngle = coordinator.videoRotationAngleForHorizonLevelPreview
                }
            })
            observations.append(coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: [.initial, .new]) { [weak self] coordinator, _ in
                self?.setAngle(Int(coordinator.videoRotationAngleForHorizonLevelCapture.rounded()))
            })
        } else {
            // iOS 15/16: gravity gives the physical orientation even with the rotation lock on.
            let front = device.position == .front
            motion.deviceMotionUpdateInterval = 0.2
            motion.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
                guard let self, let g = data?.gravity else { return }
                let angle: Int, orientation: AVCaptureVideoOrientation
                if abs(g.x) > abs(g.y) {
                    // Device top pointing left (gravity along -x): sensor already upright for the back camera.
                    angle = (g.x < 0) != front ? 0 : 180
                    orientation = g.x < 0 ? .landscapeRight : .landscapeLeft
                } else {
                    angle = g.y < 0 ? 90 : 270
                    orientation = g.y < 0 ? .portrait : .portraitUpsideDown
                }
                self.setAngle(angle)
                if let connection = self.previewLayer?.connection, connection.isVideoOrientationSupported {
                    connection.videoOrientation = orientation
                }
            }
        }
    }

    private func setAngle(_ degrees: Int) {
        lock.lock(); captureAngle = (degrees % 360 + 360) % 360; lock.unlock()
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
        let angle = captureAngle, gen = generation
        let flags: UInt8 = (recordingFlag ? 1 : 0) | (frontFlag ? 2 : 0), zoomByte = zoomTenths
        lock.unlock()
        guard let network, !busy else { return }
        let finish = { (ok: Bool) in self.finish(sent: ok, generation: gen) }
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer),
              let (image, region) = cropped(pixels) else { finish(false); return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let width = CVPixelBufferGetWidth(image), height = CVPixelBufferGetHeight(image)
        guard let encoder = jpegEncoder(width: width, height: height) else { finish(false); return }
        let status = VTCompressionSessionEncodeFrame(encoder, imageBuffer: image, presentationTimeStamp: pts,
                                                     duration: .invalid, frameProperties: nil, infoFlagsOut: nil) { status, _, sample in
            guard status == noErr, let sample, let block = CMSampleBufferGetDataBuffer(sample) else { finish(false); return }
            var jpeg = Data(count: CMBlockBufferGetDataLength(block))
            let copied = jpeg.withUnsafeMutableBytes {
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: $0.count, destination: $0.baseAddress!)
            }
            guard copied == kCMBlockBufferNoErr else { finish(false); return }
            self.lock.lock(); self.sequence &+= 1; let sequence = self.sequence; self.lock.unlock()
            var header = Data("GMS3".utf8)
            self.append(&header, sequence.littleEndian)
            self.append(&header, pts.seconds.bitPattern.littleEndian)
            self.append(&header, UInt16(width).littleEndian)
            self.append(&header, UInt16(height).littleEndian)
            header.append(contentsOf: [flags, zoomByte])
            self.append(&header, UInt16(angle).littleEndian)
            for value in [region.minX, region.minY, region.width, region.height] {
                self.append(&header, UInt16((min(max(value, 0), 1) * 65535).rounded()).littleEndian)
            }
            self.append(&header, UInt32(jpeg.count).littleEndian)
            network.send(content: header + jpeg, completion: .contentProcessed { error in finish(error == nil) })
        }
        if status != noErr { finish(false) }
    }

    private func finish(sent ok: Bool, generation gen: Int) {
        lock.lock()
        if gen == generation { pending -= 1 }
        if ok { sent += 1 }
        lock.unlock()
    }

    private func append<T>(_ data: inout Data, _ value: T) {
        withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
    }

    /// Full mode returns the camera buffer as is. Crop mode copies the requested rectangle (even-aligned,
    /// shrunk around its centre along the shorter side to the pixel budget) out of the native frame.
    private func cropped(_ pixels: CVPixelBuffer) -> (CVPixelBuffer, CGRect)? {
        let full = CGRect(x: 0, y: 0, width: 1, height: 1)
        if isFull(crop) { return (pixels, full) }
        let sw = CVPixelBufferGetWidth(pixels), sh = CVPixelBufferGetHeight(pixels)
        var w = Double(crop.width) * Double(sw), h = Double(crop.height) * Double(sh)
        let budget = Double(cropPixelBudget)
        if w * h > budget {
            if w < h { w = max(64, budget / h) } else { h = max(64, budget / w) }
        }
        let cw = min(sw, Int(w) / 2 * 2), ch = min(sh, Int(h) / 2 * 2)
        let cx = min(max(0, Int(Double(crop.midX) * Double(sw) - Double(cw) / 2) / 2 * 2), sw - cw)
        let cy = min(max(0, Int(Double(crop.midY) * Double(sh) - Double(ch) / 2) / 2 * 2), sh - ch)
        guard cw >= 16, ch >= 16, let out = buffer(width: cw, height: ch) else { return nil }
        CVPixelBufferLockBaseAddress(pixels, .readOnly)
        CVPixelBufferLockBaseAddress(out, [])
        defer {
            CVPixelBufferUnlockBaseAddress(out, [])
            CVPixelBufferUnlockBaseAddress(pixels, .readOnly)
        }
        for plane in 0..<2 {
            guard let src = CVPixelBufferGetBaseAddressOfPlane(pixels, plane),
                  let dst = CVPixelBufferGetBaseAddressOfPlane(out, plane) else { return nil }
            let srcRow = CVPixelBufferGetBytesPerRowOfPlane(pixels, plane)
            let dstRow = CVPixelBufferGetBytesPerRowOfPlane(out, plane)
            // Luma is 1 byte per pixel; interleaved CbCr is 2 bytes per 2x2 block.
            let rows = plane == 0 ? ch : ch / 2, top = plane == 0 ? cy : cy / 2, bytes = cw
            for row in 0..<rows {
                memcpy(dst + row * dstRow, src + (top + row) * srcRow + cx, bytes)
            }
        }
        let region = CGRect(x: Double(cx) / Double(sw), y: Double(cy) / Double(sh),
                            width: Double(cw) / Double(sw), height: Double(ch) / Double(sh))
        return (out, region)
    }

    private func buffer(width: Int, height: Int) -> CVPixelBuffer? {
        let key = "\(width)x\(height)"
        if pools[key] == nil {
            if pools.count > 8 { pools.removeAll() }
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
        // Full frames are for framing; crops carry the measurement and get more bits per pixel.
        let quality = width * height > cropPixelBudget ? 0.4 : 0.55
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_Quality, value: quality as CFNumber)
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
