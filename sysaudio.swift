// Stream system audio output (what's playing in the AirPods) to stdout as 16 kHz mono float32.
// Uses ScreenCaptureKit, so it needs Screen & System Audio Recording permission.
// Exit 3 if capture can't start (usually missing permission).
import Foundation
import ScreenCaptureKit
import CoreMedia

final class Sink: NSObject, SCStreamOutput, SCStreamDelegate {
    let out = FileHandle.standardOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        var abl = AudioBufferList()
        var block: CMBlockBuffer?
        guard CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sb, bufferListSizeNeededOut: nil, bufferListOut: &abl,
            bufferListSize: MemoryLayout<AudioBufferList>.size, blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil, flags: 0, blockBufferOut: &block) == noErr,
              let data = abl.mBuffers.mData else { return }
        out.write(Data(bytes: data, count: Int(abl.mBuffers.mDataByteSize)))
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("stopped: \(error)\n".data(using: .utf8)!)
        exit(3)
    }
}

let sink = Sink()
var keep: SCStream?

Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else { exit(3) }
        let cfg = SCStreamConfiguration()
        cfg.capturesAudio = true
        cfg.excludesCurrentProcessAudio = true
        cfg.sampleRate = 16_000
        cfg.channelCount = 1
        cfg.width = 2
        cfg.height = 2
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let stream = SCStream(filter: SCContentFilter(display: display, excludingWindows: []),
                              configuration: cfg, delegate: sink)
        try stream.addStreamOutput(sink, type: .audio, sampleHandlerQueue: DispatchQueue(label: "audio"))
        try await stream.startCapture()
        keep = stream
    } catch {
        FileHandle.standardError.write("capture failed: \(error)\n".data(using: .utf8)!)
        exit(3)
    }
}
signal(SIGPIPE) { _ in exit(0) }
RunLoop.main.run()
