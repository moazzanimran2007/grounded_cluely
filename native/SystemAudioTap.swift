// SystemAudioTap: capture this Mac's system output audio with a Core Audio
// process tap (no virtual audio driver needed) and write it to stdout as raw
// 16 bit PCM, mono. All human readable diagnostics go to stderr so that stdout
// stays a clean binary audio stream that another process can read.
//
// Why a process tap and not ScreenCaptureKit: Apple's process tap API is the
// native path for audio only capture and is steady over long captures, which
// matters for meetings that run for an hour. It needs macOS 14.4 or newer.
//
// Why mono 16 bit PCM on stdout: that is the simplest thing the Python side can
// consume. The tap hands us float samples, and we convert them to 16 bit
// integers here so Python receives exactly the format Deepgram expects.

import AudioToolbox
import CoreAudio
import Foundation

// ----- small helpers ----------------------------------------------------------

// Write a line to stderr. Used for all diagnostics.
func logLine(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// Write raw bytes to stdout (file descriptor 1) using the plain C write call,
// looping until everything is written. Safe to call from the audio callback.
func emitBytes(_ base: UnsafeRawPointer, _ count: Int) {
    var written = 0
    while written < count {
        let n = write(1, base + written, count - written)
        if n <= 0 { break }
        written += n
    }
}

// Running total of audio frames we have emitted. Read by the status printer.
var totalFrames: UInt64 = 0

// Objects we create and must tear down on exit.
var tapID = AudioObjectID(kAudioObjectUnknown)
var aggregateID = AudioObjectID(kAudioObjectUnknown)
var ioProcID: AudioDeviceIOProcID?

func cleanup() {
    if let proc = ioProcID, aggregateID != kAudioObjectUnknown {
        AudioDeviceStop(aggregateID, proc)
        AudioDeviceDestroyIOProcID(aggregateID, proc)
    }
    if aggregateID != kAudioObjectUnknown {
        AudioHardwareDestroyAggregateDevice(aggregateID)
    }
    if tapID != kAudioObjectUnknown {
        AudioHardwareDestroyProcessTap(tapID)
    }
}

// ----- step 1: build the tap ---------------------------------------------------

// A mono, global tap of all system output, excluding no apps. Mono means the
// tap mixes the call down to one channel for us, so we do not have to.
let tapDescription = CATapDescription(monoGlobalTapButExcludeProcesses: [])
tapDescription.name = "MeetingSystemAudioTap"
tapDescription.isPrivate = true            // do not show up as a system device
tapDescription.muteBehavior = .unmuted     // you still hear the call normally

// Create the tap object. This is the call that needs the system audio
// permission. If it is denied, status will be non zero.
var status = AudioHardwareCreateProcessTap(tapDescription, &tapID)
guard status == noErr, tapID != kAudioObjectUnknown else {
    logLine("ERROR: could not create the process tap (status \(status)).")
    logLine("If this is a permission error, grant audio recording to the app "
        + "running this helper in System Settings, Privacy and Security.")
    exit(1)
}

// Ask the tap what audio format it will hand us (sample rate, channels, type).
var asbd = AudioStreamBasicDescription()
var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
var formatAddress = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain
)
status = AudioObjectGetPropertyData(tapID, &formatAddress, 0, nil, &asbdSize, &asbd)
guard status == noErr else {
    logLine("ERROR: could not read the tap format (status \(status)).")
    cleanup()
    exit(1)
}

let sampleRate = asbd.mSampleRate
let channels = asbd.mChannelsPerFrame
let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0

// A machine readable line the Python side parses to learn the real sample rate.
// Output is always mono after the tap mixdown, so channels is 1 downstream.
logLine("META sample_rate=\(Int(sampleRate)) channels=1")

logLine("Tap format: sample rate \(Int(sampleRate)) Hz, \(channels) channel(s), "
    + "\(asbd.mBitsPerChannel) bit, float \(isFloat).")
logLine("Emitting mono 16 bit PCM to stdout at \(Int(sampleRate)) Hz.")

// ----- wrap the tap in a private aggregate device ------------------------------
// A process tap cannot be read directly. We put it inside a private aggregate
// device, then attach a callback to that device to receive the frames.

let aggregateUID = UUID().uuidString
let subTap: [String: Any] = [
    kAudioSubTapUIDKey as String: tapDescription.uuid.uuidString,
    kAudioSubTapDriftCompensationKey as String: true,
]
let aggregateDescription: [String: Any] = [
    kAudioAggregateDeviceNameKey as String: "MeetingTapAggregate",
    kAudioAggregateDeviceUIDKey as String: aggregateUID,
    kAudioAggregateDeviceIsPrivateKey as String: true,
    kAudioAggregateDeviceIsStackedKey as String: false,
    kAudioAggregateDeviceTapAutoStartKey as String: true,
    kAudioAggregateDeviceSubDeviceListKey as String: [],
    kAudioAggregateDeviceTapListKey as String: [subTap],
]

status = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregateID)
guard status == noErr, aggregateID != kAudioObjectUnknown else {
    logLine("ERROR: could not create the aggregate device (status \(status)).")
    cleanup()
    exit(1)
}

// ----- the audio callback ------------------------------------------------------
// This block runs on a dedicated audio queue every time a chunk of frames is
// ready. We convert the float samples to 16 bit integers and write them out.

let ioQueue = DispatchQueue(label: "meeting.tap.io")
let ioBlock: AudioDeviceIOBlock = { _, inInputData, _, _, _ in
    let bufferList = UnsafeMutableAudioBufferListPointer(
        UnsafeMutablePointer(mutating: inInputData)
    )
    for buffer in bufferList {
        guard let raw = buffer.mData else { continue }
        let floatCount = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
        let floats = raw.assumingMemoryBound(to: Float.self)

        // Convert each float sample in the range -1 to 1 into a 16 bit integer.
        var pcm = [Int16](repeating: 0, count: floatCount)
        for i in 0..<floatCount {
            var sample = floats[i]
            if sample > 1.0 { sample = 1.0 }
            if sample < -1.0 { sample = -1.0 }
            pcm[i] = Int16(sample * 32767.0)
        }
        pcm.withUnsafeBytes { bytes in
            if let base = bytes.baseAddress { emitBytes(base, bytes.count) }
        }
        totalFrames += UInt64(floatCount)
    }
}

status = AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggregateID, ioQueue, ioBlock)
guard status == noErr, ioProcID != nil else {
    logLine("ERROR: could not create the IO proc (status \(status)).")
    cleanup()
    exit(1)
}

status = AudioDeviceStart(aggregateID, ioProcID)
guard status == noErr else {
    logLine("ERROR: could not start the device (status \(status)).")
    cleanup()
    exit(1)
}

logLine("Capturing. Play some audio on your Mac and watch the frame count rise. "
    + "Press Ctrl+C to stop.")

// ----- status printer ----------------------------------------------------------
// Once a second, report how many frames we have captured so far. This is how you
// confirm capture is alive without looking at the binary stdout stream.

DispatchQueue.global().async {
    var lastFrames: UInt64 = 0
    while true {
        sleep(1)
        let now = totalFrames
        let seconds = Double(now) / sampleRate
        let delta = now - lastFrames
        lastFrames = now
        logLine("frames: \(now) (~\(String(format: "%.1f", seconds))s captured, "
            + "\(delta) in the last second)")
    }
}

// ----- clean shutdown on Ctrl+C ------------------------------------------------

signal(SIGINT, SIG_IGN)
let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
sigintSource.setEventHandler {
    logLine("Stopping.")
    cleanup()
    exit(0)
}
sigintSource.resume()

// Hand control to the dispatch main queue so the callback and signal handler run.
dispatchMain()
