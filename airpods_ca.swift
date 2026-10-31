// Toggle AirPods Conversation Awareness over Apple's AACP protocol (L2CAP PSM 0x1001).
// Usage: airpods_ca on|off     Exit 0 on success, 2 no AirPods, 3 channel failed, 4 write failed.
// Packet layout follows the LibrePods reverse-engineering; unofficial and may break with firmware updates.
import Foundation
import IOBluetooth

let handshake: [UInt8] = [0x00, 0x00, 0x04, 0x00, 0x01, 0x00, 0x02, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
func caPacket(_ on: Bool) -> [UInt8] {
    [0x04, 0x00, 0x04, 0x00, 0x09, 0x00, 0x28, on ? 0x01 : 0x02, 0x00, 0x00, 0x00]
}

guard CommandLine.arguments.count == 2, ["on", "off"].contains(CommandLine.arguments[1]) else {
    FileHandle.standardError.write("usage: airpods_ca on|off\n".data(using: .utf8)!)
    exit(64)
}
let enable = CommandLine.arguments[1] == "on"

let paired = (IOBluetoothDevice.pairedDevices() as? [IOBluetoothDevice]) ?? []
guard let dev = paired.first(where: { $0.isConnected() && ($0.name ?? "").contains("AirPods") }) else {
    FileHandle.standardError.write("no connected AirPods\n".data(using: .utf8)!)
    exit(2)
}

var channel: IOBluetoothL2CAPChannel?
let rc = dev.openL2CAPChannelSync(&channel, withPSM: 0x1001, delegate: nil)
guard rc == kIOReturnSuccess, let ch = channel else {
    FileHandle.standardError.write("L2CAP open failed: \(rc)\n".data(using: .utf8)!)
    exit(3)
}

func send(_ bytes: [UInt8]) -> Bool {
    var b = bytes
    return ch.writeSync(&b, length: UInt16(b.count)) == kIOReturnSuccess
}

let ok = send(handshake) && {
    RunLoop.current.run(until: Date().addingTimeInterval(0.3))
    return send(caPacket(enable))
}()
RunLoop.current.run(until: Date().addingTimeInterval(0.2))
ch.close()
exit(ok ? 0 : 4)
