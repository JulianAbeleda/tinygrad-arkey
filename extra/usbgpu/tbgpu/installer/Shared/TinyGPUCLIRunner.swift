import Foundation
import SystemExtensions

enum TinyGPUCLIExit: Int32 { case ok = 0, usage = 2, failed = 3, needsApproval = 4 }
enum DextState { case unloaded, activating, needsApproval, rebootRequired, activated }

final class TinyGPUCLIRunner: NSObject, OSSystemExtensionRequestDelegate {
  private let dextID: String
  private var done: ((TinyGPUCLIExit) -> Void)?
  private var isInstall = true

  init(_ dextID: String) { self.dextID = dextID }

  static func bundledDextVersion(_ bundleID: String) -> String? {
    let url = Bundle.main.bundleURL.appendingPathComponent("Contents/Library/SystemExtensions/\(bundleID).dext")
    return Bundle(url: url)?.object(forInfoDictionaryKey: "CFBundleVersion") as? String
  }

  static func classifyDextState(_ output: String, bundleID: String, expectedVersion: String) -> DextState {
    let legacyBundleID = "org.tinygrad.tinygpu.driver2"
    let allLines = output.split(separator: "\n")
    let lines = allLines.filter { $0.contains("\(bundleID) (") }
    let legacyActive = allLines.contains { $0.contains("\(legacyBundleID) (") && $0.contains("[activated enabled]") }
    guard !lines.isEmpty else { return legacyActive ? .rebootRequired : .unloaded }
    let currentActive = lines.filter { $0.contains("/\(expectedVersion))") && $0.contains("[activated enabled]") }
    let transitional = lines.contains { $0.contains("terminating") || $0.contains("waiting to uninstall") || $0.contains("being replaced") }
    if currentActive.count == 1 && lines.count == 1 && !transitional && !legacyActive { return .activated }
    if lines.count > 1 || transitional || legacyActive { return .rebootRequired }
    if lines.contains(where: { $0.contains("[activated waiting for user]") }) { return .needsApproval }
    return .activating
  }

  static func queryDextState(_ bundleID: String) -> DextState {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/systemextensionsctl")
    process.arguments = ["list"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()
    guard (try? process.run()) != nil else { return .unloaded }
    process.waitUntilExit()

    guard let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8),
          let expectedVersion = bundledDextVersion(bundleID) else { return .unloaded }
    return classifyDextState(output, bundleID: bundleID, expectedVersion: expectedVersion)
  }

  private static let approvalHelp = """
    Please go to System Settings > Privacy & Security and allow the extension.

    If previously disabled: System Settings > General > Login Items & Extensions > Driver Extensions > Toggle TinyGPU ON

    """

  static func statusText(_ state: DextState) -> String {
    switch state {
    case .unloaded: return "Driver extension not installed.\n\n"
    case .activating: return "Extension is activating...\n\n"
    case .needsApproval: return "Extension awaiting approval.\n\n" + approvalHelp
    case .rebootRequired: return "Extension upgrade is pending. Restart macOS before using TinyGPU.\n\n"
    case .activated: return "Extension registration is clean and active. Verify keepalive and power status before using the GPU.\n\n"
    }
  }

  private static func keepaliveJSON(status: Bool) -> String? {
    var bytes = [CChar](repeating: 0, count: 4096)
    var length: UInt = 0
    let result = status
      ? tinygpu_keepalive_status(&bytes, UInt(bytes.count), &length)
      : tinygpu_keepalive_handshake(&bytes, UInt(bytes.count), &length)
    let schema = status ? "tinygpu.keepalive.v1" : "tinygpu.handshake.v1"
    guard result == 0, length <= bytes.count,
          let value = String(bytes: bytes.prefix(Int(length)).map({ UInt8(bitPattern: $0) }), encoding: .utf8),
          let data = value.data(using: .utf8),
          let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any], dictionary["schema"] as? String == schema
    else { return nil }
    return value
  }

  private static func powerResidencyJSON() -> String? {
    var bytes = [CChar](repeating: 0, count: 4096)
    var length: UInt = 0
    guard tinygpu_power_residency_status(&bytes, UInt(bytes.count), &length) == 0, length <= bytes.count,
          let value = String(bytes: bytes.prefix(Int(length)).map({ UInt8(bitPattern: $0) }), encoding: .utf8),
          let data = value.data(using: .utf8),
          let object = try? JSONSerialization.jsonObject(with: data),
          let dictionary = object as? [String: Any], dictionary["schema"] as? String == "tinygpu.power-residency.v2"
    else { return nil }
    return value
  }

  func run(args: [String], done: @escaping (TinyGPUCLIExit) -> Void) {
    self.done = done
    guard args.count > 1 else { return usage() }

    switch args[1] {
    case "keepalive":
      guard args.count == 3, args[2] == "status" || args[2] == "handshake" else { return usage() }
      guard let value = Self.keepaliveJSON(status: args[2] == "status") else {
        print("keepalive \(args[2]) unavailable")
        return done(.failed)
      }
      print(value)
      done(.ok)
    case "power":
      guard args.count == 3, args[2] == "status" else { return usage() }
      guard let value = Self.powerResidencyJSON() else {
        print("power status unavailable")
        return done(.failed)
      }
      print(value)
      done(.ok)
    case "status":
      print(Self.statusText(Self.queryDextState(dextID)))
      done(.ok)
    case "install":
      if Self.queryDextState(dextID) == .needsApproval {
        print(Self.statusText(.needsApproval))
        return done(.needsApproval)
      }
      print("Installing TinyGPU driver extension...\nYou may need to approve in System Settings.\n")
      submitRequest(activate: true)
    case "uninstall":
      guard Self.queryDextState(dextID) != .unloaded else {
        print("Not installed.\n")
        return done(.ok)
      }
      print("Uninstalling TinyGPU driver extension...\n")
      isInstall = false
      submitRequest(activate: false)
    case "server":
      guard args.count > 2 else {
        print("Error: server requires socket path\n")
        return usage()
      }
      done(run_server(args[2]) == 0 ? .ok : .failed)
    case "help", "-h", "--help":
      printUsage()
      done(.ok)
    default:
      print("Unknown command: \(args[1])\n")
      usage()
    }
  }

  private func printUsage() {
    print("""
      Usage: TinyGPU <command>
        status          Show extension status
        keepalive status  Show DriverKit keeper status as JSON
        keepalive handshake  Show diagnostic protocol capabilities as JSON
        power status     Show DriverKit power-residency status as JSON
        install         Install the driver extension
        uninstall       Remove the driver extension
        server <path>   Start server on Unix socket
      """)
  }

  private func usage() {
    printUsage()
    done?(.usage)
  }

  private func submitRequest(activate: Bool) {
    let request = activate
      ? OSSystemExtensionRequest.activationRequest(forExtensionWithIdentifier: dextID, queue: .main)
      : OSSystemExtensionRequest.deactivationRequest(forExtensionWithIdentifier: dextID, queue: .main)
    request.delegate = self
    OSSystemExtensionManager.shared.submitRequest(request)
  }

  func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
    print("\nUser approval required!\n\n\(Self.approvalHelp)After approval, connect the gpu and use it with tinygrad-arkey.\n")
    done?(.needsApproval)
  }

  func request(_ request: OSSystemExtensionRequest, didFinishWithResult result: OSSystemExtensionRequest.Result) {
    switch result {
    case .completed: print("Driver extension registration request completed; verify the complete registration set before use.\n")
    case .willCompleteAfterReboot: print("Will complete after reboot.\n")
    @unknown default: print("Completed: \(result)\n")
    }
    done?(.ok)
  }

  func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
    print("\nError: \(error.localizedDescription)\n")
    let code = (error as NSError).code
    if code == 2 { print("Missing entitlements. Rebuild with proper signing.\n") }
    else if code == 4 { print("Extension not found in the containing app bundle.\n") }
    else if code == 8 { print("Extension code signature is invalid.\n") }
    else if code == 9 { print("Extension validation failed.\n") }
    else if code == 10 { print("Extension is blocked by system policy.\n") }
    else if code == 13 { print("Authorization is required to change the extension registration.\n") }
    done?(.failed)
  }

  func request(_ request: OSSystemExtensionRequest, actionForReplacingExtension existing: OSSystemExtensionProperties,
               withExtension ext: OSSystemExtensionProperties) -> OSSystemExtensionRequest.ReplacementAction {
    print("Updating v\(existing.bundleShortVersion) -> v\(ext.bundleShortVersion)...\n")
    return .replace
  }
}
