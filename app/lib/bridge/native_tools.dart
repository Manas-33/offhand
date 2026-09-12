import 'package:flutter/services.dart';

/// Wraps the `com.offhand/tools` method channel: the small set of actions that
/// no Android intent can perform. The flashlight and media transport keys are
/// handled natively in `MainActivity`; a benchmark also reads process memory
/// through here.
class NativeTools {
  static const MethodChannel _channel = MethodChannel('com.offhand/tools');

  /// Turn the device flashlight on or off. Returns false if the device has no
  /// controllable torch.
  Future<bool> setTorch(bool on) async {
    final ok = await _channel.invokeMethod<bool>('setTorch', {'on': on});
    return ok ?? false;
  }

  /// Send a media transport key (`play` | `pause` | `next` | `previous`) to
  /// whatever holds the active media session.
  Future<void> mediaKey(String action) =>
      _channel.invokeMethod<void>('mediaKey', {'action': action});

  /// Current process memory in MB (PSS), for the benchmark screen. Returns null
  /// if unavailable.
  Future<double?> memoryMb() async {
    final v = await _channel.invokeMethod<num>('memoryMb');
    return v?.toDouble();
  }
}
