import 'dart:async';

import 'package:flutter/services.dart';

/// A streamed event from the native generation engine.
sealed class BridgeEvent {}

/// One decoded chunk of text (already spaced for display).
class TokenEvent extends BridgeEvent {
  TokenEvent(this.text);
  final String text;
}

/// End of a generation, carrying the metrics the benchmark screen needs.
class DoneEvent extends BridgeEvent {
  DoneEvent({
    required this.ttftMs,
    required this.tps,
    required this.tokens,
    required this.totalMs,
    required this.cancelled,
  });
  final int ttftMs;
  final double tps;
  final int tokens;
  final int totalMs;
  final bool cancelled;
}

class ErrorEvent extends BridgeEvent {
  ErrorEvent(this.message);
  final String message;
}

/// Wraps the platform channels to the on-device runtime.
///
/// Control (request/response) goes over a [MethodChannel]; generated tokens
/// stream back over an [EventChannel] so the UI can render them as they arrive.
class OffhandBridge {
  static const MethodChannel _control = MethodChannel('com.offhand/control');
  static const EventChannel _tokens = EventChannel('com.offhand/tokens');

  /// Broadcast stream of generation events (tokens, then a done or error).
  Stream<BridgeEvent> events() {
    return _tokens.receiveBroadcastStream().map((dynamic e) {
      final map = Map<String, dynamic>.from(e as Map);
      switch (map['type']) {
        case 'token':
          return TokenEvent(map['text'] as String);
        case 'done':
          return DoneEvent(
            ttftMs: (map['ttftMs'] as num).toInt(),
            tps: (map['tps'] as num).toDouble(),
            tokens: (map['tokens'] as num).toInt(),
            totalMs: (map['totalMs'] as num).toInt(),
            cancelled: (map['cancelled'] as bool?) ?? false,
          );
        default:
          return ErrorEvent(map['message']?.toString() ?? 'unknown event');
      }
    });
  }

  /// Load the model bundle at [modelPath]. Returns true on success.
  Future<bool> loadModel(String modelPath) async {
    final ok = await _control.invokeMethod<bool>('loadModel', {'modelPath': modelPath});
    return ok ?? false;
  }

  /// Start generating for [prompt]. Tokens arrive on [events]; returns immediately.
  Future<void> generate(String prompt) =>
      _control.invokeMethod<void>('generate', {'prompt': prompt});

  /// Cancel the in-flight generation.
  Future<void> stop() => _control.invokeMethod<void>('stop');
}
