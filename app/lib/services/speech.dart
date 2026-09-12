import 'package:speech_to_text/speech_to_text.dart';

/// Thin wrapper over the Android SpeechRecognizer (via speech_to_text) for
/// voice-in. The first [start] triggers the runtime microphone-permission
/// prompt through the plugin's own [SpeechToText.initialize].
class SpeechInput {
  final SpeechToText _stt = SpeechToText();
  bool _ready = false;

  bool get isListening => _stt.isListening;

  Future<bool> _ensureReady({
    void Function(String status)? onStatus,
    void Function(String error)? onError,
  }) async {
    if (_ready) return true;
    _ready = await _stt.initialize(
      onStatus: (s) => onStatus?.call(s),
      onError: (e) => onError?.call(e.errorMsg),
    );
    return _ready;
  }

  /// Begin listening. [onText] fires with partial and final transcripts;
  /// [isFinal] is true on the last one. Returns false if speech is unavailable
  /// or permission was denied.
  Future<bool> start({
    required void Function(String text, bool isFinal) onText,
    void Function(String status)? onStatus,
    void Function(String error)? onError,
  }) async {
    if (!await _ensureReady(onStatus: onStatus, onError: onError)) return false;
    await _stt.listen(
      onResult: (r) => onText(r.recognizedWords, r.finalResult),
      listenOptions: SpeechListenOptions(
        partialResults: true,
        cancelOnError: true,
        listenMode: ListenMode.dictation,
        listenFor: const Duration(seconds: 30),
        pauseFor: const Duration(seconds: 3),
      ),
    );
    return true;
  }

  Future<void> stop() => _stt.stop();

  void dispose() => _stt.cancel();
}
