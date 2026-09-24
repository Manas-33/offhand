import 'dart:async';

import 'package:flutter/material.dart';

import '../agent/actions.dart';
import '../agent/parser.dart';
import '../bridge/bridge.dart';
import '../services/speech.dart';
import '../widgets/approval_sheet.dart';
import '../widgets/metrics_bar.dart';
import '../widgets/home_design.dart';
import 'benchmark_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final OffhandBridge _bridge = OffhandBridge();
  final SpeechInput _speech = SpeechInput();
  final TextEditingController _promptController = TextEditingController();
  StreamSubscription<BridgeEvent>? _sub;

  bool _loaded = false;
  bool _loading = false;
  bool _generating = false;
  bool _stopping = false;
  bool _reviewing = false;
  bool _listening = false;
  bool _startingMic = false;
  String _output = '';
  String? _notice;
  String? _error;
  ToolCall? _action;
  DoneEvent? _metrics;

  bool get _busy => _generating || _reviewing || _loading;

  @override
  void initState() {
    super.initState();
    _listen();
  }

  void _listen() {
    _sub = _bridge.events().listen(_onEvent, onError: _onStreamError);
  }

  void _onStreamError(Object error) {
    if (!mounted) return;
    setState(() {
      _generating = false;
      _stopping = false;
      _error = 'The connection was interrupted. Please try your request again.';
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    _speech.dispose();
    _promptController.dispose();
    super.dispose();
  }

  void _onEvent(BridgeEvent event) {
    if (!mounted || !_generating) return;
    var shouldReview = false;
    setState(() {
      switch (event) {
        case TokenEvent():
          _output += event.text;
        case DoneEvent():
          _metrics = event;
          _generating = false;
          if (event.cancelled || _stopping) {
            _notice = 'Request stopped. No action was taken.';
          } else {
            final calls = parseToolCalls(_output);
            _action = calls.isEmpty ? null : calls.first;
            shouldReview = _action != null;
            final response = _output
                .replaceAll(RegExp(r'<think>.*?</think>', dotAll: true), '')
                .trim();
            _notice = shouldReview
                ? 'Ready for your review.'
                : response.isEmpty
                ? 'No action found. Try a specific request, like setting a timer.'
                : response;
          }
          _stopping = false;
        case ErrorEvent():
          _error = 'Couldn’t complete your request. ${event.message}';
          _generating = false;
          _stopping = false;
      }
    });
    if (shouldReview) unawaited(_reviewAction());
  }

  Future<void> _reviewAction() async {
    final call = _action;
    if (call == null || _reviewing) return;
    setState(() => _reviewing = true);
    try {
      final approved = await showApprovalSheet(context, call);
      if (!mounted) return;
      if (!approved) {
        setState(() => _notice = 'Not approved. No action was taken.');
        return;
      }
      setState(() => _notice = 'Opening your action…');
      final result = await executeAction(call);
      if (!mounted) return;
      setState(() {
        _notice = result;
        _action = null;
      });
    } catch (_) {
      if (mounted) {
        setState(() => _error = 'Couldn’t open this action. Please try again.');
      }
    } finally {
      if (mounted) setState(() => _reviewing = false);
    }
  }

  Future<void> _loadModel() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final ok = await _bridge.loadModel('models/Qwen3-0.6B-Specialist');
      if (!mounted) return;
      setState(() {
        _loaded = ok;
        if (!ok) {
          _error =
              'Setup couldn’t finish. Check that the on-device model is installed, then retry.';
        }
      });
    } catch (_) {
      if (mounted) {
        setState(
          () => _error =
              'Couldn’t start the assistant. Check that the on-device model is installed, then retry.',
        );
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _generate() async {
    final prompt = _promptController.text.trim();
    if (!_loaded || _busy || _listening || _startingMic || prompt.isEmpty) {
      return;
    }
    FocusManager.instance.primaryFocus?.unfocus();
    setState(() {
      _output = '';
      _notice = null;
      _error = null;
      _action = null;
      _metrics = null;
      _stopping = false;
      _generating = true;
    });
    try {
      await _bridge.generate(prompt);
    } catch (_) {
      if (mounted) {
        setState(() {
          _generating = false;
          _error = 'Couldn’t start your request. Please try again.';
        });
      }
    }
  }

  Future<void> _stop() async {
    setState(() => _stopping = true);
    try {
      await _bridge.stop();
      // Wait for the terminal event before allowing another request: the
      // native event stream does not attach request IDs to late tokens.
    } catch (_) {
      if (mounted) {
        setState(() {
          _stopping = false;
          _error = 'Couldn’t stop the request. Please try Stop again.';
        });
      }
    }
  }

  Future<void> _toggleMic() async {
    if (_listening) {
      try {
        await _speech.stop();
        if (mounted) setState(() => _listening = false);
      } catch (_) {
        if (mounted) {
          setState(
            () => _error =
                'Couldn’t stop listening. Tap the microphone to try again.',
          );
        }
      }
      return;
    }
    setState(() {
      _startingMic = true;
      _error = null;
    });
    try {
      final started = await _speech.start(
        onText: (text, isFinal) {
          if (!mounted) return;
          setState(() {
            _promptController.value = TextEditingValue(
              text: text,
              selection: TextSelection.collapsed(offset: text.length),
            );
            if (isFinal) _listening = false;
          });
        },
        onStatus: (status) {
          if (mounted && (status == 'done' || status == 'notListening')) {
            setState(() => _listening = false);
          }
        },
        onError: (_) {
          if (!mounted) return;
          setState(() {
            _listening = false;
            _error =
                'Couldn’t hear you. Try the microphone again or type your request.';
          });
        },
      );
      if (!mounted) return;
      setState(() {
        _listening = started && _speech.isListening;
        if (!started) {
          _error =
              'Voice input is unavailable. Allow microphone access in Settings, or type your request.';
        }
      });
    } catch (_) {
      if (mounted) {
        setState(
          () => _error =
              'Voice input is unavailable. You can still type your request.',
        );
      }
    } finally {
      if (mounted) setState(() => _startingMic = false);
    }
  }

  Future<void> _openBenchmark() async {
    final navigator = Navigator.of(context);
    await _sub?.cancel();
    _sub = null;
    if (!mounted) return;
    await navigator.push(
      MaterialPageRoute(builder: (_) => const BenchmarkScreen()),
    );
    if (mounted) _listen();
  }

  void _useExample(String prompt) {
    setState(() {
      _promptController.value = TextEditingValue(
        text: prompt,
        selection: TextSelection.collapsed(offset: prompt.length),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;
    final inputBlocked = _busy || _listening || _startingMic;
    final canAsk =
        _loaded && !inputBlocked && _promptController.text.trim().isNotEmpty;
    return Scaffold(
      body: SafeArea(
        child: Align(
          alignment: Alignment.topCenter,
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 620),
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(22, 10, 22, 28),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Row(
                    children: [
                      const OffhandMark(size: 27),
                      const SizedBox(width: 9),
                      Expanded(
                        child: Text(
                          'offhand',
                          style: theme.textTheme.titleLarge?.copyWith(
                            fontSize: 26,
                            fontWeight: FontWeight.w800,
                            letterSpacing: -1.3,
                          ),
                        ),
                      ),
                      IconButton(
                        tooltip: 'Performance benchmark',
                        onPressed: inputBlocked ? null : _openBenchmark,
                        style: IconButton.styleFrom(
                          side: BorderSide(color: colors.outlineVariant),
                        ),
                        icon: const Icon(Icons.tune_rounded, size: 19),
                      ),
                    ],
                  ),
                  const SizedBox(height: 28),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Expanded(
                        child: Text.rich(
                          TextSpan(
                            children: [
                              const TextSpan(text: 'Life, a little\n'),
                              TextSpan(
                                text: 'lighter.',
                                style: TextStyle(color: colors.primary),
                              ),
                            ],
                          ),
                          style: TextStyle(
                            fontFamily: 'InstrumentSerif',
                            fontSize: 58,
                            height: .96,
                            letterSpacing: -1.3,
                            color: colors.onSurface,
                          ),
                        ),
                      ),
                      if (MediaQuery.textScalerOf(context).scale(14) < 22)
                        Padding(
                          padding: const EdgeInsets.only(top: 32, right: 10),
                          child: Transform.rotate(
                            angle: -.2,
                            child: OffhandMark(
                              size: 52,
                              color: colors.primary.withValues(alpha: .7),
                            ),
                          ),
                        ),
                    ],
                  ),
                  const SizedBox(height: 18),
                  Semantics(
                    liveRegion: true,
                    child: Wrap(
                      spacing: 7,
                      runSpacing: 4,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        Container(
                          width: 6,
                          height: 6,
                          decoration: BoxDecoration(
                            color: _loaded
                                ? const Color(0xFF548261)
                                : colors.outline,
                            shape: BoxShape.circle,
                          ),
                        ),
                        Text(
                          _loaded
                              ? (_generating
                                    ? 'On-device assistant · Working'
                                    : 'On-device assistant · Ready')
                              : 'Small tasks. A little more headspace.',
                          style: TextStyle(
                            fontSize: 11,
                            color: colors.onSurfaceVariant,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 22),
                  VoiceCard(
                    loaded: _loaded,
                    loading: _loading,
                    listening: _listening,
                    onStart: inputBlocked ? null : _loadModel,
                    onSpeak: _busy || _startingMic ? null : _toggleMic,
                  ),
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.fromLTRB(16, 7, 8, 7),
                    decoration: BoxDecoration(
                      color: colors.surfaceContainerLowest,
                      borderRadius: BorderRadius.circular(22),
                      border: Border.all(
                        color: colors.outlineVariant.withValues(alpha: .6),
                      ),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Expanded(
                          child: TextField(
                            controller: _promptController,
                            readOnly: inputBlocked,
                            onChanged: (_) => setState(() {}),
                            style: const TextStyle(fontSize: 14, height: 1.5),
                            decoration: const InputDecoration(
                              hintText: 'Or type a little task…',
                              semanticCounterText: 'Your request',
                              filled: false,
                              border: InputBorder.none,
                              enabledBorder: InputBorder.none,
                              focusedBorder: InputBorder.none,
                              contentPadding: EdgeInsets.symmetric(
                                vertical: 12,
                              ),
                            ),
                            minLines: 1,
                            maxLines: 4,
                          ),
                        ),
                        const SizedBox(width: 8),
                        if (_generating)
                          IconButton.filledTonal(
                            tooltip: _stopping ? 'Stopping…' : 'Stop request',
                            onPressed: _stopping ? null : _stop,
                            style: IconButton.styleFrom(
                              minimumSize: const Size(48, 48),
                            ),
                            icon: const Icon(Icons.stop_rounded),
                          )
                        else
                          IconButton.filled(
                            tooltip: _reviewing
                                ? 'Reviewing action…'
                                : 'Ask Offhand',
                            onPressed: canAsk ? _generate : null,
                            style: IconButton.styleFrom(
                              minimumSize: const Size(48, 48),
                            ),
                            icon: const Icon(
                              Icons.arrow_upward_rounded,
                              size: 22,
                            ),
                          ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    _listening
                        ? 'Listening… Tap the orb to finish.'
                        : 'A little help. Always with your say-so.',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 10,
                      color: colors.onSurfaceVariant,
                    ),
                  ),
                  if (_error != null) ...[
                    const SizedBox(height: 20),
                    _MessageCard(
                      title: 'Let’s try that again',
                      message: _error!,
                      icon: Icons.error_outline,
                      isError: true,
                    ),
                  ],
                  if (_generating || _notice != null) ...[
                    const SizedBox(height: 24),
                    _MessageCard(
                      title: _generating
                          ? (_stopping
                                ? 'Stopping your request'
                                : 'Working on it…')
                          : 'Your request',
                      message: _generating
                          ? 'Preparing a response on your phone.'
                          : _notice!,
                      icon: _generating
                          ? Icons.more_horiz
                          : Icons.chat_bubble_outline,
                      busy: _generating,
                      action: _action != null && !_reviewing
                          ? TextButton.icon(
                              onPressed: _reviewAction,
                              icon: const Icon(Icons.fact_check_outlined),
                              label: const Text('Review action'),
                            )
                          : null,
                    ),
                  ] else if (_error == null) ...[
                    const SizedBox(height: 28),
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            'A place to start',
                            style: TextStyle(
                              fontFamily: 'InstrumentSerif',
                              fontSize: 27,
                              color: colors.onSurface,
                            ),
                          ),
                        ),
                        if (MediaQuery.textScalerOf(context).scale(14) < 22)
                          Text(
                            'THE EVERYDAY STUFF',
                            style: TextStyle(
                              fontSize: 8,
                              letterSpacing: 1.15,
                              fontWeight: FontWeight.w700,
                              color: colors.onSurfaceVariant,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 14),
                    LayoutBuilder(
                      builder: (context, constraints) {
                        final singleColumn =
                            MediaQuery.textScalerOf(context).scale(14) > 22;
                        final width = singleColumn
                            ? constraints.maxWidth
                            : (constraints.maxWidth - 12) / 2;
                        return Wrap(
                          spacing: 12,
                          runSpacing: 12,
                          children: [
                            SizedBox(
                              width: width,
                              child: SuggestionCard(
                                label: '10-minute timer',
                                detail: 'Make time for a pause',
                                icon: Icons.timelapse_rounded,
                                tint: const Color(0xFFEAE5F0),
                                onTap: inputBlocked
                                    ? null
                                    : () => _useExample(
                                        'Start a 10 minute timer',
                                      ),
                              ),
                            ),
                            SizedBox(
                              width: width,
                              child: SuggestionCard(
                                label: 'Make a note',
                                detail: 'Keep a passing thought',
                                icon: Icons.edit_note_rounded,
                                tint: const Color(0xFFF3E3D2),
                                onTap: inputBlocked
                                    ? null
                                    : () => _useExample(
                                        'Make a note to buy milk',
                                      ),
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 8,
                      runSpacing: 4,
                      children: [
                        TextButton.icon(
                          onPressed: inputBlocked
                              ? null
                              : () => _useExample('Set an alarm for 7am'),
                          icon: const Icon(Icons.alarm_rounded, size: 16),
                          label: const Text(
                            'Morning alarm',
                            style: TextStyle(fontSize: 11),
                          ),
                        ),
                        TextButton.icon(
                          onPressed: inputBlocked
                              ? null
                              : () => _useExample('Turn on the flashlight'),
                          icon: const Icon(
                            Icons.flashlight_on_outlined,
                            size: 16,
                          ),
                          label: const Text(
                            'Flashlight',
                            style: TextStyle(fontSize: 11),
                          ),
                        ),
                      ],
                    ),
                  ],
                  if (_output.isNotEmpty || _metrics != null) ...[
                    const SizedBox(height: 20),
                    ExpansionTile(
                      tilePadding: EdgeInsets.zero,
                      title: const Text(
                        'Technical details',
                        style: TextStyle(fontSize: 13),
                      ),
                      children: [
                        if (_metrics != null) MetricsBar(metrics: _metrics!),
                        const SizedBox(height: 16),
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: colors.surfaceContainer,
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: SelectableText(
                            _output,
                            style: theme.textTheme.bodySmall?.copyWith(
                              fontFamily: 'monospace',
                              height: 1.5,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _SmallProgress extends StatelessWidget {
  const _SmallProgress();

  @override
  Widget build(BuildContext context) => const SizedBox(
    width: 18,
    height: 18,
    child: CircularProgressIndicator(strokeWidth: 2),
  );
}

class _MessageCard extends StatelessWidget {
  const _MessageCard({
    required this.title,
    required this.message,
    required this.icon,
    this.isError = false,
    this.busy = false,
    this.action,
  });

  final String title;
  final String message;
  final IconData icon;
  final bool isError;
  final bool busy;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final foreground = isError ? colors.onErrorContainer : colors.onSurface;
    return Semantics(
      liveRegion: true,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: isError ? colors.errorContainer : colors.surfaceContainer,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                if (busy)
                  const _SmallProgress()
                else
                  Icon(icon, size: 22, color: foreground),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    title,
                    style: Theme.of(
                      context,
                    ).textTheme.titleMedium?.copyWith(color: foreground),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(message, style: TextStyle(color: foreground, height: 1.5)),
            if (action != null) ...[const SizedBox(height: 8), action!],
          ],
        ),
      ),
    );
  }
}
