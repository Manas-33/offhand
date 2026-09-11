import 'dart:async';

import 'package:flutter/material.dart';

import '../agent/actions.dart';
import '../agent/parser.dart';
import '../bridge/bridge.dart';
import '../widgets/approval_sheet.dart';
import '../widgets/metrics_bar.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final OffhandBridge _bridge = OffhandBridge();
  final TextEditingController _promptController =
      TextEditingController(text: 'Set an alarm for 7am');
  final ScrollController _scrollController = ScrollController();
  StreamSubscription<BridgeEvent>? _sub;

  bool _loaded = false;
  bool _loading = false;
  bool _generating = false;
  String _output = '';
  DoneEvent? _metrics;

  @override
  void initState() {
    super.initState();
    _sub = _bridge.events().listen(
      _onEvent,
      onError: (Object error) {
        if (!mounted) return;
        setState(() {
          _generating = false;
          _output += '\n[stream error] $error';
        });
      },
    );
  }

  @override
  void dispose() {
    _sub?.cancel();
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _onEvent(BridgeEvent e) {
    setState(() {
      switch (e) {
        case TokenEvent():
          _output += e.text;
        case DoneEvent():
          _metrics = e;
          _generating = false;
        case ErrorEvent():
          _output += '\n[error] ${e.message}';
          _generating = false;
      }
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
      }
    });
    if (e is DoneEvent) _maybeApprove();
  }

  /// The agent loop: once generation finishes, parse the output for a tool call
  /// and, if there is one, require approval before executing it.
  Future<void> _maybeApprove() async {
    final calls = parseToolCalls(_output);
    if (calls.isEmpty || !mounted) return;
    final call = calls.first;

    final approved = await showApprovalSheet(context, call);
    if (!mounted) return;
    if (!approved) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('Cancelled')));
      return;
    }
    final result = await executeAction(call);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(result)));
  }

  Future<void> _loadModel() async {
    setState(() => _loading = true);
    // Placeholder path; the exported bundle is pushed to the device and pointed at here.
    final ok = await _bridge.loadModel('/data/local/tmp/offhand/qwen3-1.7b');
    setState(() {
      _loaded = ok;
      _loading = false;
    });
  }

  Future<void> _generate() async {
    setState(() {
      _output = '';
      _metrics = null;
      _generating = true;
    });
    await _bridge.generate(_promptController.text);
  }

  Future<void> _stop() async {
    await _bridge.stop();
    setState(() => _generating = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Offhand')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: FilledButton.icon(
                onPressed: (_loading || _loaded) ? null : _loadModel,
                icon: _loading
                    ? const SizedBox(
                        width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                    : Icon(_loaded ? Icons.check : Icons.download),
                label: Text(_loaded ? 'Model loaded' : 'Load model'),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _promptController,
              decoration: const InputDecoration(
                labelText: 'Prompt',
                border: OutlineInputBorder(),
              ),
              minLines: 1,
              maxLines: 3,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: FilledButton(
                    onPressed: (_loaded && !_generating) ? _generate : null,
                    child: const Text('Generate'),
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: _generating ? _stop : null,
                  child: const Text('Stop'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey.shade200,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: SingleChildScrollView(
                  controller: _scrollController,
                  child: SelectableText(
                    _output.isEmpty ? 'Output will stream here.' : _output,
                    style: const TextStyle(fontFamily: 'monospace', height: 1.4),
                  ),
                ),
              ),
            ),
            if (_metrics != null) ...[
              const SizedBox(height: 12),
              MetricsBar(metrics: _metrics!),
            ],
          ],
        ),
      ),
    );
  }
}
