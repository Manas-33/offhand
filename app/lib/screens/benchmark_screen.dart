import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../agent/parser.dart';
import '../bridge/bridge.dart';
import '../bridge/native_tools.dart';

/// A representative request per tool, plus a no-call case, used to characterize
/// the on-device engine. The mock maps each to a plausible tool call so the
/// numbers are meaningful before the specialist bundle lands.
const List<String> _benchPrompts = [
  'Set an alarm for 7am',
  'Start a 10 minute timer',
  'Remind me to call the dentist at 3pm',
  'Add a team sync to my calendar tomorrow at 10am',
  'Open wifi settings',
  'Turn on bluetooth',
  "Text Mom I'm on my way",
  'Email Alex about lunch',
  'Play some lo-fi beats',
  'Pause the music',
  'Turn on the flashlight',
  'Make a note to buy milk',
  "What's the weather tomorrow?",
];

/// One prompt's measured result.
class BenchRow {
  BenchRow({
    required this.prompt,
    this.ttftMs = 0,
    this.tps = 0,
    this.tokens = 0,
    this.totalMs = 0,
    this.toolCallMs,
    this.tool,
    this.error,
  });

  final String prompt;
  final int ttftMs;
  final double tps;
  final int tokens;
  final int totalMs;

  /// Time from generate() to the first parseable tool call, or null if the
  /// model produced no call (the no-call / irrelevance path).
  final int? toolCallMs;
  final String? tool;
  final String? error;

  Map<String, Object?> toJson() => {
        'prompt': prompt,
        'ttft_ms': ttftMs,
        'tps': double.parse(tps.toStringAsFixed(2)),
        'tokens': tokens,
        'total_ms': totalMs,
        'tool_call_ms': toolCallMs,
        'tool': tool,
        if (error != null) 'error': error,
      };
}

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key});

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  final OffhandBridge _bridge = OffhandBridge();
  final NativeTools _native = NativeTools();

  bool _running = false;
  String _status = 'Idle';
  final List<BenchRow> _rows = [];
  double? _memoryMb;

  Future<void> _run() async {
    setState(() {
      _running = true;
      _rows.clear();
      _status = 'Loading model…';
      _memoryMb = null;
    });

    await _bridge.loadModel('models/Qwen3-0.6B-Specialist');

    for (var i = 0; i < _benchPrompts.length; i++) {
      if (!mounted) return;
      setState(() => _status = 'Running ${i + 1}/${_benchPrompts.length}…');
      final row = await _runOnce(_benchPrompts[i]);
      if (!mounted) return;
      setState(() => _rows.add(row));
    }

    final mem = await _native.memoryMb();
    if (!mounted) return;
    setState(() {
      _memoryMb = mem;
      _running = false;
      _status = 'Done — ${_rows.length} prompts';
    });
  }

  /// Generate for one prompt and measure it. Tool-call latency is the moment the
  /// streamed text first parses to a tool call, computed live so it reflects a
  /// real streaming engine, not just the total time.
  Future<BenchRow> _runOnce(String prompt) async {
    final completer = Completer<BenchRow>();
    final stopwatch = Stopwatch()..start();
    final buffer = StringBuffer();
    int? toolCallMs;
    String? tool;

    void checkForCall() {
      if (toolCallMs != null) return;
      final calls = parseToolCalls(buffer.toString());
      if (calls.isNotEmpty) {
        toolCallMs = stopwatch.elapsedMilliseconds;
        tool = calls.first.name;
      }
    }

    late final StreamSubscription<BridgeEvent> sub;
    sub = _bridge.events().listen((e) {
      switch (e) {
        case TokenEvent():
          buffer.write(e.text);
          checkForCall();
        case DoneEvent():
          checkForCall();
          sub.cancel();
          completer.complete(BenchRow(
            prompt: prompt,
            ttftMs: e.ttftMs,
            tps: e.tps,
            tokens: e.tokens,
            totalMs: e.totalMs,
            toolCallMs: toolCallMs,
            tool: tool,
          ));
        case ErrorEvent():
          sub.cancel();
          completer.complete(BenchRow(prompt: prompt, error: e.message));
      }
    });

    await _bridge.generate(prompt);
    return completer.future;
  }

  void _copyJson() {
    final calls = _rows.where((r) => r.tool != null).toList();
    final report = {
      'engine': 'mock',
      'generated_at': DateTime.now().toIso8601String(),
      'n': _rows.length,
      'aggregates': {
        'median_ttft_ms': _median(_rows.map((r) => r.ttftMs.toDouble())),
        'median_tps': _median(_rows.map((r) => r.tps)),
        'median_tool_call_ms':
            _median(calls.map((r) => r.toolCallMs!.toDouble())),
        'tool_calls': calls.length,
        if (_memoryMb != null)
          'process_mem_mb': double.parse(_memoryMb!.toStringAsFixed(1)),
      },
      'rows': _rows.map((r) => r.toJson()).toList(),
    };
    Clipboard.setData(ClipboardData(text: const JsonEncoder.withIndent('  ').convert(report)));
    ScaffoldMessenger.of(context)
        .showSnackBar(const SnackBar(content: Text('Benchmark JSON copied to clipboard')));
  }

  double? _median(Iterable<double> values) {
    final list = values.toList()..sort();
    if (list.isEmpty) return null;
    final mid = list.length ~/ 2;
    final m = list.length.isOdd ? list[mid] : (list[mid - 1] + list[mid]) / 2;
    return double.parse(m.toStringAsFixed(1));
  }

  @override
  Widget build(BuildContext context) {
    final withCalls = _rows.where((r) => r.tool != null).toList();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Benchmark'),
        actions: [
          IconButton(
            tooltip: 'Copy JSON',
            onPressed: _rows.isEmpty || _running ? null : _copyJson,
            icon: const Icon(Icons.copy_all),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            FilledButton.icon(
              onPressed: _running ? null : _run,
              icon: _running
                  ? const SizedBox(
                      width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.play_arrow),
              label: Text(_running ? 'Running…' : 'Run benchmark'),
            ),
            const SizedBox(height: 8),
            Text(_status, style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 12),
            _SummaryCard(
              medianTtft: _median(_rows.map((r) => r.ttftMs.toDouble())),
              medianTps: _median(_rows.map((r) => r.tps)),
              medianToolCall: _median(withCalls.map((r) => r.toolCallMs!.toDouble())),
              memoryMb: _memoryMb,
            ),
            const SizedBox(height: 12),
            Expanded(
              child: _rows.isEmpty
                  ? const Center(child: Text('Run the suite to see per-prompt metrics.'))
                  : ListView.separated(
                      itemCount: _rows.length,
                      separatorBuilder: (_, _) => const Divider(height: 1),
                      itemBuilder: (_, i) => _RowTile(row: _rows[i]),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SummaryCard extends StatelessWidget {
  const _SummaryCard({
    required this.medianTtft,
    required this.medianTps,
    required this.medianToolCall,
    required this.memoryMb,
  });

  final double? medianTtft;
  final double? medianTps;
  final double? medianToolCall;
  final double? memoryMb;

  @override
  Widget build(BuildContext context) {
    String v(double? x, String unit) => x == null ? '—' : '${x.toStringAsFixed(x >= 100 ? 0 : 1)} $unit';
    Widget cell(String value, String label) => Expanded(
          child: Column(children: [
            Text(value, style: Theme.of(context).textTheme.titleMedium),
            Text(label, style: Theme.of(context).textTheme.bodySmall, textAlign: TextAlign.center),
          ]),
        );
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            cell(v(medianTtft, 'ms'), 'median TTFT'),
            cell(v(medianTps, 'tok/s'), 'median decode'),
            cell(v(medianToolCall, 'ms'), 'median\ntool-call'),
            cell(v(memoryMb, 'MB'), 'process mem'),
          ],
        ),
      ),
    );
  }
}

class _RowTile extends StatelessWidget {
  const _RowTile({required this.row});
  final BenchRow row;

  @override
  Widget build(BuildContext context) {
    final subtitle = row.error != null
        ? 'error: ${row.error}'
        : '${row.ttftMs} ms TTFT · ${row.tps.toStringAsFixed(1)} tok/s · '
            '${row.toolCallMs == null ? 'no call' : '${row.toolCallMs} ms → ${row.tool}'}';
    return ListTile(
      dense: true,
      title: Text(row.prompt, maxLines: 1, overflow: TextOverflow.ellipsis),
      subtitle: Text(subtitle),
      trailing: Icon(
        row.tool != null ? Icons.check_circle : Icons.chat_bubble_outline,
        size: 18,
        color: row.tool != null ? Colors.green : Colors.grey,
      ),
    );
  }
}
