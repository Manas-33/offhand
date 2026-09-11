import 'package:flutter/material.dart';

import '../bridge/bridge.dart';

/// The live TTFT / tokens-per-second / token-count row shown after a generation.
class MetricsBar extends StatelessWidget {
  const MetricsBar({super.key, required this.metrics});

  final DoneEvent metrics;

  @override
  Widget build(BuildContext context) {
    Widget cell(String value, String label) => Expanded(
          child: Column(
            children: [
              Text(value, style: Theme.of(context).textTheme.titleMedium),
              Text(label, style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
        );
    return Row(
      children: [
        cell('${metrics.ttftMs} ms', 'TTFT'),
        cell('${metrics.tps.toStringAsFixed(1)} tok/s', 'decode'),
        cell('${metrics.tokens}', 'tokens'),
      ],
    );
  }
}
