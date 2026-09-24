import 'package:flutter/material.dart';

import '../bridge/bridge.dart';

/// Performance details shown after a generation.
class MetricsBar extends StatelessWidget {
  const MetricsBar({super.key, required this.metrics});

  final DoneEvent metrics;

  @override
  Widget build(BuildContext context) {
    Widget cell(String value, String label) => Padding(
      padding: const EdgeInsets.only(right: 24, bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(value, style: Theme.of(context).textTheme.titleMedium),
          Text(label, style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        cell('${metrics.ttftMs} ms', 'First token'),
        cell('${metrics.tps.toStringAsFixed(1)} tok/s', 'Decode speed'),
        cell('${metrics.tokens}', 'Tokens'),
      ],
    );
  }
}
