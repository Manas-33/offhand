import 'package:flutter/material.dart';

import '../agent/actions.dart';
import '../agent/parser.dart';

/// Ask the user to approve a tool call before it runs. Resolves true if approved.
///
/// Nothing user-visible should fire without this returning true.
Future<bool> showApprovalSheet(BuildContext context, ToolCall call) async {
  final approved = await showModalBottomSheet<bool>(
    context: context,
    showDragHandle: true,
    builder: (context) => SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Approve action?', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.bolt),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(describeAction(call),
                      style: Theme.of(context).textTheme.bodyLarge),
                ),
              ],
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => Navigator.pop(context, false),
                    child: const Text('Cancel'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton(
                    onPressed: () => Navigator.pop(context, true),
                    child: const Text('Approve'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    ),
  );
  return approved ?? false;
}
