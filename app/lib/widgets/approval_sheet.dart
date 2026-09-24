import 'package:flutter/material.dart';

import '../agent/actions.dart';
import '../agent/parser.dart';

/// Ask the user to approve a tool call before it runs. Resolves true if approved.
Future<bool> showApprovalSheet(BuildContext context, ToolCall call) async {
  final detail = actionDetail(call);
  final isDraft = call.name == 'draft_sms' || call.name == 'draft_email';
  final approved = await showModalBottomSheet<bool>(
    context: context,
    showDragHandle: true,
    isScrollControlled: true,
    useSafeArea: true,
    constraints: const BoxConstraints(maxWidth: 640),
    builder: (context) => SafeArea(
      top: false,
      child: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(
          24,
          0,
          24,
          24 + MediaQuery.viewInsetsOf(context).bottom,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'YOU’RE IN CONTROL',
              style: TextStyle(
                fontSize: 10,
                letterSpacing: 1.5,
                fontWeight: FontWeight.w700,
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
            const SizedBox(height: 14),
            Text(
              'Review action',
              style: TextStyle(
                fontFamily: 'InstrumentSerif',
                fontSize: 42,
                height: 1.05,
                color: Theme.of(context).colorScheme.onSurface,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              isDraft
                  ? 'This opens a draft in your messaging app. You decide when to send it.'
                  : 'Check the details before continuing.',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              describeAction(call),
              style: Theme.of(context).textTheme.titleLarge,
            ),
            if (detail != null) ...[
              const SizedBox(height: 16),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: SelectableText(
                  detail,
                  style: Theme.of(
                    context,
                  ).textTheme.bodyLarge?.copyWith(height: 1.5),
                ),
              ),
            ],
            const SizedBox(height: 28),
            FilledButton.icon(
              onPressed: () => Navigator.pop(context, true),
              icon: Icon(isDraft ? Icons.open_in_new : Icons.check_rounded),
              label: Text(isDraft ? 'Open draft' : 'Approve action'),
            ),
            const SizedBox(height: 10),
            OutlinedButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Not now'),
            ),
          ],
        ),
      ),
    ),
  );
  return approved ?? false;
}
