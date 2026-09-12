# Offhand app

The offline voice agent: a Flutter UI over a Kotlin platform-channel bridge to
the on-device runtime.

## What runs today

An end-to-end agent loop against a mock engine: speak or type a request → the
engine streams a tool call → the app parses it, shows an approval sheet, and on
approval performs the **real** Android action. Metrics (TTFT, tokens/sec) stream
back, and a benchmark screen characterizes the engine across the tool set. The
generation engine is still `MockEngine`; everything above the `Engine` interface
(execution, voice, benchmark) is real and engine-independent, so swapping in the
NPU runtime is a one-line change in `MainActivity`.

## Agent loop

1. **Voice or text in.** The mic button uses Android's SpeechRecognizer (via
   `speech_to_text`); recognized words flow into the prompt field.
2. **Generate.** The prompt goes to the engine over the control channel; tokens
   stream back over the token channel.
3. **Parse.** `lib/agent/parser.dart` extracts a `<tool_call>{…}</tool_call>`
   (mirroring the eval harness's strict parser).
4. **Approve.** `showApprovalSheet` shows a plain-English summary and, for
   drafts, the actual message/note body — nothing fires without approval.
5. **Execute.** `lib/agent/actions.dart` dispatches the call (see below).

## Tool execution (the 10 eval tools)

`lib/agent/dispatch.dart` maps a parsed `ToolCall` to a concrete Android
dispatch — kept pure and unit-tested, since the mapping is the easy part to get
wrong. Nine tools launch an Android intent via `android_intent_plus`; the
flashlight and media transport keys go over a native channel.

| tool | how |
|------|-----|
| `set_alarm` | `ACTION_SET_ALARM` (HOUR/MINUTES/MESSAGE) |
| `set_timer` | `ACTION_SET_TIMER` (LENGTH in seconds) |
| `create_calendar_event` | `ACTION_INSERT` on the events URI |
| `set_reminder` | `ACTION_INSERT` calendar event; calendar's default notification is the reminder (all-day today when no time) |
| `open_settings` | `ACTION_*_SETTINGS` per panel |
| `draft_sms` | `ACTION_SENDTO` `smsto:` + `sms_body` (composer opens; user sends) |
| `draft_email` | `ACTION_SENDTO` `mailto:` (RFC-6068 encoded subject/body) |
| `play_music` | query → `MEDIA_PLAY_FROM_SEARCH`; transport → native media key |
| `create_note` | `ACTION_SEND` `text/plain` (share to a notes app) |
| `toggle_flashlight` | native `CameraManager.setTorchMode` |

Drafts follow draft-don't-send: the SMS/email composer opens pre-filled and the
user sends it. Loose `time`/`start_datetime` strings are parsed best-effort in
`lib/agent/datetime.dart` (ISO-8601, `7am`, `HH:MM`, `today`/`tomorrow`); an
unparseable value opens the editor with the field left blank.

## Benchmark screen

Runs a representative prompt per tool (plus a no-call case) and reports median
TTFT, decode tok/s, and **tool-call latency** (time from `generate()` to the
first parseable tool call, measured live so it reflects a streaming engine), plus
process memory. "Copy JSON" exports the full run.

## Architecture

- `lib/bridge/bridge.dart`: `OffhandBridge` — control `MethodChannel`
  (`com.offhand/control`: loadModel, generate, stop) and token `EventChannel`
  (`com.offhand/tokens`).
- `lib/bridge/native_tools.dart`: `NativeTools` — `com.offhand/tools` channel for
  torch, media keys, and process memory.
- `android/.../Engine.kt`: the backend interface both engines implement.
- `android/.../MockEngine.kt`: streams a canned tool call per prompt with
  realistic timing (~22 tok/s, matching the device proof), covering all 10 tools.
- `android/.../GenieEngine.kt`: the stub where the on-device runtime (Genie /
  GenieX over the QAIRT context binary) gets wired in.
- `android/.../MainActivity.kt`: registers the channels and the native tools.

## Run it

Open the `app` folder in Android Studio and run on the S25 (or an emulator for
the mock). From the CLI: `flutter run`. Voice-in needs the microphone
permission (requested on first use).

Verified green: `flutter analyze`, `flutter test` (33 tests), and
`flutter build apk --debug`. On-device execution of the intents and voice-in is
the next step once a device is attached.

## Wiring the real runtime

Implement `GenieEngine` against the runtime proven in the device spike, then swap
`MockEngine()` for `GenieEngine()` in `MainActivity`. Nothing else changes — the
agent loop, tool execution, voice, and benchmark all sit above the `Engine`
interface. Keep the QAIRT version matched to the model bundle's compile version.

## Model

Use the 0.6B specialist (or the 1.7B generalist if the kill-test chooses it). The
eval study shows 4B gives no tool-calling gain and is memory-marginal on the
12 GB S25.
