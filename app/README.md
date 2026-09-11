# Offhand app

The offline voice agent: a Flutter UI over a Kotlin platform-channel bridge to
the on-device runtime.

## What runs today

A working end-to-end skeleton. The UI loads a model, sends a prompt, and streams
tokens back with TTFT and tokens/sec. Generation is served by a mock engine, so
the whole pipeline (UI to bridge to native and back) is verifiable before the
NPU runtime is attached. On an emulator you see the mock stream; on the S25 the
same, until the real engine is wired in.

## Architecture

- `lib/bridge.dart`: `OffhandBridge` wraps a `MethodChannel` (`com.offhand/control`:
  loadModel, generate, stop) and an `EventChannel` (`com.offhand/tokens`: streamed
  token and done events).
- `lib/main.dart`: the home screen (load, prompt, streaming output, metrics bar).
- `android/.../Engine.kt`: the backend interface both engines implement.
- `android/.../MockEngine.kt`: streams a canned response with realistic timing
  (~22 tok/s, matching the device proof).
- `android/.../GenieEngine.kt`: the stub where the on-device runtime (Genie /
  GenieX over the QAIRT context binary) gets wired in.
- `android/.../MainActivity.kt`: registers the channels and forwards engine
  events to Flutter.

## Run it

Open the `app` folder in Android Studio and run on the S25 (or an emulator for
the mock). From the CLI, with the Android SDK and a JDK on PATH: `flutter run`.

## Wiring the real runtime

Implement `GenieEngine` against the runtime proven in the device spike, then swap
`MockEngine()` for `GenieEngine()` in `MainActivity`. Keep the QAIRT version
matched to the model bundle's compile version.

## Model

Use the 1.7B bundle. The eval study shows 1.7B is the on-device sweet spot: 4B
gives no accuracy gain on tool calling and is memory-marginal on the 12 GB S25.
