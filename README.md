# Offhand

Offhand is an offline voice assistant for Android. Talk to your phone and it gets things done for you, from setting an alarm to drafting a text, and it runs entirely on the device. Nothing you say leaves your phone.

## What it does

Ask in your own words and Offhand turns it into an action:

- Set alarms, timers, and reminders
- Add calendar events
- Draft texts and emails for you to review before sending
- Open settings like wifi, bluetooth, display, and sound
- Play music, toggle the flashlight, and take notes

Anything that sends a message or makes a visible change asks for your confirmation first, so nothing happens without your say so.

## Why it runs on your phone

- Private by default. Your voice and requests never go to a server.
- Works in airplane mode. No connection required.
- Quick to respond. It uses the phone's built in AI accelerator.

## How it works

Your speech is transcribed on the device. A compact language model running on the phone's NPU works out what you want and fills in the details. The app carries out the action once you approve it, then reads the result back to you.

## Reliability

Offhand is tested on how accurately it turns everyday requests into the right action with the right details. The test suite and its results live in the [eval](eval) folder.

## Requirements

- An Android phone with a supported on-device AI accelerator (built and tested on the Snapdragon 8 Elite)
- Microphone permission for voice input

## Status

In active development. The voice flow and core actions are being built and tested on real hardware. A demo and performance numbers will be posted here as they are ready.
