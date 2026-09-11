package com.offhand.offhand

/**
 * The real on-device engine. This is where the Qualcomm runtime proven out in
 * the device spike (Genie/GenieX over the QAIRT context binary) gets wired in:
 *
 *   loadModel -> map the exported model bundle at modelPath (the QAIRT context
 *                binary + genie config pushed to the device)
 *   generate  -> run streaming inference on the Hexagon NPU, calling emit() with
 *                a "token" event per decoded token, then a "done" event carrying
 *                TTFT and tokens/sec
 *   stop      -> cancel the in-flight decode
 *
 * Not implemented yet. MainActivity uses MockEngine until this lands. Keep the
 * QAIRT version matched to the bundle's compile version (the device-proof trap).
 */
class GenieEngine : Engine {
    override fun loadModel(modelPath: String): Boolean {
        TODO("Load the QAIRT/Genie context binary from modelPath")
    }

    override fun generate(prompt: String, emit: (Map<String, Any?>) -> Unit) {
        TODO("Run streaming NPU inference and emit token/done events")
    }

    override fun stop() {
        TODO("Cancel the in-flight generation")
    }
}
