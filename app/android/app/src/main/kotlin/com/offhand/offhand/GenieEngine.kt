package com.offhand.offhand

import android.content.Context
import android.util.Log
import com.geniex.sdk.GenieXSdk
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.bean.ComputeUnitValue
import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ModelConfig
import java.io.File
import java.util.concurrent.CountDownLatch
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.json.JSONObject

/**
 * The real on-device engine: the GenieX SDK's qairt plugin running the W4A16
 * specialist bundle on the Hexagon NPU, the same runtime proven in the device
 * spike and the seeding run.
 *
 * loadModel -> create an LlmWrapper over the bundle's QAIRT context binaries.
 *              The path may be absolute or relative to the app's filesDir.
 *              The bundle's geniex.json names the model and the shard to open,
 *              read the way the GenieX demo reads them, so stock catalog
 *              bundles (1.7B, 4B) load the same way as the 0.6B specialists.
 * generate  -> send PREFIX + user text + SUFFIX straight to the runtime. The
 *              prefix and suffix (assets/prompt_template.json) are rendered on
 *              the host by eval/export_app_prompt.py through the eval's own
 *              apply_chat_template call, so every request is byte-identical to
 *              what the GPU eval runs saw. The SDK's applyChatTemplate is not
 *              used on purpose: it stops including the tools after the first
 *              call. The runtime is reset before each request so none inherits
 *              state from the one before.
 * stop      -> stop the native decode; the flow then completes normally and
 *              still emits the "done" event.
 *
 * Decoding is greedy, the eval's contract: sampling is off in the create input
 * and the bundle's genie_config.json pins top-k to 1.
 *
 * Blocking calls run on the caller's thread (MainActivity invokes loadModel
 * off the main thread); generation runs on an internal scope.
 */
class GenieEngine(private val context: Context) : Engine {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var llm: LlmWrapper? = null
    private var generateJob: Job? = null

    @Volatile private var sdkReady = false
    @Volatile private var stopping = false

    /** Fixed text before and after the user request. */
    private val promptParts: Pair<String, String> by lazy {
        val json = JSONObject(context.assets.open(PROMPT_TEMPLATE).bufferedReader().use { it.readText() })
        json.getString("prefix") to json.getString("suffix")
    }

    override fun loadModel(modelPath: String): Boolean {
        if (llm != null) return true
        if (runCatching { promptParts }.isFailure) {
            Log.e(TAG, "missing or unreadable asset $PROMPT_TEMPLATE")
            return false
        }
        if (!ensureSdk()) return false

        val dir = if (modelPath.startsWith("/")) File(modelPath) else File(context.filesDir, modelPath)
        val (modelName, shardName) = bundleManifest(dir)
        val shard = File(dir, shardName)
        if (!shard.exists()) {
            Log.e(TAG, "model shard not found: ${shard.absolutePath}")
            return false
        }

        // QAIRT rejects non-zero nCtx / nGpuLayers (both fixed at compile time
        // in the bundle), same as the GenieX demo's qairt path.
        val input =
            LlmCreateInput(
                model_name = modelName,
                model_path = shard.absolutePath,
                tokenizer_path = null,
                config = ModelConfig(nCtx = 0, nGpuLayers = 0, enable_thinking = false),
                runtime_id = GenieXSdk.PLUGIN_ID_QAIRT,
                compute_unit = ComputeUnitValue.NPU.value,
            )

        val result = runBlocking { LlmWrapper.builder().llmCreateInput(input).build() }
        return result.fold(
            onSuccess = { wrapper ->
                llm = wrapper
                // What we asked for. The SDK's own "QAIRT LLM created
                // successfully" line is the confirmation it runs on the NPU.
                Log.i(TAG, "model loaded from ${dir.absolutePath} ($modelName, requested plugin=qairt, compute_unit=NPU)")
                true
            },
            onFailure = { error ->
                Log.e(TAG, "model load failed: ${error.message}")
                false
            },
        )
    }

    override fun generate(prompt: String, emit: (Map<String, Any?>) -> Unit) {
        val wrapper = llm
        if (wrapper == null) {
            emit(mapOf("type" to "error", "message" to "model not loaded"))
            return
        }
        stopping = false
        val (prefix, suffix) = promptParts
        generateJob =
            scope.launch {
                try {
                    // Each request stands alone: clear whatever the previous
                    // one left in the runtime before prefilling this one.
                    val rc = wrapper.reset()
                    if (rc != 0) Log.w(TAG, "reset returned $rc")
                    streamGenerate(wrapper, prefix + prompt + suffix, emit)
                } catch (e: Exception) {
                    Log.e(TAG, "generate failed", e)
                    emit(mapOf("type" to "error", "message" to (e.message ?: "generation failed")))
                }
            }
    }

    private suspend fun streamGenerate(
        wrapper: LlmWrapper,
        fullPrompt: String,
        emit: (Map<String, Any?>) -> Unit,
    ) {
        val output = StringBuilder()
        wrapper.generateStreamFlow(fullPrompt, GenerationConfig(maxTokens = MAX_TOKENS)).collect { result ->
            when (result) {
                is LlmStreamResult.Token -> {
                    output.append(result.text)
                    emit(mapOf("type" to "token", "text" to result.text))
                }

                is LlmStreamResult.Completed -> {
                    val p = result.profile
                    Log.d(TAG, "output: $output")
                    Log.d(TAG, "Completed: $p")
                    emit(
                        mapOf(
                            "type" to "done",
                            "ttftMs" to p.ttftMs.toLong(),
                            "tps" to p.decodingSpeed,
                            "tokens" to p.generatedTokens.toInt(),
                            "totalMs" to (p.promptTimeMs + p.decodeTimeMs).toLong(),
                            "cancelled" to stopping,
                            // Extra runtime numbers for the eval replay
                            // (EvalActivity); the Dart bridge ignores them.
                            "promptTokens" to p.promptTokens.toInt(),
                            "prefillTps" to p.prefillSpeed,
                            "stopReason" to p.stopReason,
                        ),
                    )
                }

                is LlmStreamResult.Error -> {
                    Log.e(TAG, "generate error", result.throwable)
                    emit(
                        mapOf(
                            "type" to "error",
                            "message" to (result.throwable.message ?: "generation failed"),
                        ),
                    )
                }
            }
        }
    }

    override fun stop() {
        stopping = true
        // Stop the native decode rather than cancelling the collector: the
        // flow then completes on its own and the "done" event still fires.
        scope.launch {
            runCatching { llm?.stopStream() }
                .onFailure { Log.w(TAG, "stopStream failed: ${it.message}") }
        }
    }

    /**
     * The model name and the shard to open, from the bundle's geniex.json the
     * way the GenieX demo reads them. A bundle without a readable one falls
     * back to the 0.6B's.
     */
    private fun bundleManifest(dir: File): Pair<String, String> {
        val json = runCatching { JSONObject(File(dir, MANIFEST).readText()) }.getOrNull()
            ?: return MODEL_NAME to MODEL_SHARD
        val name = json.optString("ModelName").ifEmpty { MODEL_NAME }
        val files = json.optJSONObject("ModelFile")
        val shard = files?.keys()?.asSequence()?.firstOrNull()?.let { files.optJSONObject(it)?.optString("Name") }
        return name to (shard?.ifEmpty { null } ?: MODEL_SHARD)
    }

    /** One-time SDK init; blocks until the native side reports back. */
    private fun ensureSdk(): Boolean {
        if (sdkReady) return true
        var ok = false
        val latch = CountDownLatch(1)
        GenieXSdk.getInstance().init(
            context,
            object : GenieXSdk.InitCallback {
                override fun onSuccess() {
                    ok = true
                    latch.countDown()
                }

                override fun onFailure(reason: String) {
                    Log.e(TAG, "GenieXSdk init failed: $reason")
                    latch.countDown()
                }
            },
        )
        latch.await()
        sdkReady = ok
        return ok
    }

    companion object {
        private const val TAG = "Offhand"

        /** Read from each bundle for the model name and the shard to open. */
        private const val MANIFEST = "geniex.json"

        /** Fallbacks for a bundle without a readable geniex.json: the 0.6B's name and shard. */
        private const val MODEL_NAME = "qwen3_0_6b"

        /** The qairt plugin discovers all shards from the directory of this file. */
        private const val MODEL_SHARD = "part2_of_2.bin"

        /** Same budget as the eval harness's max_new_tokens. */
        private const val MAX_TOKENS = 256

        /** Written by eval/export_app_prompt.py. */
        private const val PROMPT_TEMPLATE = "prompt_template.json"
    }
}
