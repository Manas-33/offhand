package com.offhand.offhand

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.os.Debug
import android.view.KeyEvent

/**
 * The tool actions no Android intent can perform, dispatched over the
 * `com.offhand/tools` method channel: the flashlight (CameraManager torch) and
 * media transport keys (routed to the active media session). Also exposes the
 * process memory reading the benchmark screen shows.
 */
class NativeTools(private val context: Context) {

    private val cameraManager: CameraManager
        get() = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private val audioManager: AudioManager
        get() = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

    /** The first camera that has a controllable flash unit, if any. */
    private fun torchCameraId(): String? =
        cameraManager.cameraIdList.firstOrNull { id ->
            cameraManager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
        }

    /** Turn the torch on/off. Returns false if the device has no torch. */
    fun setTorch(on: Boolean): Boolean {
        val id = torchCameraId() ?: return false
        return try {
            cameraManager.setTorchMode(id, on)
            true
        } catch (e: Exception) {
            false
        }
    }

    /** Dispatch a media key to whatever holds the active session. */
    fun mediaKey(action: String) {
        val code = when (action) {
            "pause" -> KeyEvent.KEYCODE_MEDIA_PAUSE
            "next" -> KeyEvent.KEYCODE_MEDIA_NEXT
            "previous" -> KeyEvent.KEYCODE_MEDIA_PREVIOUS
            else -> KeyEvent.KEYCODE_MEDIA_PLAY
        }
        audioManager.dispatchMediaKeyEvent(KeyEvent(KeyEvent.ACTION_DOWN, code))
        audioManager.dispatchMediaKeyEvent(KeyEvent(KeyEvent.ACTION_UP, code))
    }

    /** Total PSS of this process, in MB. */
    fun memoryMb(): Double {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024.0 // totalPss is reported in KB
    }
}
