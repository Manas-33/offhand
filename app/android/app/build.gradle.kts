plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.offhand.offhand"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.offhand.offhand"
        // GenieX SDK ships arm64 natives and declares minSdk 27; the target
        // device is a Snapdragon 8 Elite, so pin both rather than inherit
        // Flutter's defaults.
        minSdk = 31
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
            // The GenieX AAR ships no consumer ProGuard rules for its
            // JNI-facing classes, so keep R8 away from them (the GenieX demo
            // app ships its release build unminified for the same reason).
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    packaging {
        // Genie/QNN native libs must be extracted on install (dlopen by path),
        // same setting the GenieX demo app uses.
        jniLibs.useLegacyPackaging = true
    }
}

dependencies {
    // GenieX Android binding (Maven Central): the NPU runtime proven in the
    // device spike and the model seeding.
    implementation("com.qualcomm.qti:geniex-android:0.3.5")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}

flutter {
    source = "../.."
}
