# Consumer ProGuard rules for llmedge library
# These rules will be included in apps that use this library

# Keep all vision and OCR API classes
-keep class io.aatricks.llmedge.vision.** { *; }
-keep interface io.aatricks.llmedge.vision.** { *; }


# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep ML Kit classes
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.internal.mlkit_text_recognition.** { *; }
-keep class com.google.android.gms.internal.mlkit_text_recognition_common.** { *; }

# Suppress warnings for optional dependencies
-dontwarn com.google.mlkit.**

# Keep enums
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep Parcelable
-keepclassmembers class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator CREATOR;
}

# Keep serializable classes
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}

# Keep data classes
-keep class io.aatricks.llmedge.vision.ImageSource$* { *; }
-keep class io.aatricks.llmedge.vision.OcrParams { *; }
-keep class io.aatricks.llmedge.vision.OcrResult { *; }
-keep class io.aatricks.llmedge.vision.VisionParams { *; }
-keep class io.aatricks.llmedge.vision.VisionResult { *; }
-keep class io.aatricks.llmedge.vision.ImageUnderstandingResult { *; }
-keep enum io.aatricks.llmedge.vision.VisionMode { *; }