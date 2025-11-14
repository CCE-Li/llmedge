package io.aatricks.llmedge;

public class NativeTestProgressCallback {
    public int lastStep;
    public int lastFrame;
    public int lastTotalFrames;
    public float lastTime;
    public int callCount;

    public void onProgress(int step, int steps, int frame, int totalFrames, float timeMs) {
        this.lastStep = step;
        this.lastFrame = frame;
        this.lastTotalFrames = totalFrames;
        this.lastTime = timeMs;
        this.callCount++;
    }
}
