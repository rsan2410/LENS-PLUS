export async function startCameraStream(targetFramerate = 15): Promise<MediaStream> {
  if (!("mediaDevices" in navigator) || !navigator.mediaDevices) {
    const secureHint =
      !window.isSecureContext
        ? "Camera APIs require HTTPS on most mobile browsers (or localhost)."
        : "This browser does not expose mediaDevices APIs.";
    throw new Error(`Camera API unavailable. ${secureHint}`);
  }

  if (typeof navigator.mediaDevices.getUserMedia !== "function") {
    throw new Error("getUserMedia is not available in this browser.");
  }

  const viewportIsPortrait = window.matchMedia("(orientation: portrait)").matches;
  const idealWidth = viewportIsPortrait ? 720 : 1280;
  const idealHeight = viewportIsPortrait ? 1280 : 720;
  const idealAspectRatio = viewportIsPortrait ? 9 / 16 : 16 / 9;

  const videoConstraints: MediaTrackConstraints = {
    facingMode: { ideal: "environment" },
    width: { ideal: idealWidth },
    height: { ideal: idealHeight },
    aspectRatio: { ideal: idealAspectRatio }
  };

  if (Number.isFinite(targetFramerate) && targetFramerate > 0) {
    videoConstraints.frameRate = {
      ideal: targetFramerate,
      max: targetFramerate
    };
  }

  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: videoConstraints
  });
}

export function stopStream(stream: MediaStream | null): void {
  if (!stream) {
    return;
  }
  for (const track of stream.getTracks()) {
    track.stop();
  }
}
