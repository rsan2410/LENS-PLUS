import { sendIceCandidate, sendOffer } from "./signaling";

type ConnectionStateHandler = (state: RTCPeerConnectionState) => void;
type DataMessageHandler = (data: string) => void;
type LogHandler = (message: string) => void;

export class WebRtcClient {
  private peerConnection: RTCPeerConnection | null = null;
  private videoSender: RTCRtpSender | null = null;
  private sessionId: string | null = null;
  private pendingIceCandidates: RTCIceCandidate[] = [];
  private connectionStateHandler: ConnectionStateHandler;
  private dataMessageHandler: DataMessageHandler;
  private logHandler: LogHandler;

  constructor(
    connectionStateHandler: ConnectionStateHandler,
    dataMessageHandler: DataMessageHandler,
    logHandler: LogHandler
  ) {
    this.connectionStateHandler = connectionStateHandler;
    this.dataMessageHandler = dataMessageHandler;
    this.logHandler = logHandler;
  }

  getSessionId(): string | null {
    return this.sessionId;
  }

  async connect(stream: MediaStream): Promise<void> {
    if (this.peerConnection) {
      await this.disconnect();
    }

    const peerConnection = new RTCPeerConnection({ iceServers: [] });
    this.peerConnection = peerConnection;
    this.videoSender = null;
    this.pendingIceCandidates = [];

    const channel = peerConnection.createDataChannel("results");
    channel.onopen = () => this.logHandler("Data channel open");
    channel.onclose = () => this.logHandler("Data channel closed");
    channel.onmessage = (event) => this.dataMessageHandler(String(event.data));

    for (const track of stream.getTracks()) {
      const sender = peerConnection.addTrack(track, stream);
      if (track.kind === "video") {
        this.videoSender = sender;
      }
    }

    peerConnection.onconnectionstatechange = () => {
      this.connectionStateHandler(peerConnection.connectionState);
      this.logHandler(`Peer state: ${peerConnection.connectionState}`);
    };

    peerConnection.oniceconnectionstatechange = () => {
      this.logHandler(`ICE state: ${peerConnection.iceConnectionState}`);
    };

    peerConnection.onicecandidate = (event) => {
      if (!event.candidate) {
        return;
      }

      if (!this.sessionId) {
        this.pendingIceCandidates.push(event.candidate);
        return;
      }

      void this.sendIce(event.candidate);
    };

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    await waitForIceGathering(peerConnection);

    const localDescription = peerConnection.localDescription;
    if (!localDescription?.sdp) {
      throw new Error("Failed to create local SDP offer");
    }

    const answer = await sendOffer({
      sdp: localDescription.sdp,
      type: localDescription.type
    });

    this.sessionId = answer.session_id;

    while (this.pendingIceCandidates.length > 0) {
      const candidate = this.pendingIceCandidates.shift();
      if (!candidate) {
        break;
      }
      await this.sendIce(candidate);
    }

    await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
    this.logHandler(`Connected signaling session ${this.sessionId}`);
  }

  async setOutgoingFramerate(maxFps: number): Promise<boolean> {
    if (!this.videoSender) {
      return false;
    }

    if (!Number.isFinite(maxFps) || maxFps <= 0) {
      return false;
    }

    try {
      const parameters = this.videoSender.getParameters();
      if (!parameters.encodings || parameters.encodings.length === 0) {
        parameters.encodings = [{}];
      }

      for (const encoding of parameters.encodings) {
        encoding.maxFramerate = maxFps;
      }

      await this.videoSender.setParameters(parameters);
      return true;
    } catch (error: unknown) {
      this.logHandler(`Outgoing FPS update failed: ${String(error)}`);
      return false;
    }
  }

  private async sendIce(candidate: RTCIceCandidate): Promise<void> {
    if (!this.sessionId) {
      return;
    }
    try {
      await sendIceCandidate({
        session_id: this.sessionId,
        candidate: candidate.candidate,
        sdpMid: candidate.sdpMid,
        sdpMLineIndex: candidate.sdpMLineIndex
      });
    } catch (error: unknown) {
      this.logHandler(`ICE send failed: ${String(error)}`);
    }
  }

  async disconnect(): Promise<void> {
    if (this.peerConnection) {
      this.peerConnection.close();
      this.peerConnection = null;
    }
    this.videoSender = null;
    this.sessionId = null;
    this.pendingIceCandidates = [];
    this.connectionStateHandler("closed");
  }
}

async function waitForIceGathering(
  peerConnection: RTCPeerConnection,
  timeoutMs = 3000
): Promise<void> {
  if (peerConnection.iceGatheringState === "complete") {
    return;
  }

  await new Promise<void>((resolve) => {
    const timeout = window.setTimeout(() => {
      peerConnection.removeEventListener("icegatheringstatechange", onStateChange);
      resolve();
    }, timeoutMs);

    const onStateChange = (): void => {
      if (peerConnection.iceGatheringState === "complete") {
        window.clearTimeout(timeout);
        peerConnection.removeEventListener("icegatheringstatechange", onStateChange);
        resolve();
      }
    };

    peerConnection.addEventListener("icegatheringstatechange", onStateChange);
  });
}
