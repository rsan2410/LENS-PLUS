from __future__ import annotations

import asyncio
import io
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import choice, random
from time import monotonic
from typing import Any

import av.logging
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel


class OfferRequest(BaseModel):
    sdp: str
    type: str
    session_id: str | None = None


class OfferResponse(BaseModel):
    sdp: str
    type: str
    session_id: str


class IceRequest(BaseModel):
    session_id: str
    candidate: str
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None


class IceResponse(BaseModel):
    ok: bool


class InferenceIngestRequest(BaseModel):
    timestamp: str | None = None
    guidance_text: str | None = None
    scene_summary: str | None = None
    objects: list[dict[str, Any]] = []


DEFAULT_ANALYSIS_TARGET_FPS = 15.0
MIN_ANALYSIS_TARGET_FPS = 1.0
MAX_ANALYSIS_TARGET_FPS = 30.0
FPS_WINDOW_SECONDS = 1.0
SESSION_MANIFEST_FILENAME = "session.json"
MAX_INFERENCE_SAMPLE_AGE_MS = 3000
MAX_DIRECTIONAL_SAMPLE_AGE_MS = 5000
GROUP_SESSION_TIME_CUT_SECONDS = 10 # seconds


def clamp_analysis_fps(value: float) -> float:
    if not math.isfinite(value):
        return DEFAULT_ANALYSIS_TARGET_FPS

    return max(MIN_ANALYSIS_TARGET_FPS, min(MAX_ANALYSIS_TARGET_FPS, value))


def load_analysis_target_fps_from_env() -> float:
    raw_value = os.getenv("ANALYSIS_TARGET_FPS")
    if raw_value is None:
        return DEFAULT_ANALYSIS_TARGET_FPS

    try:
        return clamp_analysis_fps(float(raw_value))
    except ValueError:
        return DEFAULT_ANALYSIS_TARGET_FPS


CONFIGURED_ANALYSIS_TARGET_FPS = load_analysis_target_fps_from_env()


def load_session_artifacts_root_from_env() -> Path:
    raw_value = os.getenv("SESSION_ARTIFACTS_DIR")
    if raw_value:
        return Path(raw_value).expanduser().resolve()

    return Path(__file__).resolve().parent / "session_artifacts"


SESSION_ARTIFACTS_ROOT = load_session_artifacts_root_from_env()


@dataclass
class Session:
    peer_connection: RTCPeerConnection
    data_channel: Any | None = None
    frame_task: asyncio.Task[None] | None = None
    mock_task: asyncio.Task[None] | None = None
    analysis_target_fps: float = CONFIGURED_ANALYSIS_TARGET_FPS
    total_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    incoming_fps: float = 0.0
    processed_fps: float = 0.0
    last_frame_at: datetime | None = None
    latest_jpeg: bytes | None = None
    latest_jpeg_at: datetime | None = None
    snapshot_errors: int = 0
    last_snapshot_error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_id: str | None = None
    artifact_dir: Path | None = None
    artifact_manifest_path: Path | None = None
    dumped_frames: int = 0
    dump_errors: int = 0
    last_dump_error: str | None = None
    directional_samples_received: int = 0
    directional_messages_ignored: int = 0
    directional_parse_errors: int = 0
    latest_directional_sample: dict[str, Any] | None = None
    latest_directional_received_at: datetime | None = None
    latest_directional_client_timestamp_ms: int | None = None
    latest_processed_directional: dict[str, Any] | None = None
    inference_messages_emitted: int = 0
    latest_inference_payload: dict[str, Any] | None = None
    latest_inference_at: datetime | None = None
    latest_processed_detections: dict[str, Any] | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


app = FastAPI(title="lens-plus-signaling")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, Session] = {}


def read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


SNAPSHOT_INTERVAL_SECONDS = max(0.03, read_float_env("SNAPSHOT_INTERVAL_SECONDS", 0.05))
SNAPSHOT_JPEG_QUALITY = min(95, max(60, read_int_env("SNAPSHOT_JPEG_QUALITY", 92)))


@app.on_event("startup")
async def startup() -> None:
    av.logging.set_level(av.logging.ERROR)
    SESSION_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    app.state.gc_task = asyncio.create_task(session_gc())


@app.on_event("shutdown")
async def shutdown() -> None:
    gc_task: asyncio.Task[None] | None = getattr(app.state, "gc_task", None)
    if gc_task:
        gc_task.cancel()
    for session_id in list(sessions.keys()):
        await close_session(session_id)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/sessions")
async def debug_sessions() -> dict[str, list[dict[str, Any]]]:
    session_list: list[dict[str, Any]] = []
    for session_id, session in sessions.items():
        session_list.append(
            {
                "session_id": session_id,
                "connection_state": session.peer_connection.connectionState,
                "ice_state": session.peer_connection.iceConnectionState,
                "analysis_target_fps": session.analysis_target_fps,
                "total_frames": session.total_frames,
                "processed_frames": session.processed_frames,
                "dropped_frames": session.dropped_frames,
                "incoming_fps": session.incoming_fps,
                "processed_fps": session.processed_fps,
                "last_frame_at": (
                    session.last_frame_at.isoformat() if session.last_frame_at else None
                ),
                "latest_jpeg_at": (
                    session.latest_jpeg_at.isoformat()
                    if session.latest_jpeg_at
                    else None
                ),
                "has_snapshot": session.latest_jpeg is not None,
                "snapshot_errors": session.snapshot_errors,
                "last_snapshot_error": session.last_snapshot_error,
                "started_at": session.started_at.isoformat(),
                "artifact_id": session.artifact_id,
                "artifact_dir": str(session.artifact_dir)
                if session.artifact_dir
                else None,
                "dumped_frames": session.dumped_frames,
                "dump_errors": session.dump_errors,
                "last_dump_error": session.last_dump_error,
                "directional_samples_received": session.directional_samples_received,
                "directional_messages_ignored": session.directional_messages_ignored,
                "directional_parse_errors": session.directional_parse_errors,
                "latest_directional_received_at": (
                    session.latest_directional_received_at.isoformat()
                    if session.latest_directional_received_at
                    else None
                ),
                "latest_directional_client_timestamp_ms": (
                    session.latest_directional_client_timestamp_ms
                ),
                "latest_processed_directional": session.latest_processed_directional,
                "inference_messages_emitted": session.inference_messages_emitted,
                "latest_inference_at": (
                    session.latest_inference_at.isoformat()
                    if session.latest_inference_at
                    else None
                ),
                "latest_processed_detections": session.latest_processed_detections,
                "updated_at": session.updated_at.isoformat(),
            }
        )

    return {"sessions": session_list}


@app.get("/debug/sessions/history")
async def debug_sessions_history() -> dict[str, list[dict[str, Any]]]:
    return {"sessions": load_session_history()}


@app.get("/debug/sessions/{session_id}/latest.jpg")
async def debug_latest_snapshot(session_id: str) -> Response:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session")

    if not session.latest_jpeg:
        raise HTTPException(status_code=404, detail="No snapshot available yet")

    return Response(
        content=session.latest_jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/webrtc/offer", response_model=OfferResponse)
async def offer(payload: OfferRequest) -> OfferResponse:
    for existing_session_id in list(sessions.keys()):
        await close_session(existing_session_id)

    peer_connection = RTCPeerConnection()

    session_id = payload.session_id or str(uuid.uuid4())
    session_started_at = datetime.now(timezone.utc)
    artifact_id, artifact_dir, artifact_manifest_path, artifact_error = (
        create_session_artifact(session_id=session_id, started_at=session_started_at)
    )
    session = Session(
        peer_connection=peer_connection,
        started_at=session_started_at,
        updated_at=session_started_at,
        artifact_id=artifact_id,
        artifact_dir=artifact_dir,
        artifact_manifest_path=artifact_manifest_path,
    )
    if artifact_error is not None:
        session.dump_errors += 1
        session.last_dump_error = artifact_error

    sessions[session_id] = session
    write_session_manifest(session_id, session)

    @peer_connection.on("track")
    async def on_track(track: Any) -> None:
        if getattr(track, "kind", "") != "video":
            return

        session.updated_at = datetime.now(timezone.utc)

        async def consume_frames() -> None:
            last_snapshot_at = datetime.min.replace(tzinfo=timezone.utc)
            fps_window_started_at = monotonic()
            window_incoming_frames = 0
            window_processed_frames = 0
            next_analysis_at = 0.0

            def update_fps_window(now_mono: float) -> None:
                nonlocal fps_window_started_at
                nonlocal window_incoming_frames
                nonlocal window_processed_frames

                elapsed = now_mono - fps_window_started_at
                if elapsed < FPS_WINDOW_SECONDS:
                    return

                session.incoming_fps = round(window_incoming_frames / elapsed, 2)
                session.processed_fps = round(window_processed_frames / elapsed, 2)
                fps_window_started_at = now_mono
                window_incoming_frames = 0
                window_processed_frames = 0

            while True:
                try:
                    frame = await track.recv()
                    now = datetime.now(timezone.utc)
                    now_mono = monotonic()
                    session.total_frames += 1
                    session.last_frame_at = now
                    session.updated_at = now
                    window_incoming_frames += 1

                    snapshot_due = (
                        (now - last_snapshot_at).total_seconds()
                        >= SNAPSHOT_INTERVAL_SECONDS
                    )
                    snapshot_jpeg: bytes | None = None
                    snapshot_error: str | None = None
                    if snapshot_due:
                        snapshot_jpeg, snapshot_error = frame_to_jpeg(
                            frame, jpeg_quality=SNAPSHOT_JPEG_QUALITY
                        )
                        if snapshot_jpeg:
                            session.latest_jpeg = snapshot_jpeg
                            session.latest_jpeg_at = now
                            last_snapshot_at = now
                            session.last_snapshot_error = None
                        else:
                            session.snapshot_errors += 1
                            session.last_snapshot_error = snapshot_error

                    target_analysis_fps = clamp_analysis_fps(
                        session.analysis_target_fps
                    )
                    analysis_interval_seconds = 1.0 / target_analysis_fps

                    if now_mono < next_analysis_at:
                        session.dropped_frames += 1
                        update_fps_window(now_mono)
                        continue

                    next_analysis_at = now_mono + analysis_interval_seconds
                    session.processed_frames += 1
                    window_processed_frames += 1
                    directional_context = build_directional_context_for_frame(
                        session=session,
                        frame_at=now,
                    )
                    session.latest_processed_directional = directional_context
                    detection_context = build_detection_context_for_frame(
                        session=session,
                        frame_at=now,
                    )
                    session.latest_processed_detections = detection_context

                    processed_frame_jpeg = snapshot_jpeg
                    processed_frame_error = snapshot_error
                    if processed_frame_jpeg is None:
                        processed_frame_jpeg, processed_frame_error = frame_to_jpeg(
                            frame, jpeg_quality=SNAPSHOT_JPEG_QUALITY
                        )

                    if processed_frame_jpeg:
                        persist_processed_frame(
                            session=session,
                            frame_jpeg=processed_frame_jpeg,
                            frame_at=now,
                            directional_context=directional_context,
                            detection_context=detection_context,
                        )
                    else:
                        session.dump_errors += 1
                        session.last_dump_error = (
                            f"processed frame encode failed: {processed_frame_error}"
                        )

                    update_fps_window(now_mono)
                except Exception:
                    break

        session.frame_task = asyncio.create_task(consume_frames())

    @peer_connection.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        session.data_channel = channel
        session.updated_at = datetime.now(timezone.utc)

        @channel.on("message")
        def on_message(message: Any) -> None:
            ingest_directional_message(session=session, message=message)

        @channel.on("open")
        def on_open() -> None:
            if session.mock_task is None:
                session.mock_task = asyncio.create_task(send_mock_results(session))

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if peer_connection.connectionState in {"failed", "closed"}:
            await close_session(session_id)

    offer_description = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
    await peer_connection.setRemoteDescription(offer_description)
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    await wait_for_ice_gathering(peer_connection)
    session.updated_at = datetime.now(timezone.utc)

    if peer_connection.localDescription is None:
        raise HTTPException(status_code=500, detail="Failed to build answer")

    return OfferResponse(
        sdp=peer_connection.localDescription.sdp,
        type=peer_connection.localDescription.type,
        session_id=session_id,
    )


@app.post("/webrtc/ice", response_model=IceResponse)
async def ice(payload: IceRequest) -> IceResponse:
    session = sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session")

    try:
        candidate_sdp = payload.candidate
        if candidate_sdp.startswith("candidate:"):
            candidate_sdp = candidate_sdp[len("candidate:") :]

        candidate = candidate_from_sdp(candidate_sdp)
        candidate.sdpMid = payload.sdpMid
        candidate.sdpMLineIndex = payload.sdpMLineIndex
        await session.peer_connection.addIceCandidate(candidate)
        session.updated_at = datetime.now(timezone.utc)
    except Exception as error:
        raise HTTPException(
            status_code=400, detail=f"Invalid ICE candidate: {error}"
        ) from error

    return IceResponse(ok=True)


@app.post("/debug/sessions/{session_id}/inference", response_model=IceResponse)
async def ingest_inference(
    session_id: str,
    payload: InferenceIngestRequest,
) -> IceResponse:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session")

    normalized_payload = {
        "timestamp": payload.timestamp or datetime.now(timezone.utc).isoformat(),
        "guidance_text": payload.guidance_text,
        "scene_summary": payload.scene_summary,
        "objects": payload.objects,
    }

    update_latest_inference(session=session, payload=normalized_payload)
    session.inference_messages_emitted += 1
    session.updated_at = datetime.now(timezone.utc)

    # If external detections are being ingested, stop mock updates to avoid
    # overwriting real detection state used by artifact dumps.
    if session.mock_task is not None:
        session.mock_task.cancel()
        session.mock_task = None

    return IceResponse(ok=True)


async def send_mock_results(session: Session) -> None:
    object_labels = ["chair", "desk", "door", "bag", "keys", "person"]
    while True:
        channel = session.data_channel
        if channel is None:
            await asyncio.sleep(1)
            continue

        if getattr(channel, "readyState", "") != "open":
            await asyncio.sleep(1)
            continue

        label = choice(object_labels)
        confidence = round(0.55 + random() * 0.4, 2)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "guidance_text": f"Caution: {label} ahead.",
            "scene_summary": f"Detected one {label} in view.",
            "directional": session.latest_processed_directional,
            "objects": [
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": [0.2, 0.2, 0.35, 0.4],
                }
            ],
        }

        try:
            channel.send(json.dumps(payload))
            update_latest_inference(session=session, payload=payload)
            session.inference_messages_emitted += 1
            session.updated_at = datetime.now(timezone.utc)
        except Exception:
            break

        await asyncio.sleep(1)


def ingest_directional_message(session: Session, message: Any) -> None:
    payload: Any = message
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            session.directional_parse_errors += 1
            return

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            session.directional_parse_errors += 1
            return

    normalized_payload = normalize_directional_payload(payload)
    if normalized_payload is None:
        session.directional_messages_ignored += 1
        return

    now = datetime.now(timezone.utc)
    session.directional_samples_received += 1
    session.latest_directional_sample = normalized_payload
    session.latest_directional_received_at = now
    session.latest_directional_client_timestamp_ms = normalized_payload["timestamp_ms"]
    session.updated_at = now


def normalize_directional_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    if payload.get("type") != "client_sensor":
        return None

    if payload.get("sensor") != "gyro":
        return None

    timestamp_ms = coerce_optional_int(payload.get("timestamp_ms"))
    rotation_rate_dps = normalize_rotation_rate(payload.get("rotation_rate_dps"))
    orientation_deg = normalize_orientation(payload.get("orientation_deg"))

    if rotation_rate_dps is None and orientation_deg is None:
        return None

    return {
        "timestamp_ms": timestamp_ms,
        "rotation_rate_dps": rotation_rate_dps,
        "orientation_deg": orientation_deg,
    }


def normalize_rotation_rate(value: Any) -> dict[str, float | None] | None:
    if not isinstance(value, dict):
        return None

    alpha = coerce_finite_float(value.get("alpha"))
    beta = coerce_finite_float(value.get("beta"))
    gamma = coerce_finite_float(value.get("gamma"))
    if alpha is None and beta is None and gamma is None:
        return None

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


def normalize_orientation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    alpha = coerce_finite_float(value.get("alpha"))
    beta = coerce_finite_float(value.get("beta"))
    gamma = coerce_finite_float(value.get("gamma"))
    absolute = value.get("absolute") if isinstance(value.get("absolute"), bool) else None
    if alpha is None and beta is None and gamma is None and absolute is None:
        return None

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "absolute": absolute,
    }


def coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)

    return None


def coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float) and math.isfinite(value):
        return int(value)

    return None


def build_directional_context_for_frame(
    session: Session,
    frame_at: datetime,
) -> dict[str, Any] | None:
    latest_directional_sample = session.latest_directional_sample
    latest_directional_received_at = session.latest_directional_received_at
    if latest_directional_sample is None or latest_directional_received_at is None:
        return None

    age_ms = max(
        0,
        int(round((frame_at - latest_directional_received_at).total_seconds() * 1000)),
    )
    return {
        "sample_timestamp_ms": latest_directional_sample.get("timestamp_ms"),
        "server_received_at": latest_directional_received_at.isoformat(),
        "age_ms": age_ms,
        "is_stale": age_ms > MAX_DIRECTIONAL_SAMPLE_AGE_MS,
        "rotation_rate_dps": latest_directional_sample.get("rotation_rate_dps"),
        "orientation_deg": latest_directional_sample.get("orientation_deg"),
    }


def update_latest_inference(session: Session, payload: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    session.latest_inference_payload = payload
    session.latest_inference_at = now


def normalize_inference_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        return []

    normalized_objects: list[dict[str, Any]] = []
    for raw_obj in raw_objects:
        if not isinstance(raw_obj, dict):
            continue
        label = raw_obj.get("label")
        confidence = raw_obj.get("confidence")
        bbox = raw_obj.get("bbox")
        if not isinstance(label, str):
            continue
        confidence_value = None
        if isinstance(confidence, (int, float)) and math.isfinite(confidence):
            confidence_value = float(confidence)

        normalized_bbox = None
        if isinstance(bbox, list) and len(bbox) == 4:
            parsed_bbox: list[float] = []
            for value in bbox:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    parsed_bbox.append(float(value))
            if len(parsed_bbox) == 4:
                normalized_bbox = parsed_bbox

        normalized_objects.append(
            {
                "label": label,
                "confidence": confidence_value,
                "bbox": normalized_bbox,
            }
        )
    return normalized_objects


def build_detection_context_for_frame(
    session: Session,
    frame_at: datetime,
) -> dict[str, Any] | None:
    payload = session.latest_inference_payload
    inference_at = session.latest_inference_at
    if payload is None or inference_at is None:
        return {
            "inference_timestamp": None,
            "server_inference_received_at": None,
            "age_ms": None,
            "is_stale": True,
            "scene_summary": None,
            "guidance_text": None,
            "objects": [],
            "metrics": {
                "num_detections": 0,
                "num_with_confidence": 0,
                "avg_confidence": None,
                "max_confidence": None,
                "min_confidence": None,
            },
        }

    objects = normalize_inference_objects(payload)
    confidences = [
        item["confidence"] for item in objects if isinstance(item.get("confidence"), float)
    ]
    age_ms = max(0, int(round((frame_at - inference_at).total_seconds() * 1000)))

    metrics = {
        "num_detections": len(objects),
        "num_with_confidence": len(confidences),
        "avg_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "max_confidence": round(max(confidences), 4) if confidences else None,
        "min_confidence": round(min(confidences), 4) if confidences else None,
    }

    return {
        "inference_timestamp": payload.get("timestamp"),
        "server_inference_received_at": inference_at.isoformat(),
        "age_ms": age_ms,
        "is_stale": age_ms > MAX_INFERENCE_SAMPLE_AGE_MS,
        "scene_summary": payload.get("scene_summary"),
        "guidance_text": payload.get("guidance_text"),
        "objects": objects,
        "metrics": metrics,
    }


async def close_session(session_id: str) -> None:
    session = sessions.pop(session_id, None)
    if not session:
        return

    closed_at = datetime.now(timezone.utc)

    for task in [session.frame_task, session.mock_task]:
        if task:
            task.cancel()

    await session.peer_connection.close()
    session.updated_at = closed_at
    write_session_manifest(session_id, session, closed_at=closed_at)


async def session_gc(timeout_seconds: int = 60) -> None:
    while True:
        await asyncio.sleep(10)
        now = datetime.now(timezone.utc)
        stale_sessions: list[str] = []

        for session_id, session in sessions.items():
            if (now - session.updated_at).total_seconds() > timeout_seconds:
                stale_sessions.append(session_id)

        for session_id in stale_sessions:
            await close_session(session_id)


async def wait_for_ice_gathering(
    peer_connection: RTCPeerConnection, timeout_seconds: float = 3.0
) -> None:
    if peer_connection.iceGatheringState == "complete":
        return

    async def _wait() -> None:
        while peer_connection.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(_wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return


def sanitize_artifact_component(value: str) -> str:
    safe_chars: list[str] = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "session"


def create_session_artifact(session_id: str, started_at: datetime) -> tuple[str | None, Path | None, Path | None, str | None]:
    safe_session_id = sanitize_artifact_component(session_id)
    started_fragment = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    artifact_id = f"{started_fragment}--{safe_session_id}"
    artifact_dir = SESSION_ARTIFACTS_ROOT / artifact_id
    artifact_manifest_path = artifact_dir / SESSION_MANIFEST_FILENAME

    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)

    
    except FileExistsError:
        artifact_id = f"{artifact_id}--{uuid.uuid4().hex[:8]}"
        artifact_dir = SESSION_ARTIFACTS_ROOT / artifact_id
        artifact_manifest_path = artifact_dir / SESSION_MANIFEST_FILENAME
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
        except Exception as error:
            return None, None, None, f"artifact init failed: {error}"
    except Exception as error:
        return None, None, None, f"artifact init failed: {error}"

    return artifact_id, artifact_dir, artifact_manifest_path, None


def get_group_dir(artifact_dir: Path, started_at: datetime, frame_at: datetime) -> Path:
    elapsed_seconds = max(0.0, (frame_at - started_at).total_seconds())
    group_number = int(elapsed_seconds // GROUP_SESSION_TIME_CUT_SECONDS) + 1
    group_dir = artifact_dir / f"group-{group_number:03d}"
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def coerce_bbox_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None

    parsed: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or not math.isfinite(item):
            return None
        parsed.append(float(item))
    return parsed


def clamp_pixel(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def bbox_to_xyxy_pixels(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    if image_width <= 0 or image_height <= 0:
        return None

    x, y, width, height = bbox

    if all(0.0 <= value <= 1.0 for value in bbox):
        left = int(round(x * image_width))
        top = int(round(y * image_height))
        right = int(round((x + width) * image_width))
        bottom = int(round((y + height) * image_height))
    else:
        if width > x and height > y:
            left = int(round(x))
            top = int(round(y))
            right = int(round(width))
            bottom = int(round(height))
        else:
            left = int(round(x))
            top = int(round(y))
            right = int(round(x + width))
            bottom = int(round(y + height))

    max_x = max(0, image_width - 1)
    max_y = max(0, image_height - 1)
    left = clamp_pixel(left, 0, max_x)
    top = clamp_pixel(top, 0, max_y)
    right = clamp_pixel(right, 0, max_x)
    bottom = clamp_pixel(bottom, 0, max_y)

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def select_primary_detected_object(
    detected_objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not detected_objects:
        return None

    best_object = detected_objects[0]
    best_confidence = (
        best_object.get("confidence")
        if isinstance(best_object.get("confidence"), float)
        else -1.0
    )
    for candidate in detected_objects[1:]:
        confidence = candidate.get("confidence")
        if isinstance(confidence, float) and confidence > best_confidence:
            best_object = candidate
            best_confidence = confidence

    return best_object


def render_detection_overlay(
    frame_jpeg: bytes,
    detection_context: dict[str, Any] | None,
) -> tuple[bytes, list[dict[str, Any]], str | None]:
    if not isinstance(detection_context, dict):
        return frame_jpeg, [], None

    raw_objects = detection_context.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        return frame_jpeg, [], None

    parsed_objects: list[dict[str, Any]] = []
    for raw_obj in raw_objects:
        if not isinstance(raw_obj, dict):
            continue
        label = raw_obj.get("label")
        if not isinstance(label, str):
            continue
        confidence = raw_obj.get("confidence")
        confidence_value = None
        if isinstance(confidence, (int, float)) and math.isfinite(confidence):
            confidence_value = float(confidence)
        parsed_objects.append(
            {
                "label": label,
                "confidence": confidence_value,
                "bbox": coerce_bbox_list(raw_obj.get("bbox")),
            }
        )

    if not parsed_objects:
        return frame_jpeg, [], None

    try:
        image = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
    except Exception as error:
        return frame_jpeg, parsed_objects, f"overlay decode failed: {error}"

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    image_width, image_height = image.size
    draw_color = (0, 220, 90)
    drawn_any_box = False

    for obj in parsed_objects:
        bbox = obj.get("bbox")
        if not isinstance(bbox, list):
            continue

        pixel_box = bbox_to_xyxy_pixels(bbox, image_width=image_width, image_height=image_height)
        if pixel_box is None:
            continue

        x1, y1, x2, y2 = pixel_box
        draw.rectangle((x1, y1, x2, y2), outline=draw_color, width=3)
        confidence = obj.get("confidence")
        if isinstance(confidence, float):
            text = f"{obj['label']} {confidence:.2f}"
        else:
            text = obj["label"]

        try:
            text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), text, font=font)
            text_width = max(1, text_right - text_left)
            text_height = max(1, text_bottom - text_top)
        except Exception:
            text_width = max(1, int(len(text) * 7))
            text_height = 12

        text_x = x1
        text_y = y1 - text_height - 6
        if text_y < 0:
            text_y = min(image_height - text_height - 1, y1 + 3)

        bg_right = min(image_width - 1, text_x + text_width + 6)
        bg_bottom = min(image_height - 1, text_y + text_height + 4)
        draw.rectangle((text_x, text_y, bg_right, bg_bottom), fill=draw_color)
        draw.text((text_x + 3, text_y + 1), text, fill=(0, 0, 0), font=font)
        drawn_any_box = True

    if not drawn_any_box:
        return frame_jpeg, parsed_objects, None

    try:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=SNAPSHOT_JPEG_QUALITY)
        return buffer.getvalue(), parsed_objects, None
    except Exception as error:
        return frame_jpeg, parsed_objects, f"overlay encode failed: {error}"


def persist_processed_frame(
    session: Session,
    frame_jpeg: bytes,
    frame_at: datetime,
    directional_context: dict[str, Any] | None,
    detection_context: dict[str, Any] | None,
) -> None:
    if session.artifact_dir is None:
        return

    group_dir = get_group_dir(
        artifact_dir=session.artifact_dir,
        started_at=session.started_at,
        frame_at=frame_at,
    )

    frame_name = (
        f"frame-{session.processed_frames:06d}-"
        f"{frame_at.strftime('%Y%m%dT%H%M%S%fZ')}.jpg"
    )
    frame_path = group_dir / frame_name
    metadata_path = frame_path.with_suffix(".json")
    elapsed_seconds = max(0.0, (frame_at - session.started_at).total_seconds())
    group_number = int(elapsed_seconds // GROUP_SESSION_TIME_CUT_SECONDS) + 1
    rendered_frame_jpeg, detected_objects, overlay_error = render_detection_overlay(
        frame_jpeg=frame_jpeg,
        detection_context=detection_context,
    )
    object_detected = select_primary_detected_object(detected_objects)
    frame_metadata = {
        "frame_id": session.processed_frames,
        "frame_name": frame_name,
        "frame_index": session.processed_frames,
        "frame_at": frame_at.isoformat(),
        "group_id": group_number,
        "group_dir": group_dir.name,
        "directional": directional_context,
    }

    try:
        frame_path.write_bytes(rendered_frame_jpeg)
        metadata_path.write_text(json.dumps(frame_metadata, indent=2) + "\n")
        session.dumped_frames += 1
        session.last_dump_error = None

    except Exception as error:
        session.dump_errors += 1
        session.last_dump_error = f"frame dump failed: {error}"


def build_session_manifest(
    session_id: str,
    session: Session,
    closed_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "artifact_id": session.artifact_id,
        "artifact_dir": str(session.artifact_dir) if session.artifact_dir else None,
        "started_at": session.started_at.isoformat(),
        "closed_at": closed_at.isoformat() if closed_at else None,
        "analysis_target_fps": session.analysis_target_fps,
        "total_frames": session.total_frames,
        "processed_frames": session.processed_frames,
        "dropped_frames": session.dropped_frames,
        "incoming_fps": session.incoming_fps,
        "processed_fps": session.processed_fps,
        "dumped_frames": session.dumped_frames,
        "dump_errors": session.dump_errors,
        "last_dump_error": session.last_dump_error,
        "directional_samples_received": session.directional_samples_received,
        "directional_messages_ignored": session.directional_messages_ignored,
        "directional_parse_errors": session.directional_parse_errors,
        "latest_directional_received_at": (
            session.latest_directional_received_at.isoformat()
            if session.latest_directional_received_at
            else None
        ),
        "latest_directional_client_timestamp_ms": (
            session.latest_directional_client_timestamp_ms
        ),
        "latest_processed_directional": session.latest_processed_directional,
        "inference_messages_emitted": session.inference_messages_emitted,
        "latest_inference_at": (
            session.latest_inference_at.isoformat()
            if session.latest_inference_at
            else None
        ),
        "latest_processed_detections": session.latest_processed_detections,
        "latest_jpeg_at": (
            session.latest_jpeg_at.isoformat() if session.latest_jpeg_at else None
        ),
        "last_frame_at": session.last_frame_at.isoformat()
        if session.last_frame_at
        else None,
        "updated_at": session.updated_at.isoformat(),
    }


def write_session_manifest(
    session_id: str,
    session: Session,
    closed_at: datetime | None = None,
) -> None:
    if session.artifact_manifest_path is None:
        return

    payload = build_session_manifest(
        session_id=session_id,
        session=session,
        closed_at=closed_at,
    )

    try:
        session.artifact_manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception as error:
        session.dump_errors += 1
        session.last_dump_error = f"manifest write failed: {error}"


def load_session_history(limit: int = 100) -> list[dict[str, Any]]:
    if not SESSION_ARTIFACTS_ROOT.exists():
        return []

    history: list[dict[str, Any]] = []
    artifact_dirs = sorted(
        [entry for entry in SESSION_ARTIFACTS_ROOT.iterdir() if entry.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    for artifact_dir in artifact_dirs[:limit]:
        manifest_path = artifact_dir / SESSION_MANIFEST_FILENAME
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text())
                if isinstance(payload, dict):
                    history.append(payload)
                    continue
            except Exception as error:
                history.append(
                    {
                        "artifact_id": artifact_dir.name,
                        "artifact_dir": str(artifact_dir),
                        "manifest_error": str(error),
                    }
                )
                continue

        frame_files = [
            entry for entry in artifact_dir.iterdir() if entry.name.endswith(".jpg")
        ]
        history.append(
            {
                "artifact_id": artifact_dir.name,
                "artifact_dir": str(artifact_dir),
                "dumped_frames": len(frame_files),
            }
        )

    return history
def frame_to_jpeg(
    frame: Any, jpeg_quality: int = SNAPSHOT_JPEG_QUALITY
) -> tuple[bytes | None, str | None]:
    errors: list[str] = []

    try:
        image = frame.to_image()
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"to_image failed: {error}")

    try:
        rgb = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"rgb24 fallback failed: {error}")

    try:
        gray = frame.to_ndarray(format="gray")
        image = Image.fromarray(gray, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"gray fallback failed: {error}")

    return None, " | ".join(errors)
