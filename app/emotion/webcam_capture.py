"""
BAYMAX AI – Webcam Capture
============================
Asynchronous webcam frame producer using OpenCV.
Runs in a dedicated background thread and exposes frames via a queue.

Usage:
    from app.emotion.webcam_capture import WebcamCapture
    with WebcamCapture() as cam:
        for frame in cam.frames():
            process(frame)
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, Optional

import cv2
import numpy as np

from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class VideoFrame:
    """
    A captured video frame.

    Attributes:
        image:      BGR numpy array (H x W x 3).
        timestamp:  Unix timestamp of capture.
        frame_id:   Monotonically increasing frame counter.
    """
    image: np.ndarray
    timestamp: float
    frame_id: int

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    def to_rgb(self) -> np.ndarray:
        """Return frame as RGB array."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)


class WebcamCapture:
    """
    Background webcam frame capturer.

    Frames are placed in a thread-safe queue. If the queue is full,
    the oldest frame is dropped (non-blocking).

    Attributes:
        device_index:   OpenCV camera index (0 = primary webcam).
        target_fps:     Target capture frame rate.
        queue_maxsize:  Maximum frames to buffer.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        target_fps: Optional[int] = None,
        queue_maxsize: int = 10,
    ) -> None:
        from config import settings

        self.device_index = device_index if device_index is not None else settings.WEBCAM_INDEX
        self.target_fps = target_fps or settings.WEBCAM_FPS
        self.queue_maxsize = queue_maxsize
        self._frame_queue: queue.Queue[VideoFrame] = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_counter = 0
        self._cap: Optional[cv2.VideoCapture] = None

        log.info(
            "WebcamCapture configured | device={} fps={}",
            self.device_index,
            self.target_fps,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open webcam and start background capture thread."""
        if self._thread and self._thread.is_alive():
            log.warning("WebcamCapture already running")
            return

        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam device {self.device_index}. "
                "Check camera connection and device index."
            )

        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="webcam-capture",
        )
        self._thread.start()
        log.info("WebcamCapture started | device={}", self.device_index)

    def stop(self) -> None:
        """Stop capture thread and release webcam."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        log.info("WebcamCapture stopped")

    def get_frame(self, timeout: float = 0.5) -> Optional[VideoFrame]:
        """
        Get the next frame from the queue (blocking).

        Args:
            timeout: Seconds to wait for a frame.

        Returns:
            VideoFrame or None if timeout expired.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[VideoFrame]:
        """
        Drain the queue and return only the most recent frame.

        Returns:
            Most recent VideoFrame or None if queue is empty.
        """
        latest: Optional[VideoFrame] = None
        while True:
            try:
                latest = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def frames(
        self,
        timeout: float = 0.5,
    ) -> Generator[VideoFrame, None, None]:
        """
        Generator that yields frames from the capture queue.

        Args:
            timeout: Wait timeout per frame.

        Yields:
            VideoFrame instances.
        """
        if not self._thread or not self._thread.is_alive():
            self.start()
        while not self._stop_event.is_set():
            frame = self.get_frame(timeout=timeout)
            if frame is not None:
                yield frame

    def capture_single_frame(self) -> Optional[VideoFrame]:
        """
        Open the webcam, capture one frame, and immediately close it.
        Useful for API endpoint single-shot emotion detection.

        Returns:
            VideoFrame or None.
        """
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            log.error("Cannot open webcam device {}", self.device_index)
            return None
        try:
            ret, frame = cap.read()
            if ret:
                return VideoFrame(
                    image=frame,
                    timestamp=time.time(),
                    frame_id=0,
                )
            return None
        finally:
            cap.release()

    # ── Private Methods ───────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background loop: read frames from OpenCV at target FPS."""
        interval = 1.0 / self.target_fps
        try:
            while not self._stop_event.is_set() and self._cap and self._cap.isOpened():
                t_start = time.time()
                ret, raw_frame = self._cap.read()

                if not ret:
                    log.warning("Webcam read failed – retrying...")
                    time.sleep(0.1)
                    continue

                self._frame_counter += 1
                frame = VideoFrame(
                    image=raw_frame,
                    timestamp=time.time(),
                    frame_id=self._frame_counter,
                )

                # Drop old frames if queue full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put_nowait(frame)

                # Throttle to target FPS
                elapsed = time.time() - t_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as exc:
            log.error("Webcam capture error: {}", exc)

    def __enter__(self) -> "WebcamCapture":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
