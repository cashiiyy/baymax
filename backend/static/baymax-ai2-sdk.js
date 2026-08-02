/**
 * BAYMAX AI Engine 2 — JavaScript SDK
 * =====================================
 * Drop-in SDK for AI Engine 1's webpage to communicate with AI Engine 2.
 *
 * Usage (in AI Engine 1's HTML):
 *   <script>window.BAYMAX_AI2_URL = 'http://100.86.102.107:8001';</script>
 *   <script src="/static/baymax-ai2-sdk.js"></script>
 *   <script>
 *     const ai2 = new BaymaxAI2({ debug: true });
 *     await ai2.tts("Hello, I am BAYMAX.");
 *     const result = await ai2.ocr(fileInput.files[0]);
 *   </script>
 *
 * Alternatively, configure via window.BAYMAX_AI2_URL before loading the script.
 */

(function (global) {
  "use strict";

  // ── Default configuration ───────────────────────────────────────────────────
  const DEFAULT_BASE_URL =
    global.BAYMAX_AI2_URL || "http://100.79.169.64:8001";

  // ── Wake word defaults ──────────────────────────────────────────────────────
  const DEFAULT_WAKE_WORD = "hey baymax";
  const WAKE_WORD_CHUNK_MS = 3000; // record 3s chunks for wake word detection

  // ── SDK class ───────────────────────────────────────────────────────────────
  class BaymaxAI2 {
    /**
     * @param {object} options
     * @param {string} [options.baseUrl]    - Base URL for AI Engine 2 (e.g. http://100.86.102.107:8001)
     * @param {string} [options.wakeWord]   - Wake word phrase to listen for (default: "hey baymax")
     * @param {boolean} [options.debug]     - If true, log debug info to console
     */
    constructor(options = {}) {
      this.baseUrl = (options.baseUrl || DEFAULT_BASE_URL).replace(/\/$/, "");
      this.wakeWord = (options.wakeWord || DEFAULT_WAKE_WORD).toLowerCase();
      this.debug = options.debug || false;

      this._wakeWordActive = false;
      this._mediaRecorder = null;
      this._audioContext = null;
      this._voices = null; // cached voice list

      this._log("BaymaxAI2 SDK initialised", { baseUrl: this.baseUrl });
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    _log(...args) {
      if (this.debug) console.log("[BaymaxAI2]", ...args);
    }

    _err(...args) {
      console.error("[BaymaxAI2]", ...args);
    }

    /**
     * Generic fetch wrapper. Returns parsed JSON or throws.
     */
    async _fetch(path, init = {}) {
      const url = `${this.baseUrl}${path}`;
      this._log("fetch", init.method || "GET", url);
      const response = await fetch(url, init);
      if (!response.ok) {
        let detail = "";
        try {
          const err = await response.json();
          detail = err.message || JSON.stringify(err);
        } catch (_) {
          detail = await response.text();
        }
        throw new Error(`AI Engine 2 error [${response.status}]: ${detail}`);
      }
      return response;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Text-to-Speech
     * Synthesizes `text` using the XTTS v2 voice model running on AI Engine 2.
     * Automatically plays the result through the browser's audio system.
     *
     * @param {string} text          - Text to synthesize
     * @param {string} [voice]       - Voice profile name (default: "default")
     * @param {string} [language]    - Language code (default: "en")
     * @returns {Promise<HTMLAudioElement>} - The audio element that is playing
     */
    async tts(text, voice = "default", language = "en") {
      const body = JSON.stringify({
        text,
        voice,
        language,
        stream: false,
        format: "wav",
      });

      const response = await this._fetch("/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });

      const data = await response.json();
      if (!data.success || !data.audio_base64) {
        throw new Error("TTS returned no audio data");
      }

      // Decode base64 → ArrayBuffer → Blob → audio element
      const binaryStr = atob(data.audio_base64);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "audio/wav" });
      const audioUrl = URL.createObjectURL(blob);

      const audio = new Audio(audioUrl);
      audio.onended = () => URL.revokeObjectURL(audioUrl);
      await audio.play();

      this._log("tts played", {
        voice,
        duration: data.duration_seconds,
        voice_used: data.voice_used,
      });
      return audio;
    }

    /**
     * OCR — Medical Document Analysis
     * Uploads an image or PDF file and returns structured OCR results.
     *
     * @param {File|Blob} file    - Image or PDF file
     * @param {string} [lang]     - Tesseract language code (default: "eng")
     * @returns {Promise<object>} - OCR result with raw_text, extracted_fields, document_type, etc.
     */
    async ocr(file, lang = "eng") {
      const form = new FormData();
      form.append("file", file);
      form.append("lang", lang);
      form.append("psm", "3");
      form.append("oem", "3");

      const response = await this._fetch("/ocr", {
        method: "POST",
        body: form,
      });

      const data = await response.json();
      this._log("ocr complete", {
        doc_type: data.document_type,
        confidence: data.overall_confidence,
        fields: data.extracted_fields?.length,
      });
      return data;
    }

    /**
     * Vision Analysis — Emotion, Face, & Presence Detection
     * Analyses an image for face detection, emotion context, and person presence.
     *
     * @param {File|Blob} imageFile    - Image file (JPG, PNG, etc.)
     * @returns {Promise<object>}      - Vision result with faces, emotions, roi_observations, etc.
     */
    async vision(imageFile) {
      const form = new FormData();
      form.append("file", imageFile);
      form.append("face_detection", "true");
      form.append("emotion_context", "true");
      form.append("roi_detection", "true");
      form.append("mediapipe", "true");
      form.append("person_detection", "true");

      const response = await this._fetch("/vision", {
        method: "POST",
        body: form,
      });

      const data = await response.json();
      this._log("vision complete", {
        faces: data.faces?.length,
        person: data.person_detected,
        lighting: data.lighting_assessment,
      });
      return data;
    }

    /**
     * Speech-to-Text — Transcription
     * Transcribes an audio file using Faster-Whisper.
     *
     * @param {File|Blob} audioFile    - Audio file (WAV, MP3, OGG, etc.)
     * @param {string} [language]      - Force language (empty = auto-detect)
     * @returns {Promise<object>}      - Transcription with text, segments, language, etc.
     */
    async transcribe(audioFile, language = "") {
      const form = new FormData();
      form.append("file", audioFile);
      form.append("language", language);
      form.append("vad_filter", "true");
      form.append("word_timestamps", "true");

      const response = await this._fetch("/transcribe", {
        method: "POST",
        body: form,
      });

      const data = await response.json();
      this._log("transcribe complete", {
        language: data.language,
        words: data.word_count,
        text_preview: data.text?.slice(0, 60),
      });
      return data;
    }

    /**
     * Combined Analysis — Run multiple tasks on a single file in one call.
     * Uses the POST /analyse endpoint for parallel server-side processing.
     *
     * @param {File|Blob} file         - Image, audio, or document file
     * @param {string[]} tasks         - Array of tasks: ["ocr", "vision", "transcribe"]
     * @returns {Promise<object>}      - Combined result from all tasks
     */
    async analyse(file, tasks = ["ocr"]) {
      const form = new FormData();
      form.append("file", file);
      form.append("tasks", tasks.join(","));

      const response = await this._fetch("/analyse", {
        method: "POST",
        body: form,
      });

      const data = await response.json();
      this._log("analyse complete", {
        tasks_succeeded: data.tasks_succeeded,
        tasks_failed: data.tasks_failed,
        processing_ms: data.processing_time_ms,
      });
      return data;
    }

    /**
     * List available TTS voice profiles registered on AI Engine 2.
     * @returns {Promise<object[]>}  - Array of { name, language, description, is_cloned }
     */
    async getVoices() {
      if (this._voices) return this._voices;
      const response = await this._fetch("/tts/voices");
      const data = await response.json();
      this._voices = data.voices || [];
      this._log("voices loaded", this._voices.length);
      return this._voices;
    }

    /**
     * Health check — returns true if AI Engine 2 is reachable and healthy.
     * @returns {Promise<boolean>}
     */
    async isHealthy() {
      try {
        const response = await this._fetch("/health");
        const data = await response.json();
        return data.status === "ok" || data.status === "healthy";
      } catch (_) {
        return false;
      }
    }

    // ── Wake Word Detection ───────────────────────────────────────────────────

    /**
     * Start listening for the wake word (default: "hey baymax").
     * When detected, calls `callback(transcriptionResult)`.
     *
     * Uses the browser's MediaRecorder API + AI Engine 2's /transcribe endpoint
     * to check short audio chunks for the wake word phrase.
     *
     * @param {function} callback          - Called when wake word is detected
     * @param {object}   [options]
     * @param {string}   [options.wakeWord]  - Override the wake word
     * @param {number}   [options.chunkMs]   - Recording chunk duration in ms (default: 3000)
     */
    async startWakeWordDetection(callback, options = {}) {
      if (this._wakeWordActive) {
        this._log("Wake word already active");
        return;
      }

      const wakeWord = (options.wakeWord || this.wakeWord).toLowerCase();
      const chunkMs = options.chunkMs || WAKE_WORD_CHUNK_MS;

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      } catch (err) {
        this._err("Microphone access denied:", err);
        throw new Error("Microphone permission required for wake word detection.");
      }

      this._wakeWordActive = true;
      this._log("Wake word detection started, listening for:", wakeWord);

      const checkChunk = () => {
        if (!this._wakeWordActive) return;

        const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        const chunks = [];

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data);
        };

        recorder.onstop = async () => {
          if (!this._wakeWordActive) return;
          try {
            const blob = new Blob(chunks, { type: "audio/webm" });
            const result = await this.transcribe(blob);
            const transcript = (result.text || "").toLowerCase();
            this._log("Wake word chunk transcript:", transcript);

            if (result.has_speech && transcript.includes(wakeWord)) {
              this._log("🔔 Wake word detected!");
              callback(result);
            }
          } catch (err) {
            this._err("Wake word chunk transcription failed:", err);
          }
          // Schedule next chunk
          if (this._wakeWordActive) setTimeout(checkChunk, 100);
        };

        recorder.start();
        setTimeout(() => {
          if (recorder.state === "recording") recorder.stop();
        }, chunkMs);

        this._mediaRecorder = recorder;
      };

      checkChunk();
    }

    /**
     * Stop wake word detection and release microphone.
     */
    stopWakeWordDetection() {
      this._wakeWordActive = false;
      if (this._mediaRecorder && this._mediaRecorder.state === "recording") {
        this._mediaRecorder.stop();
      }
      this._log("Wake word detection stopped");
    }

    /**
     * Capture a frame from a video element and run vision analysis on it.
     * Useful for real-time emotion/face detection from a webcam feed.
     *
     * @param {HTMLVideoElement} videoEl  - A playing <video> element
     * @returns {Promise<object>}         - Vision result
     */
    async analyseVideoFrame(videoEl) {
      const canvas = document.createElement("canvas");
      canvas.width = videoEl.videoWidth;
      canvas.height = videoEl.videoHeight;
      canvas.getContext("2d").drawImage(videoEl, 0, 0);
      return new Promise((resolve, reject) => {
        canvas.toBlob(async (blob) => {
          try {
            const result = await this.vision(blob);
            resolve(result);
          } catch (err) {
            reject(err);
          }
        }, "image/jpeg", 0.85);
      });
    }

    /**
     * Start a webcam and return the video element.
     * Pair with analyseVideoFrame() for real-time vision analysis.
     *
     * @param {HTMLVideoElement} videoEl  - Target <video> element to stream into
     * @returns {Promise<MediaStream>}    - The active media stream
     */
    async startCamera(videoEl) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      videoEl.srcObject = stream;
      await videoEl.play();
      this._log("Camera started");
      return stream;
    }
  }

  // ── Export ────────────────────────────────────────────────────────────────
  global.BaymaxAI2 = BaymaxAI2;

  // Also support ES module environments
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { BaymaxAI2 };
  }

})(typeof window !== "undefined" ? window : global);
