/**
 * B.A.Y.M.A.X. v2 — Frontend Application (AI Engine 1 Fixed)
 * ============================================================
 * Fixes applied:
 *   ✅ LLM/Chat — proper error messaging, shows offline fallback notice
 *   ✅ STT       — browser Web Speech API fallback when AE2/FFmpeg fails
 *   ✅ Wake Word — browser SpeechRecognition fallback when AE2 STT fails
 *   ✅ TTS       — AE2 → backend /tts local route → browser fallback chain
 *   ✅ Vision    — JPEG pre-processing, better error handling
 *   ✅ OCR       — better error messaging with diagnostic info
 *   ✅ Voice Toggle — proper state management & label update
 *   ✅ Icon      — handled in HTML
 */

"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────
const API_BASE = "";          // Backend is same origin
const AE2_HEALTH_FAST_MS = 5_000;
const AE2_HEALTH_SLOW_MS = 30_000;

// ── Global State ──────────────────────────────────────────────────────────────
let ai2 = null;
let ae2Online = false;
let ae2PollCount = 0;

let isRecording = false;
let mediaRecorder = null;
let micChunks = [];

let cameraStream = null;
let _visionInterval = null;   // auto-analysis timer handle
let wakeWordActive = false;
let wakeWordRecognition = null;   // Browser SpeechRecognition for wake word

// ── Voice State ───────────────────────────────────────────────────────────────
let voiceEnabled = true;
let currentAudioElement = null;

function stopSpeaking() {
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
    }
    if (currentAudioElement) {
        try {
            currentAudioElement.pause();
            currentAudioElement.currentTime = 0;
        } catch (_) { }
        currentAudioElement = null;
    }
}

function onVoiceToggleChange(checked) {
    voiceEnabled = checked;
    const label = document.getElementById("voiceLabel");
    if (label) label.textContent = checked ? "🔊 " : "🔇 ";
    if (!checked) {
        stopSpeaking();
    }
}

// ── SDK Initialisation ────────────────────────────────────────────────────────
async function initSDK() {
    if (typeof BaymaxAI2 === "undefined") {
        console.warn("[BAYMAX] BaymaxAI2 SDK not loaded — multimodal features disabled.");
        setAE2Status(false, "SDK missing", "#ef4444");
        return;
    }

    ai2 = new BaymaxAI2({
        baseUrl: window.BAYMAX_AI2_URL || "http://100.79.169.64:8001",
        wakeWord: "hey baymax",
        debug: false,
    });

    await checkAE2Health();

    const fastTimer = setInterval(async () => {
        ae2PollCount++;
        await checkAE2Health();
        if (ae2PollCount >= 12) {
            clearInterval(fastTimer);
            setInterval(checkAE2Health, AE2_HEALTH_SLOW_MS);
        }
    }, AE2_HEALTH_FAST_MS);
}

// ── AE2 Health ────────────────────────────────────────────────────────────────
async function checkAE2Health() {
    if (!ai2) return;
    try {
        const r = await fetch(`${API_BASE}/proxy/ae2-health`);
        const data = await r.json();

        const online = data.status === "ok" ||
            data.status === "healthy" ||
            data.status === "degraded";

        ae2Online = online && data.status !== "offline";

        if (data.status === "ok" || data.status === "healthy") {
            setAE2Status(true, "AE2 Ready ✓", "#22c55e");
        } else if (data.status === "degraded") {
            setAE2Status(true, "AE2 Degraded ⚠", "#f59e0b");
            ae2Online = true;
        } else {
            setAE2Status(false, "AE2 Offline", "#ef4444");
        }

        showAE2Components(data);
    } catch (_) {
        ae2Online = false;
        setAE2Status(false, "AE2 Offline", "#ef4444");
    }
}

function setAE2Status(online, label, colour) {
    const dot = document.getElementById("ae2StatusDot");
    const text = document.getElementById("ae2StatusText");
    if (!dot || !text) return;
    dot.style.background = colour;
    dot.style.boxShadow = `0 0 0 3px ${colour}33`;
    text.textContent = label;
}

function showAE2Components(data) {
    const panel = document.getElementById("ae2ComponentsPanel");
    const container = document.getElementById("ae2Components");
    if (!panel || !container) return;

    let rows = "";
    if (Array.isArray(data.models) && data.models.length) {
        rows = data.models.map(m => {
            const ok = m.loaded === true;
            const colour = ok ? "#22c55e" : "#f59e0b";
            const label = ok ? `✓ loaded (${m.device})` : "⚠ not loaded";
            return `<div style="display:flex;justify-content:space-between;padding:2px 0;gap:8px;">
                        <span>${m.name}</span>
                        <span style="color:${colour};font-weight:600;">${label}</span>
                    </div>`;
        }).join("");

        const bk = data.backend_reachable;
        rows += `<div style="display:flex;justify-content:space-between;padding:2px 0;">
                     <span>backend link</span>
                     <span style="color:${bk ? '#22c55e' : '#ef4444'};font-weight:600;">${bk ? '✓ connected' : '✗ not connected'}</span>
                 </div>`;

        if (data.uptime_seconds != null) {
            const up = Math.floor(data.uptime_seconds);
            rows += `<div style="color:var(--text-muted);font-size:0.7rem;margin-top:4px;">
                         Uptime: ${up}s &nbsp;|&nbsp; v${data.version || "?"}
                     </div>`;
        }
    } else if (data.components && typeof data.components === "object") {
        rows = Object.entries(data.components).map(([name, status]) => {
            const ok = status === "ready" || status === "ok";
            const colour = ok ? "#22c55e" : "#f59e0b";
            return `<div style="display:flex;justify-content:space-between;padding:2px 0;gap:8px;">
                        <span>${name}</span>
                        <span style="color:${colour};font-weight:600;">${ok ? '✓ ready' : '⚠ ' + status}</span>
                    </div>`;
        }).join("");
    }

    if (rows) {
        container.innerHTML = rows;
        panel.style.display = "block";
    }
}

// ── Text-to-Speech (AE2 → Backend local → Browser fallback) ──────────────────
async function speakText(text) {
    const toggle = document.getElementById("voiceToggle");
    if (!toggle?.checked || !voiceEnabled) return;

    // Filter out standard disclaimer text from being read
    const clean = text
        .split(/disclaimer:/i)[0]
        .split(/educational notice:/i)[0]
        .replace(/[*#_`>|🚨🩹🫀🦟🩺🐍🌡️🫁📋💡🔥🐍😐🔲💡📋👤]/gu, "")
        .replace(/<[^>]*>/g, "")
        .replace(/\n+/g, " ")
        .trim();
    if (!clean || clean.length < 2) return;

    // 1. Try AE2 TTS
    if (ae2Online && ai2) {
        try {
            const audio = await ai2.tts(clean);
            if (audio) {
                if (currentAudioElement) {
                    currentAudioElement.pause();
                }
                currentAudioElement = audio;
            }
            return;
        } catch (err) {
            console.warn("[BAYMAX] AE2 TTS failed:", err.message);
        }
    }

    // 2. Try backend local TTS route (uses XTTS engine)
    try {
        const res = await fetch(`${API_BASE}/tts`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: clean, voice: "default", language: "en" }),
        });
        if (res.ok) {
            const data = await res.json();
            if (data.audio_base64) {
                if (currentAudioElement) {
                    currentAudioElement.pause();
                    currentAudioElement = null;
                }
                const binaryStr = atob(data.audio_base64);
                const bytes = new Uint8Array(binaryStr.length);
                for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
                const blob = new Blob([bytes], { type: "audio/mp3" });
                const audioUrl = URL.createObjectURL(blob);
                const audio = new Audio(audioUrl);
                audio.onended = () => URL.revokeObjectURL(audioUrl);
                currentAudioElement = audio; // Keep track of local fallback audio too!
                await audio.play();
                return;
            }
        }
    } catch (err) {
        console.warn("[BAYMAX] Backend TTS failed:", err.message);
    }

    // 3. Browser Web Speech API fallback
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(clean.slice(0, 500));
        utterance.rate = 0.95;
        utterance.pitch = 0.9;
        utterance.volume = 1.0;
        // Try to pick a natural-sounding voice
        const voices = window.speechSynthesis.getVoices();
        const preferred = voices.find(v =>
            v.name.includes("Google") || v.name.includes("Natural") || v.name.includes("Neural")
        ) || voices.find(v => v.lang.startsWith("en"));
        if (preferred) utterance.voice = preferred;
        window.speechSynthesis.speak(utterance);
    }
}

// ── Mic Recording (AE2 Whisper STT → Browser Web Speech fallback) ─────────────
async function toggleMicRecording() {
    stopSpeaking();
    if (isRecording) { stopMicRecording(); return; }
    await startMicRecording();
}

let audioContext = null;
let audioAnalyser = null;
let audioStream = null;
let silenceTimer = null;

async function startMicRecording() {
    stopSpeaking();
    if (!navigator.mediaDevices?.getUserMedia) {
        // Fallback directly to browser speech recognition
        startBrowserSTT();
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        audioStream = stream;
        micChunks = [];
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) micChunks.push(e.data); };
        mediaRecorder.onstop = async () => {
            cleanupAudioContext();
            await transcribeAndSubmit(new Blob(micChunks, { type: "audio/webm" }));
        };

        // Silence detection using Web Audio API
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        audioAnalyser = audioContext.createAnalyser();
        audioAnalyser.fftSize = 512;
        source.connect(audioAnalyser);

        const bufferLength = audioAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        let lastSoundTime = Date.now();
        const silenceThreshold = 10; // low volume threshold

        function detectSilence() {
            if (!isRecording) return;
            audioAnalyser.getByteFrequencyData(dataArray);

            // Calculate average volume
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                sum += dataArray[i];
            }
            const averageVolume = sum / bufferLength;

            if (averageVolume > silenceThreshold) {
                lastSoundTime = Date.now();
            } else {
                const silentDuration = Date.now() - lastSoundTime;
                if (silentDuration >= 3000) { // 3 seconds of silence
                    console.log("[BAYMAX] 3 seconds of silence detected. Stopping recording...");
                    stopMicRecording();
                    return;
                }
            }
            requestAnimationFrame(detectSilence);
        }

        mediaRecorder.start();
        isRecording = true;
        detectSilence();

        const btn = document.getElementById("micBtn");
        if (btn) { btn.classList.add("recording"); btn.title = "Recording… click to stop"; }

    } catch (err) {
        console.warn("[BAYMAX] Mic access failed, using browser STT:", err.message);
        startBrowserSTT();
    }
}

function stopMicRecording() {
    if (mediaRecorder?.state === "recording") {
        mediaRecorder.stop();
    }
    isRecording = false;
    const btn = document.getElementById("micBtn");
    if (btn) { btn.classList.remove("recording"); btn.title = "Click to Speak"; }
}

function cleanupAudioContext() {
    if (audioStream) {
        audioStream.getTracks().forEach(t => t.stop());
        audioStream = null;
    }
    if (audioContext) {
        audioContext.close().catch(() => { });
        audioContext = null;
    }
    audioAnalyser = null;
}

// Browser Web Speech API STT (fallback when AE2/MediaRecorder fails)
function startBrowserSTT() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        appendMessage("⚠️ Speech recognition is not supported in this browser. Please type your query.", "assistant");
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";
    recognition.interimResults = false;

    const btn = document.getElementById("micBtn");
    if (btn) { btn.classList.add("recording"); btn.title = "Listening… (browser STT)"; }
    isRecording = true;

    recognition.onresult = async (event) => {
        const transcript = event.results[0][0].transcript.trim();
        if (transcript) {
            appendMessage(transcript, "user");
            await submitQuery(transcript);
        }
    };

    recognition.onerror = (event) => {
        console.warn("[BAYMAX] Browser STT error:", event.error);
        if (event.error !== "no-speech") {
            appendMessage(`⚠️ Voice recognition error: ${event.error}. Please type your query.`, "assistant");
        }
    };

    recognition.onend = () => {
        isRecording = false;
        if (btn) { btn.classList.remove("recording"); btn.title = "Click to Speak"; }
    };

    recognition.start();
}

async function transcribeAndSubmit(audioBlob) {
    const loadingId = appendMessage("🎙️ Transcribing your voice via AI Engine 1…", "assistant");
    try {
        let transcriptText = "";

        // 1. Send directly to AI Engine 1 STT
        try {
            const form = new FormData();
            form.append("file", audioBlob, "audio.webm");
            const res = await fetch(`${API_BASE}/stt`, { method: "POST", body: form });
            if (res.ok) {
                const data = await res.json();
                if (data.text) {
                    transcriptText = data.text.trim();
                }
            } else {
                console.warn(`[BAYMAX] AI Engine 1 STT HTTP ${res.status}`);
            }
        } catch (ae1Err) {
            console.warn("[BAYMAX] AI Engine 1 STT failed:", ae1Err.message);
        }

        // 2. Fallback to Browser Speech Recognition if AI Engine 1 STT yielded no text
        if (!transcriptText) {
            console.warn("[BAYMAX] AI Engine 1 STT yielded no text, attempting browser STT fallback...");
            removeMessage(loadingId);
            startBrowserSTT();
            return;
        }

        removeMessage(loadingId);
        appendMessage(transcriptText, "user");
        await submitQuery(transcriptText);

    } catch (err) {
        updateMessage(loadingId, `⚠️ Transcription error. Please type your query instead.`);
        console.error("[BAYMAX] Transcription error:", err);
    }
}

// ── Chat ──────────────────────────────────────────────────────────────────────
async function sendMessage() {
    const input = document.getElementById("userInput");
    const query = input?.value?.trim();
    if (!query) return;
    input.value = "";
    appendMessage(query, "user");
    await submitQuery(query);
}

async function submitQuery(query) {
    stopSpeaking();
    const loadingId = appendMessage(
        `<span class="typing-dots"><span></span><span></span><span></span></span> BAYMAX is thinking…`,
        "assistant"
    );

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: 1, query }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        const text = data.response || data.response_text || "(No response)";

        // Detect if it's the hardcoded offline fallback
        const isOfflineFallback = text.includes("LLM backends are currently offline") ||
            text.includes("OpenRouter / Ollama");

        let html = formatMarkdown(text);
        if (isOfflineFallback) {
            html += `<br><br><small style="opacity:0.65;">ℹ️ <em>Using offline fallback — LLM API unavailable. Check OPENROUTER_API_KEY in .env.</em></small>`;
        }

        updateMessage(loadingId, html);
        await speakText(text);

    } catch (err) {
        updateMessage(loadingId,
            `⚠️ Error connecting to BAYMAX backend. Ensure the server is running.<br>
             <small style="opacity:0.7;">Detail: ${err.message}</small>`
        );
        console.error("[BAYMAX] Chat error:", err);
    }
}

function quickQuery(text) {
    const input = document.getElementById("userInput");
    if (input) input.value = text;
    sendMessage();
}

function handleKeyPress(event) {
    if (event.key === "Enter") sendMessage();
}

// ── OCR — Medical Document Scanner ───────────────────────────────────────────
async function handleOCRUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    appendMessage(`📄 Uploaded: **${file.name}**`, "user");
    const loadingId = appendMessage("🔬 Scanning document via Medical OCR…", "assistant");
    const statusEl = document.getElementById("ocrStatus");
    if (statusEl) { statusEl.style.display = "block"; statusEl.textContent = "Scanning…"; }

    // Reset file input so same file can be re-uploaded
    event.target.value = "";

    try {
        let result;

        // 1. Try AE2 OCR (via SDK)
        if (ae2Online && ai2) {
            try {
                result = await ai2.ocr(file);
            } catch (ae2Err) {
                console.warn("[BAYMAX] AE2 OCR failed:", ae2Err.message);
                result = null;
            }
        }

        // 2. Fall back to backend OCR proxy
        if (!result) {
            try {
                const form = new FormData();
                form.append("file", file);
                form.append("lang", "eng");
                const r = await fetch(`${API_BASE}/ocr`, { method: "POST", body: form });
                if (r.ok) {
                    result = await r.json();
                } else {
                    const errData = await r.json().catch(() => ({}));
                    throw new Error(errData.detail || `HTTP ${r.status}`);
                }
            } catch (proxyErr) {
                console.warn("[BAYMAX] Backend OCR proxy failed:", proxyErr.message);
                result = null;
            }
        }

        if (!result) {
            updateMessage(loadingId,
                `❌ Medical OCR failed. Both AI Engine 2 and the local backend could not process this document.<br><br>` +
                `<strong>Possible reasons:</strong><br>` +
                `• AI Engine 2 needs Tesseract OCR installed<br>` +
                `• The image format may not be supported<br>` +
                `• Try uploading a clear JPEG/PNG image of a document`
            );
            if (statusEl) { statusEl.textContent = "OCR Failed — see chat for details"; }
            return;
        }

        let html = "";
        const rawText = result?.raw_text || result?.extracted_text || "";

        if (rawText && rawText.trim().length > 2) {
            html += `<strong>📄 Extracted Text:</strong><br>${formatMarkdown(rawText)}`;

            if (result.document_type) {
                html += `<br><br><strong>📋 Document Type:</strong> ${result.document_type}`;
            }
            if (result.overall_confidence != null) {
                const pct = Math.round(result.overall_confidence * 100);
                html += ` &nbsp;|&nbsp; <strong>Confidence:</strong> ${pct}%`;
            }
            if (result.extracted_fields && Object.keys(result.extracted_fields).length > 0) {
                html += "<br><br><strong>🔍 Extracted Fields:</strong><ul>";
                Object.entries(result.extracted_fields).forEach(([key, value]) => {
                    // Format key from snake_case to Title Case
                    const titleKey = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                    html += `<li><strong>${titleKey}:</strong> ${value}</li>`;
                });
                html += "</ul>";
            }
        } else {
            html = "⚠️ No readable text could be extracted from this document. " +
                "Please ensure the image is clear and contains printed text.";
        }

        updateMessage(loadingId, html);
        if (statusEl) statusEl.textContent = "Done ✓";

        if (rawText && rawText.trim().length > 5) {
            await submitQuery(
                `I've uploaded a medical document. Here is the extracted text:\n\n${rawText}\n\nPlease analyse this and provide relevant medical information.`
            );
        }

    } catch (err) {
        updateMessage(loadingId,
            `❌ Error processing document. Please try again with a different image.<br>` +
            `<small style="opacity:0.7;">${err.message}</small>`
        );
        if (statusEl) statusEl.textContent = "Error";
        console.error("[BAYMAX] OCR error:", err);
    }
}

// ── Vision / Webcam ───────────────────────────────────────────────────────────
async function toggleCamera() {
    const videoEl = document.getElementById("webcamVideo");
    const cameraBtn = document.getElementById("cameraBtn");
    const captureBtn = document.getElementById("captureBtn");

    if (cameraStream) {
        // Stop camera + auto-analysis loop
        clearInterval(_visionInterval);
        _visionInterval = null;
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        if (videoEl) videoEl.style.display = "none";
        if (cameraBtn) cameraBtn.textContent = "📷 Start Camera";
        if (captureBtn) captureBtn.style.display = "none";
        return;
    }

    // Camera doesn't require AE2 — just needs getUserMedia
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoEl) {
            videoEl.srcObject = cameraStream;
            await videoEl.play();
            videoEl.style.display = "block";
        }
        if (cameraBtn) cameraBtn.textContent = "⏹ Stop Camera";
        // Hide manual capture button — auto-analysis handles it
        if (captureBtn) captureBtn.style.display = "none";

        // Start auto-analysis: run immediately then every 5 seconds
        await captureAndAnalyse();
        _visionInterval = setInterval(captureAndAnalyse, 5000);
    } catch (err) {
        appendMessage(`⚠️ Camera error: ${err.message}. Please allow camera access in your browser.`, "assistant");
    }
}

async function captureAndAnalyse() {
    const videoEl = document.getElementById("webcamVideo");
    const resultEl = document.getElementById("visionResult");

    if (!videoEl) return;
    if (resultEl) { resultEl.style.display = "block"; resultEl.textContent = "Analysing frame…"; }

    try {
        // Capture frame to canvas, convert to JPEG blob (avoids OpenCV format issues)
        const canvas = document.createElement("canvas");
        canvas.width = videoEl.videoWidth || 640;
        canvas.height = videoEl.videoHeight || 480;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoEl, 0, 0);

        // Convert to JPEG blob
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));

        if (!blob) {
            if (resultEl) resultEl.textContent = "⚠️ Could not capture camera frame.";
            return;
        }

        // ── Route through /proxy/vision (AI Engine 1) instead of AE2 directly ──
        // This applies the local HuggingFace emotion model override.
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const resp = await fetch("/proxy/vision", { method: "POST", body: formData });
        if (!resp.ok) {
            const txt = await resp.text().catch(() => "");
            if (resultEl) resultEl.textContent = `⚠️ Vision analysis failed (${resp.status}): ${txt.slice(0, 120)}`;
            return;
        }
        const result = await resp.json();

        if (!result) {
            if (resultEl) resultEl.textContent = "No result from vision analysis.";
            return;
        }

        const parts = [];
        if (result.person_detected != null)
            parts.push(`👤 Person: ${result.person_detected ? "detected" : "not detected"}`);
        if (result.emotions?.length)
            parts.push(`😐 Emotion: ${result.emotions[0].label} (${Math.round(result.emotions[0].confidence * 100)}%)`);
        if (result.faces?.length)
            parts.push(`🔲 Faces: ${result.faces.length} detected`);
        if (result.lighting_assessment)
            parts.push(`💡 Lighting: ${result.lighting_assessment}`);
        if (result.observations?.length)
            parts.push(`📋 ${result.observations.join("; ")}`);

        if (resultEl) resultEl.innerHTML = parts.join("<br>") || "No observations.";

    } catch (err) {
        const errMsg = err.message || "";
        let friendly = errMsg;
        if (errMsg.includes("VisionError") || errMsg.includes("OpenCV") || errMsg.includes("422")) {
            friendly = "⚠️ Vision analysis failed. AI Engine 2 has an OpenCV compatibility issue.";
        } else if (errMsg.includes("unavailable") || errMsg.includes("offline")) {
            friendly = "⚠️ AI Engine 2 is offline. Vision analysis requires AE2 to be running.";
        }
        if (resultEl) resultEl.textContent = friendly;
        console.warn("[BAYMAX] Vision error:", errMsg);
    }
}


// ── Wake Word (Browser SpeechRecognition + AI Engine 1 fallback) ─────────────
async function toggleWakeWord() {
    const btn = document.getElementById("wakeWordBtn");
    const status = document.getElementById("wakeWordStatus");
    if (status) status.style.display = "block";

    if (wakeWordActive) {
        // Stop wake word detection
        stopWakeWordDetection();
        if (btn) btn.textContent = "🎙️ Enable Wake Word";
        if (status) status.textContent = "";
        return;
    }

    // Activate Web Speech API wake word listener
    startBrowserWakeWord(btn, status);
}

async function startAE1WakeWord(btn, status) {
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch (err) {
        startBrowserWakeWord(btn, status);
        return;
    }

    wakeWordActive = true;
    if (btn) btn.textContent = "🛑 Disable Wake Word";
    if (status) status.textContent = `🎙️ Listening for "Hey Baymax" (AI Engine 1)…`;

    const checkChunk = () => {
        if (!wakeWordActive) {
            stream.getTracks().forEach(t => t.stop());
            return;
        }
        if (isRecording) {
            setTimeout(checkChunk, 1000);
            return;
        }

        const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        const chunks = [];

        recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
        recorder.onstop = async () => {
            if (!wakeWordActive) return;
            try {
                const blob = new Blob(chunks, { type: "audio/webm" });
                const form = new FormData();
                form.append("file", blob, "chunk.webm");
                form.append("wake_word", "hey baymax");

                const res = await fetch(`${API_BASE}/wakeword`, { method: "POST", body: form });
                if (res.ok) {
                    const data = await res.json();
                    if (data.wake_word_detected) {
                        if (status) status.textContent = "🔔 Wake word detected! Activating mic...";
                        stopWakeWordDetection();
                        startMicRecording();
                        return;
                    }
                }
            } catch (err) {
                console.warn("[BAYMAX] AE1 wake word chunk check error:", err);
            }
            if (wakeWordActive) setTimeout(checkChunk, 200);
        };

        recorder.start();
        setTimeout(() => {
            if (recorder.state === "recording") recorder.stop();
        }, 3000);
    };

    checkChunk();
}

function startBrowserWakeWord(btn, status) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        if (status) status.textContent = "⚠️ Wake word requires Chrome or Edge browser.";
        return;
    }

    if (wakeWordRecognition) {
        try { wakeWordRecognition.stop(); } catch (_) { }
    }

    wakeWordRecognition = new SpeechRecognition();
    wakeWordRecognition.continuous = true;
    wakeWordRecognition.lang = "en-US";
    wakeWordRecognition.interimResults = true;

    wakeWordRecognition.onresult = (event) => {
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript.toLowerCase();
            if (transcript.includes("hey baymax") || transcript.includes("baymax")) {
                if (status) status.textContent = "🔔 Wake word detected! Activating mic...";

                // Stop wake word listening temporarily so audio input isn't locked
                try { wakeWordRecognition.stop(); } catch (_) { }

                // Trigger recording
                toggleMicRecording();
                break;
            }
        }
    };

    wakeWordRecognition.onerror = (event) => {
        console.warn("[BAYMAX] Wake word error:", event.error);
        if (event.error === "not-allowed") {
            if (status) status.textContent = "⚠️ Microphone access denied. Please allow mic permissions.";
            stopWakeWordDetection();
            if (btn) btn.textContent = "🎙️ Enable Wake Word";
        }
    };

    wakeWordRecognition.onend = () => {
        // Restart continuously if still active and not recording
        if (wakeWordActive && !isRecording) {
            try { wakeWordRecognition.start(); } catch (_) { }
        }
    };

    try {
        wakeWordRecognition.start();
        wakeWordActive = true;
        if (btn) btn.textContent = "🛑 Disable Wake Word";
        if (status) status.textContent = `🎙️ Listening for "Hey Baymax"...`;
    } catch (err) {
        if (status) status.textContent = `⚠️ Could not start wake word: ${err.message}`;
    }
}

function stopWakeWordDetection() {
    wakeWordActive = false;

    // Stop AE2 wake word if active
    if (ai2) {
        try { ai2.stopWakeWordDetection(); } catch (_) { }
    }

    // Stop browser wake word if active
    if (wakeWordRecognition) {
        try { wakeWordRecognition.stop(); } catch (_) { }
        wakeWordRecognition = null;
    }
}

// ── Message Helpers ───────────────────────────────────────────────────────────
function appendMessage(text, role) {
    const chat = document.getElementById("chatMessages");
    const wrapper = document.createElement("div");
    const msgId = "msg-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6);
    wrapper.className = `msg ${role}`;

    if (role === "assistant") {
        wrapper.innerHTML = `
            <img src="/static/icon.png" class="msg-avatar" alt="Baymax" />
            <div class="msg-content" id="${msgId}">${formatMarkdown(text)}</div>`;
    } else {
        wrapper.innerHTML = `<div class="msg-content" id="${msgId}">${formatMarkdown(text)}</div>`;
    }

    chat?.appendChild(wrapper);
    if (chat) chat.scrollTop = chat.scrollHeight;
    return msgId;
}

function updateMessage(msgId, html) {
    const el = document.getElementById(msgId);
    if (el) el.innerHTML = html;
    const chat = document.getElementById("chatMessages");
    if (chat) chat.scrollTop = chat.scrollHeight;
}

function removeMessage(msgId) {
    document.getElementById(msgId)?.closest(".msg")?.remove();
}

function formatMarkdown(text) {
    if (!text) return "";
    return text
        .replace(/### (.*?)(\n|$)/g, "<h3>$1</h3>")
        .replace(/## (.*?)(\n|$)/g, "<h3>$1</h3>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/`(.*?)`/g, "<code>$1</code>")
        .replace(/\n\n/g, "<br><br>")
        .replace(/\n/g, "<br>");
}

// ── Intro Video Handling ──────────────────────────────────────────────────────
function initIntro() {
    const overlay = document.getElementById("introOverlay");
    const video = document.getElementById("introVideo");
    const skipBtn = document.getElementById("skipIntroBtn");

    if (!overlay || !video) return;

    const hideIntro = () => {
        overlay.classList.add("hidden");
        // Optional: stop the video just in case
        setTimeout(() => { video.pause(); video.src = ""; }, 1000);
    };

    video.addEventListener("ended", hideIntro);

    if (skipBtn) {
        skipBtn.addEventListener("click", hideIntro);
    }

    // Fallback if video fails to play or loads too slowly
    video.addEventListener("error", hideIntro);

    // If the browser blocks autoplay completely, we might need a timeout
    // but typically muted autoplay works. We add a fallback just in case:
    let isPlaying = video.currentTime > 0 && !video.paused && !video.ended && video.readyState > 2;
    if (!isPlaying) {
        video.play().catch(() => {
            // If play fails (e.g., policy), just hide intro
            hideIntro();
        });
    }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
    initIntro();

    // Setup menu button toggle (collapses sidebar to show chat interface only)
    const menuBtn = document.querySelector(".chat-menu-btn");
    if (menuBtn) {
        menuBtn.addEventListener("click", () => {
            document.body.classList.toggle("chat-only");
        });
    }

    // Init voice toggle label
    const toggle = document.getElementById("voiceToggle");
    if (toggle) {
        voiceEnabled = toggle.checked;
        const label = document.getElementById("voiceLabel");
        if (label) label.textContent = toggle.checked ? "🔊" : "🔇";
    }

    // Preload browser TTS voices (async)
    if ("speechSynthesis" in window) {
        window.speechSynthesis.getVoices();
        window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }

    setAE2Status(false, "AE2 Checking…", "#9CA3AF");
    initSDK();
});
