const API_BASE = "";
let recognition = null;
let isRecording = false;

// Initialize Speech Recognition if supported by browser
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById("userInput").value = transcript;
        stopSpeechRecognition();
        sendMessage();
    };

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        stopSpeechRecognition();
    };

    recognition.onend = () => {
        stopSpeechRecognition();
    };
}

function toggleSpeechRecognition() {
    if (!recognition) {
        alert("Web Speech recognition is not supported in this browser. You can type your query directly.");
        return;
    }

    if (isRecording) {
        stopSpeechRecognition();
    } else {
        startSpeechRecognition();
    }
}

function startSpeechRecognition() {
    isRecording = true;
    const btn = document.getElementById("micBtn");
    btn.classList.add("recording");
    btn.title = "Listening... Speak now";
    recognition.start();
}

function stopSpeechRecognition() {
    isRecording = false;
    const btn = document.getElementById("micBtn");
    btn.classList.remove("recording");
    btn.title = "Hold to Speak / Speech Input";
    try { recognition.stop(); } catch (e) {}
}

async function sendMessage() {
    const input = document.getElementById("userInput");
    const query = input.value.trim();
    if (!query) return;

    appendMessage(query, "user");
    input.value = "";

    const loadingId = appendMessage("BAYMAX is processing your health request...", "assistant");

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: 1, query: query })
        });

        const data = await response.json();
        const msgDiv = document.getElementById(loadingId);
        if (msgDiv) {
            msgDiv.innerHTML = formatMarkdown(data.response);
        }

        // Voice Response Output if enabled
        if (document.getElementById("voiceToggle").checked) {
            speakText(data.response);
        }
    } catch (err) {
        const msgDiv = document.getElementById(loadingId);
        if (msgDiv) {
            msgDiv.innerText = "Error connecting to BAYMAX engine. Ensure server is online.";
        }
    }
}

function quickQuery(text) {
    document.getElementById("userInput").value = text;
    sendMessage();
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

function appendMessage(text, role) {
    const chat = document.getElementById("chatMessages");
    const msgWrapper = document.createElement("div");
    const msgId = "msg-" + Date.now();
    msgWrapper.className = `msg ${role}`;
    
    if (role === "assistant") {
        msgWrapper.innerHTML = `
            <div class="msg-avatar">B</div>
            <div class="msg-content" id="${msgId}">${formatMarkdown(text)}</div>
        `;
    } else {
        msgWrapper.innerHTML = `
            <div class="msg-content" id="${msgId}">${formatMarkdown(text)}</div>
        `;
    }

    chat.appendChild(msgWrapper);
    chat.scrollTop = chat.scrollHeight;
    return msgId;
}

function formatMarkdown(text) {
    if (!text) return "";
    let html = text
        .replace(/### (.*?)\n/g, "<h3>$1</h3>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/\n\n/g, "<br><br>")
        .replace(/\n/g, "<br>");
    return html;
}

function speakText(text) {
    // Strip markdown formatting for clear speech
    const cleanText = text.replace(/[*#_🚨🩹🫀🦟🩺🐍🌡️🫁📋💡]/g, "").replace(/<[^>]*>/g, "");
    
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel(); // Stop any active speech
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }
}

async function handleOCRUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    appendMessage(`Uploaded document: ${file.name}`, "user");
    const loadingId = appendMessage("Scanning document via Medical OCR...", "assistant");

    try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch(`${API_BASE}/ocr`, {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        const msgDiv = document.getElementById(loadingId);
        if (msgDiv) {
            msgDiv.innerHTML = `<strong>OCR Extracted Text:</strong><br>${formatMarkdown(data.extracted_text)}`;
        }
    } catch (err) {
        const msgDiv = document.getElementById(loadingId);
        if (msgDiv) {
            msgDiv.innerText = "Error extracting text from medical image.";
        }
    }
}
