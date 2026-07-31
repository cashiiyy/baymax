const API_BASE = "";

async function sendMessage() {
    const input = document.getElementById("userInput");
    const query = input.value.trim();
    if (!query) return;

    appendMessage(query, "user");
    input.value = "";

    const loadingId = appendMessage("Thinking...", "assistant");

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

        // Trigger TTS if enabled
        if (document.getElementById("autoVoiceToggle").checked) {
            triggerTTS(data.response);
        }
    } catch (err) {
        const msgDiv = document.getElementById(loadingId);
        if (msgDiv) {
            msgDiv.innerText = "Error communicating with BAYMAX engine. Please check backend connection.";
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
    const msgDiv = document.createElement("div");
    const msgId = "msg-" + Date.now();
    msgDiv.id = msgId;
    msgDiv.className = `message ${role}`;
    msgDiv.innerHTML = formatMarkdown(text);
    chat.appendChild(msgDiv);
    chat.scrollTop = chat.scrollHeight;
    return msgId;
}

function formatMarkdown(text) {
    return text
        .replace(/\n/g, "<br>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

async function triggerTTS(text) {
    try {
        const formData = new FormData();
        formData.append("text", text.substring(0, 300)); // limit preview length for TTS
        const response = await fetch(`${API_BASE}/tts`, {
            method: "POST",
            body: formData
        });
        if (response.ok) {
            const blob = await response.blob();
            const audioUrl = URL.createObjectURL(blob);
            const player = document.getElementById("audioPlayer");
            player.src = audioUrl;
            player.style.display = "block";
            player.play();
        }
    } catch (e) {
        console.error("TTS playback error", e);
    }
}

async function handleOCRUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    appendMessage(`Uploaded document: ${file.name}`, "user");
    const loadingId = appendMessage("Extracting text via Medical OCR...", "assistant");

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
            msgDiv.innerText = `OCR Extracted Text:\n${data.extracted_text}`;
        }
    } catch (err) {
        const msgDiv = document.getElementById(loadingId);
        if (msgDiv) {
            msgDiv.innerText = "Error extracting text from image.";
        }
    }
}

function toggleVoiceInput() {
    alert("Voice input listening... Speak your symptoms clearly into your microphone.");
}
