// script.js

const streamImg = document.getElementById('stream');
const popup = document.getElementById('popup');
const preview = document.getElementById('preview');
const toggleRecord = document.getElementById('toggleRecord');
const toggleRaw = document.getElementById('toggleRaw');
const setBackgroundBtn = document.getElementById('setBackgroundBtn');

// === 🔁 Send command using HTTP POST instead of WebSocket
function sendCommand(command, value = null) {
    fetch("/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command, value }),
    }).catch((err) => {
        console.error("Command failed:", err);
    });
}

// === ⬛ Event handlers
toggleRecord.onchange = () => {
    sendCommand("toggle_record", toggleRecord.checked);
};

toggleRaw.onchange = () => {
    sendCommand("toggle_raw", toggleRaw.checked);
};

autoUpdateBg.onchange = () => {
    sendCommand("auto_update_bg", autoUpdateBg.checked);
};

setBackgroundBtn.onclick = () => {
    preview.src = "/snapshot.jpg?_=" + Date.now();
    popup.style.display = "block";
};

function confirmBackground() {
    sendCommand("set_background");
    hidePopup();
}

function hidePopup() {
    popup.style.display = "none";
}

// Expose for inline onclick
window.confirmBackground = confirmBackground;
window.hidePopup = hidePopup;
