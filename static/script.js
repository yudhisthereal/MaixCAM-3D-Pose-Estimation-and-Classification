const streamImg = document.getElementById('stream');
const popup = document.getElementById('popup');
const preview = document.getElementById('preview');
const toggleRecord = document.getElementById('toggleRecord');
const toggleRaw = document.getElementById('toggleRaw');
const setBackgroundBtn = document.getElementById('setBackgroundBtn');

const ws = new WebSocket("ws://" + location.hostname + ":8765");

toggleRecord.onchange = () => {
  ws.send(JSON.stringify({
    command: "toggle_record",
    value: toggleRecord.checked
  }));
};

toggleRaw.onchange = () => {
  ws.send(JSON.stringify({
    command: "toggle_raw",
    value: toggleRaw.checked
  }));
};

setBackgroundBtn.onclick = () => {
  preview.src = streamImg.src + '&_=' + Date.now(); // force refresh
  popup.style.display = "block";
};

function hidePopup() {
  popup.style.display = "none";
}

function confirmBackground() {
  ws.send(JSON.stringify({ command: "set_background" }));
  hidePopup();
}

// Attach to buttons inside popup
window.confirmBackground = confirmBackground;
window.hidePopup = hidePopup;
