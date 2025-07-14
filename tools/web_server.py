import socket
import threading
import os
import time
import json

# For MJPEG stream
from io import BytesIO
from maix import image

# === Shared State ===
latest_jpeg = None  # updated externally by main.py
img_snapshot = None # will be used by main.py to update static bg
clients = set()     # active MJPEG streaming clients
control_flags = {
    "record": True,
    "show_raw": True,
    "set_background": False,
}

# === Config ===
STATIC_DIR = os.path.join(os.path.dirname(__file__), "../static")
STREAM_JPG_PATH = "/tmp/stream_frame.jpg"
HTTP_PORT = 80
WS_PORT = 8081


# === HTTP Server Thread ===
def handle_http(conn, addr):
    global latest_jpeg, img_snapshot
    try:
        request = conn.recv(1024).decode("utf-8")
        if not request:
            conn.close()
            return
        path = request.split(" ")[1]

        if path == "/" or path == "/index.html":
            file_path = os.path.join(STATIC_DIR, "index.html")
            mime = "text/html"
        elif path.endswith(".js"):
            file_path = os.path.join(STATIC_DIR, os.path.basename(path))
            mime = "application/javascript"
        elif path.endswith(".css"):
            file_path = os.path.join(STATIC_DIR, os.path.basename(path))
            mime = "text/css"
        elif path == "/stream.mjpg":
            stream_mjpeg(conn)
            return
        elif path.startswith("/snapshot.jpg"):
            if latest_jpeg:
                img_snapshot = image.load(STREAM_JPG_PATH, format = image.Format.FMT_RGBA8888)
                conn.send(b"HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n\r\n")
                conn.send(latest_jpeg)
            else:
                conn.send(b"HTTP/1.1 503 Service Unavailable\r\n\r\n")
            conn.close()
            return
        elif path == "/command":
            try:
                header, body = request.split("\r\n\r\n", 1)
                print("[Command] Body:", body)
                msg = json.loads(body)
                handle_command(msg)
                conn.send(b"HTTP/1.1 200 OK\r\n\r\n")
            except Exception as e:
                print("Command error:", e)
                conn.send(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            conn.close()
            return
        else:
            conn.send(b"HTTP/1.1 404 Not Found\r\n\r\n")
            conn.close()
            return

        with open(file_path, "rb") as f:
            body = f.read()
        header = f"HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {len(body)}\r\n\r\n"
        conn.send(header.encode("utf-8") + body)
    except Exception as e:
        print("HTTP error:", e)
    finally:
        conn.close()


# === MJPEG Streaming ===
def stream_mjpeg(conn):
    global latest_jpeg
    try:
        conn.send(b"HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n")
        clients.add(conn)
        while True:
            if latest_jpeg:
                conn.send(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                conn.send(latest_jpeg)
                conn.send(b"\r\n")
            time.sleep(0.05)
    except:
        pass
    finally:
        clients.discard(conn)
        conn.close()

def confirm_background(path):
    global img_snapshot
    img_snapshot.save(path)
    img_snapshot = None

# === WebSocket-like Control Server ===
# def handle_ws(conn, addr):
#     try:
#         while True:
#             data = conn.recv(1024)
#             if not data:
#                 break
#             try:
#                 msg = json.loads(data.decode("utf-8"))
#                 print("Received command:", msg)
#                 handle_command(msg)
#             except Exception as e:
#                 print("Bad WS message:", e)
#     except:
#         pass
#     finally:
#         conn.close()


def handle_command(msg):
    cmd = msg.get("command")
    val = msg.get("value")
    if cmd == "toggle_record":
        control_flags["record"] = bool(val)
    elif cmd == "toggle_raw":
        control_flags["show_raw"] = bool(val)
    elif cmd == "set_background":
        control_flags["set_background"] = True


# === External API ===
def send_frame(img):
    global latest_jpeg

    try:
        img.save(STREAM_JPG_PATH, quality=80)
        with open(STREAM_JPG_PATH, "rb") as f:
            latest_jpeg = f.read()
    except Exception as e:
        print("Error saving JPEG for stream:", e)



def get_control_flags():
    return control_flags


def reset_set_background_flag():
    control_flags["set_background"] = False


# === Main Server Loops ===
def start_servers():
    # HTTP Server
    def http_loop():
        sk = socket.socket()
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sk.bind(("0.0.0.0", HTTP_PORT))
        sk.listen(5)
        print(f"[HTTP] Listening on port {HTTP_PORT}")
        while True:
            conn, addr = sk.accept()
            threading.Thread(target=handle_http, args=(conn, addr), daemon=True).start()

    # # WebSocket-like Server
    # def ws_loop():
    #     sk = socket.socket()
    #     sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #     sk.bind(("0.0.0.0", WS_PORT))
    #     sk.listen(2)
    #     print(f"[WebSocket] Listening on port {WS_PORT}")
    #     while True:
    #         conn, addr = sk.accept()
    #         threading.Thread(target=handle_ws, args=(conn, addr), daemon=True).start()

    threading.Thread(target=http_loop, daemon=True).start()
    # threading.Thread(target=ws_loop, daemon=True).start()
