from maix import network, err

def connect_wifi(ssid, password):
    w = network.wifi.Wifi()
    e = w.connect(ssid, password, wait=True, timeout=60)
    err.check_raise(e, "connect wifi failed")
    print("Connect success, got ip:", w.get_ip())
