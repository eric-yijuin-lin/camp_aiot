from machine import Pin, I2C
import time
import dht
from sh1106 import SH1106_I2C
import emoji
import framebuf


W, H = 128, 64
i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)

oled = SH1106_I2C(W, H, i2c)   # 有些版本也接受 addr=0x3c 之類參數
oled.sleep(False)
fb = framebuf.FrameBuffer(emoji.smile_24x24, emoji.W, emoji.H, framebuf.MONO_HLSB)
fb2 = framebuf.FrameBuffer(emoji.bitter_24x24, emoji.W, emoji.H, framebuf.MONO_HLSB)

# ===== 腳位設定 =====
DHT_PIN = 15
LED_OK_PIN = 18     # 綠：舒適
LED_BAD_PIN = 19    # 紅：不舒服

sensor = dht.DHT11(Pin(DHT_PIN))  # DHT11 就改成 dht.DHT11(Pin(DHT_PIN))
led_ok = Pin(LED_OK_PIN, Pin.OUT)
led_bad = Pin(LED_BAD_PIN, Pin.OUT)

# ===== KNN 訓練資料 (temp, humi, label) =====
# label: 1=舒適, 0=不舒服
# 你可以把這些點改成你自己的舒適偏好/實測標記資料
TRAIN_DATA = [
    # 舒適樣本
    (23.0, 45.0, 1),
    (24.0, 50.0, 1),
    (25.0, 55.0, 1),
    (22.5, 42.0, 1),
    (26.0, 50.0, 1),

    # 不舒服樣本（太熱/太冷/太潮/太乾）
    (30.0, 60.0, 0),
    (29.0, 75.0, 0),
    (18.0, 50.0, 0),
    (20.0, 80.0, 0),
    (27.5, 70.0, 0),
    (24.0, 25.0, 0),
    (32.0, 40.0, 0),
]

K = 5  # 建議用奇數，避免平手（資料少就用 3 或 5）

# ===== 特徵縮放（避免濕度範圍壓過溫度）=====
# 這裡用「大概的合理範圍」做縮放：
# temp: 0~50C, humi: 0~100%
def scale_features(t, h):
    return (t / 50.0, h / 100.0)

def knn_predict(t, h, k=5):
    tx, hx = scale_features(t, h)

    # 計算距離（用平方距離，省掉 sqrt）
    dists = []
    for (tt, hh, label) in TRAIN_DATA:
        sx, sy = scale_features(tt, hh)
        dx = tx - sx
        dy = hx - sy
        dist2 = dx*dx + dy*dy
        dists.append((dist2, label))

    # 取最近的 k 個
    dists.sort(key=lambda x: x[0])
    topk = dists[:k]

    # 多數決
    votes = 0
    for (_, label) in topk:
        votes += 1 if label == 1 else -1

    # votes > 0 => 舒適；votes < 0 => 不舒服
    # 平手時：用最近鄰那個的 label 當結果
    if votes > 0:
        return 1
    if votes < 0:
        return 0
    return topk[0][1]

while True:
    try:
        sensor.measure()
        t = sensor.temperature()
        h = sensor.humidity() - 30
        

        pred = knn_predict(t, h, K)  # 1=舒適 0=不舒服

        if pred == 1:
            led_ok.value(1)
            led_bad.value(0)
            print("舒適 (KNN)  T={:.1f}C H={:.1f}%".format(t, h))
        else:
            led_ok.value(0)
            led_bad.value(1)
            print("不舒服 (KNN) T={:.1f}C H={:.1f}%".format(t, h))
        
        oled.fill(0)
        oled.text(f"T: {t:.2f}", 0, 0)
        oled.text(f"H: {h:.2f}", 0, 12)
        if pred == 1:
            oled.blit(fb, 0, 25)
        else:
            oled.blit(fb2, 0, 25)
        oled.show()

    except Exception as e:
        led_ok.value(0)
        led_bad.value(0)
        print("讀取失敗:", e)

    time.sleep(2)