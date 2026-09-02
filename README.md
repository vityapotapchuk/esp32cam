# ESP32-CAM MJPEG streamer

AI-Thinker ESP32-CAM — WiFi (STA з фолбеком на AP) + MJPEG стрім-сервер.

## Залізо

- Плата: **AI-Thinker ESP32-CAM** (ESP32-D0WD-V3, 4 MB flash)
- Сенсор: **OV3660** (визначається автоматично)
- USB-UART адаптер на `/dev/ttyUSB0`, авто-reset через DTR/RTS

## Прошивка

```sh
cd ~/esp32cam
arduino-cli compile --fqbn esp32:esp32:esp32cam .
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32cam .
```

Якщо `upload` падає з `device reports readiness to read but returned no data` —
повторити 2–3 рази (авто-boot інколи не спрацьовує).

Серійний лог:

```sh
~/esp32-venv/bin/python monitor.py
```

Toolchain: `arduino-cli` 1.5.1, core `esp32:esp32` 3.1.3, FQBN `esp32:esp32:esp32cam`.

## Мережа

- STA: SSID `Dev` → отримує `192.168.1.186`, mDNS `esp32cam.local`
- Фолбек AP: `ESP32-CAM-xxxxxx` / пароль `12345678` → `http://192.168.4.1/`

## HTTP endpoints (порт 80)

| Шлях      | Опис                                  |
|-----------|---------------------------------------|
| `/`       | HTML-сторінка з `<img src="/stream">` |
| `/stream` | MJPEG (`multipart/x-mixed-replace`), VGA ~8 fps |
| `/status` | JSON: chip, revision, flash, heap, uptime, wifi |

## Відомі нюанси та обхідні рішення

- **WiFi power-save** давав пінг ~800 мс і обриви → `WiFi.setSleep(false)` після
  підключення STA. Тепер 2–5 мс, 0% втрат.
- **Періодичний збій ініціалізації камери** (`sccb: W [..] fail`,
  `Camera probe failed 0xffffffff`) → цикл із 5 повторів навколо
  `esp_camera_init` з `esp_camera_deinit` + power-cycle PWDN між спробами;
  XCLK знижено 20 → 10 МГц. Зазвичай успіх із 2-ї спроби.
  Якщо падає всі 5 разів — переткнути шлейф камери.
