/*
 * ESP32-CAM — WiFi STA (+AP fallback) · MJPEG stream :81 · control API :80
 *
 * Після прошивки:
 *   1. ESP32 підключається до WiFi "Dev" — IP дивіться в серійному моніторі
 *      або відкрийте http://esp32cam.local
 *   2. Якщо домашня мережа недоступна — плата підніме точку доступу
 *      "ESP32-CAM-XXXXXX" (пароль: 12345678), адреса http://192.168.4.1/
 *
 * Порт 80  — керування: /, /status, /capture, /control, /settings
 * Порт 81  — MJPEG стрім (окрема задача, не блокує керування)
 *
 * Плата: AI Thinker ESP32-CAM
 * Бібліотеки: esp32-camera (вбудована в ядро esp32)
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

// --- Домашня мережа (STA) ---
const char *sta_ssid = "Dev";
const char *sta_password = "538661149251361";
const char *hostname = "esp32cam";      // -> http://esp32cam.local

// --- Резервна точка доступу (AP) ---
const char *ap_ssid_prefix = "ESP32-CAM-";
const char *ap_password = "12345678";   // мін. 8 символів

// --- Піни камери (AI-Thinker ESP32-CAM) ---
#define PWDN_GPIO_NUM    32
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27
#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      21
#define Y4_GPIO_NUM      19
#define Y3_GPIO_NUM      18
#define Y2_GPIO_NUM       5
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

#define FLASH_LED_PIN     4    // біла підсвітка на AI-Thinker
#define STREAM_PORT       81
#define MAX_STREAM_CLIENTS 5

WebServer server(80);
WiFiServer streamServer(STREAM_PORT);
WiFiClient streamClients[MAX_STREAM_CLIENTS];
SemaphoreHandle_t fbMutex;              // серіалізує esp_camera_fb_get

volatile uint32_t min_frame_ms = 0;     // 0 = темп задає сенсор (~8 fps VGA)
volatile uint8_t  led_state = 0;
volatile uint8_t  stream_client_count = 0;
bool sta_mode = false;

// --- MJPEG стрім: одна задача на core 0, до 3 клієнтів на спільний кадр ---
static const char *STREAM_HEAD =
  "HTTP/1.1 200 OK\r\n"
  "Content-Type: multipart/x-mixed-replace;boundary=frame\r\n"
  "Access-Control-Allow-Origin: *\r\n\r\n";

static void drain_request(WiFiClient &c) {
  String req;
  uint32_t start = millis();
  while (millis() - start < 500) {
    while (c.available()) {
      req += (char)c.read();
      if (req.endsWith("\r\n\r\n") || req.length() > 2048) return;
    }
    delay(1);
  }
}

void streamTask(void *pv) {
  streamServer.begin();
  streamServer.setNoDelay(true);
  Serial.printf("Stream server on :%d\n", STREAM_PORT);

  uint32_t last_frame_ms = 0;
  for (;;) {
    // прийом нових клієнтів
    WiFiClient nc = streamServer.available();
    if (nc) {
      drain_request(nc);
      nc.setTimeout(2000);
      nc.setNoDelay(true);
      bool placed = false;
      for (int i = 0; i < MAX_STREAM_CLIENTS && !placed; i++) {
        if (!streamClients[i] || !streamClients[i].connected()) {
          streamClients[i] = nc;
          streamClients[i].print(STREAM_HEAD);
          placed = true;
        }
      }
      if (!placed) {
        nc.print("HTTP/1.1 503 Busy\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        nc.stop();
      }
    }

    // облік живих клієнтів
    uint8_t alive = 0;
    for (int i = 0; i < MAX_STREAM_CLIENTS; i++) {
      if (streamClients[i]) {
        if (streamClients[i].connected()) alive++;
        else streamClients[i].stop();
      }
    }
    stream_client_count = alive;

    if (alive == 0) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }
    if (min_frame_ms && millis() - last_frame_ms < min_frame_ms) {
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }

    camera_fb_t *fb = NULL;
    if (xSemaphoreTake(fbMutex, pdMS_TO_TICKS(500)) == pdTRUE) {
      fb = esp_camera_fb_get();
      if (fb) {
        for (int i = 0; i < MAX_STREAM_CLIENTS; i++) {
          if (!streamClients[i] || !streamClients[i].connected()) continue;
          String part = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                        String(fb->len) + "\r\n\r\n";
          streamClients[i].print(part);
          size_t w = streamClients[i].write(fb->buf, fb->len);
          streamClients[i].print("\r\n");
          if (w != fb->len) streamClients[i].stop();  // повільний клієнт — від'єднуємо
        }
        esp_camera_fb_return(fb);
      } else {
        Serial.println("Camera capture failed");
      }
      xSemaphoreGive(fbMutex);
      last_frame_ms = millis();
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

// --- Керування (порт 80) ---

void handle_index() {
  String html = "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>ESP32-CAM</title>"
                "<style>body{font-family:sans-serif;background:#111;color:#eee;"
                "text-align:center;margin:0;padding:20px}"
                "img{max-width:100%;border-radius:8px}"
                "a.btn,button{display:inline-block;margin:6px;padding:10px 18px;"
                "background:#2a7;color:#111;border:0;border-radius:6px;"
                "text-decoration:none;font-weight:bold;cursor:pointer;font-size:14px}"
                "button.off{background:#555;color:#eee}"
                "code{background:#222;padding:2px 6px;border-radius:4px}</style></head><body>"
                "<h2>ESP32-CAM</h2>"
                "<img id='cam' alt='stream'>"
                "<p>"
                "<a class='btn' href='/capture' target='_blank'>Snapshot</a>"
                "<a class='btn' href='/settings'>Settings JSON</a>"
                "<a class='btn' href='/status'>Status JSON</a>"
                "</p><p>"
                "<button onclick='ctl(\"led\",\"+1-led_state\")'>LED</button>"
                "<button class='off' onclick='ctl(\"led\",0)'>LED OFF</button>"
                "<button onclick='ctl(\"vflip\",\"+1-vflip\")'>VFlip</button>"
                "<button onclick='ctl(\"hmirror\",\"+1-hmirror\")'>HMirror</button>"
                "<button onclick='ctl(\"quality\",12)'>Q12</button>"
                "<button onclick='ctl(\"framesize\",\"vga\")'>VGA</button>"
                "<button onclick='ctl(\"framesize\",\"svga\")'>SVGA</button>"
                "</p><p><code>/control?var=framesize|quality|brightness|contrast|saturation|vflip|hmirror|led|fps&amp;val=…</code></p>"
                "<script>var b=location.hostname;document.getElementById('cam').src='http://'+b+':81/stream?'+Date.now();"
                "function ctl(v,val){fetch('/control?var='+v+'&val='+val).then(r=>r.text()).then(t=>console.log(v,t))}</script>"
                "</body></html>";
  server.send(200, "text/html", html);
}

void handle_status() {
  String ip = sta_mode ? WiFi.localIP().toString() : WiFi.softAPIP().toString();
  String json = "{\"chip\":\"" + String(ESP.getChipModel()) + "\"" +
                ",\"revision\":" + String(ESP.getChipRevision()) +
                ",\"flash\":" + String(ESP.getFlashChipSize() / 1048576) + "\"" +
                ",\"heap_free\":" + String(ESP.getFreeHeap()) +
                ",\"uptime_s\":" + String(millis() / 1000) +
                ",\"wifi_mode\":\"" + (sta_mode ? "STA" : "AP") + "\"" +
                ",\"ssid\":\"" + (sta_mode ? sta_ssid : "ESP32-CAM-AP") + "\"" +
                ",\"ip\":\"" + ip + "\"" +
                ",\"stream_port\":" + String(STREAM_PORT) +
                ",\"stream_clients\":" + String(stream_client_count) +
                ",\"led\":" + String(led_state) + "}";
  server.send(200, "application/json", json);
}

void handle_capture() {
  camera_fb_t *fb = NULL;
  if (xSemaphoreTake(fbMutex, pdMS_TO_TICKS(1500)) != pdTRUE) {
    server.send(503, "text/plain", "busy");
    return;
  }
  fb = esp_camera_fb_get();
  if (!fb) {
    xSemaphoreGive(fbMutex);
    server.send(500, "text/plain", "capture failed");
    return;
  }
  String head = "HTTP/1.1 200 OK\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: " + String(fb->len) + "\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
  server.sendContent(head);
  server.client().write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
  xSemaphoreGive(fbMutex);
}

void handle_settings() {
  sensor_t *s = esp_camera_sensor_get();
  if (!s) {
    server.send(500, "application/json", "{\"error\":\"no sensor\"}");
    return;
  }
  char buf[384];
  snprintf(buf, sizeof(buf),
           "{\"framesize\":%d,\"quality\":%d,\"brightness\":%d,\"contrast\":%d,"
           "\"saturation\":%d,\"sharpness\":%d,\"special_effect\":%d,"
           "\"vflip\":%d,\"hmirror\":%d,\"led\":%d,\"fps_limit_ms\":%u,"
           "\"stream_clients\":%u,\"stream_port\":%d}",
           s->status.framesize, s->status.quality, s->status.brightness,
           s->status.contrast, s->status.saturation, s->status.sharpness,
           s->status.special_effect, s->status.vflip, s->status.hmirror,
           led_state, min_frame_ms, stream_client_count, STREAM_PORT);
  server.send(200, "application/json", buf);
}

static bool framesize_from_name(const String &n, framesize_t &out) {
  String v = n; v.toLowerCase();
  if      (v == "qqvga") out = FRAMESIZE_QQVGA;
  else if (v == "qvga")  out = FRAMESIZE_QVGA;
  else if (v == "vga")   out = FRAMESIZE_VGA;
  else if (v == "svga")  out = FRAMESIZE_SVGA;
  else if (v == "xga")   out = FRAMESIZE_XGA;
  else if (v == "sxga")  out = FRAMESIZE_SXGA;
  else if (v == "uxga")  out = FRAMESIZE_UXGA;
  else return false;
  return true;
}

void handle_control() {
  if (!server.hasArg("var") || !server.hasArg("val")) {
    server.send(400, "text/plain", "usage: /control?var=NAME&val=VALUE");
    return;
  }
  String var = server.arg("var");
  String val = server.arg("val");
  sensor_t *s = esp_camera_sensor_get();
  if (!s) {
    server.send(500, "text/plain", "no sensor");
    return;
  }

  String res = "ok " + var + "=" + val;
  bool is_toggle = val.startsWith("+1-");   // "+1-<current>" -> інвертувати
  int  ival = val.toInt();

  if (var == "framesize") {
    framesize_t fs;
    if (!framesize_from_name(val, fs)) { server.send(400, "text/plain", "bad framesize (qqvga..uxga)"); return; }
    s->set_framesize(s, fs);
  } else if (var == "quality") {
    if (ival < 10 || ival > 63) { server.send(400, "text/plain", "quality: 10..63"); return; }
    s->set_quality(s, ival);
  } else if (var == "brightness") {
    s->set_brightness(s, constrain(ival, -2, 2));
  } else if (var == "contrast") {
    s->set_contrast(s, constrain(ival, -2, 2));
  } else if (var == "saturation") {
    s->set_saturation(s, constrain(ival, -2, 2));
  } else if (var == "vflip") {
    int cur = s->status.vflip;
    s->set_vflip(s, is_toggle ? !cur : (ival != 0));
  } else if (var == "hmirror") {
    int cur = s->status.hmirror;
    s->set_hmirror(s, is_toggle ? !cur : (ival != 0));
  } else if (var == "led") {
    int cur = led_state;
    led_state = is_toggle ? !cur : (ival != 0);
    digitalWrite(FLASH_LED_PIN, led_state ? HIGH : LOW);
  } else if (var == "fps") {
    min_frame_ms = (ival > 0 && ival <= 30) ? 1000 / ival : 0;
  } else {
    server.send(400, "text/plain", "unknown var: " + var);
    return;
  }
  server.send(200, "text/plain", res);
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== ESP32-CAM boot ===");

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Power-cycle сенсора перед ініціалізацією (допомагає при "липкому" I2C)
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, HIGH);  // сенсор у sleep
  delay(500);
  digitalWrite(PWDN_GPIO_NUM, LOW);   // будимо
  delay(500);

  // Конфігурація камери
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;      // 10МГц надійніше за 20 при довгому шлейфі / OV3660
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;    // 640x480
  config.jpeg_quality = 12;             // 10-63, менше = краща якість
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_DRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  // Ініціалізація з повторами — SCCB-запис іноді зривається з першого разу
  esp_err_t err = ESP_FAIL;
  for (int attempt = 1; attempt <= 5; attempt++) {
    err = esp_camera_init(&config);
    if (err == ESP_OK) {
      Serial.printf("Camera OK (спроба %d)\n", attempt);
      break;
    }
    Serial.printf("Camera init спроба %d невдала: 0x%x — повтор...\n", attempt, err);
    esp_camera_deinit();
    // швидкий power-cycle сенсора між спробами
    digitalWrite(PWDN_GPIO_NUM, HIGH);
    delay(400);
    digitalWrite(PWDN_GPIO_NUM, LOW);
    delay(400);
  }
  if (err != ESP_OK) {
    Serial.printf("Camera init FAILED остаточно: 0x%x — перевірте шлейф камери\n", err);
    delay(3000);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, 0);       // переверніть у 1, якщо зображення догори дриґом
  s->set_hmirror(s, 0);

  fbMutex = xSemaphoreCreateMutex();

  // WiFi: спершу STA (домашня мережа), при невдачі — резервна AP
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(hostname);
  WiFi.begin(sta_ssid, sta_password);
  Serial.printf("Підключення до \"%s\"", sta_ssid);

  bool sta_ok = true;
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500);
    Serial.print(".");
    if (i == 39) sta_ok = false;
  }

  if (sta_ok) {
    sta_mode = true;
    WiFi.setSleep(false);   // вимкнути WiFi power-save (інакше пінг ~800мс і обриви)
    Serial.println("\nSTA підключено!");
    Serial.print("IP: http://");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nSTA не вдалось — піднімаю резервну AP");
    WiFi.mode(WIFI_AP);
    uint8_t mac[6];
    WiFi.macAddress(mac);
    String ssid = ap_ssid_prefix + String(mac[3], HEX) + String(mac[4], HEX) + String(mac[5], HEX);
    WiFi.softAP(ssid.c_str(), ap_password);
    Serial.print("AP SSID: ");
    Serial.println(ssid);
    Serial.print("AP IP:   ");
    Serial.println(WiFi.softAPIP());
  }

  // mDNS: http://esp32cam.local (працює в STA)
  if (MDNS.begin(hostname)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("mDNS: http://%s.local\n", hostname);
  }

  // Керування — порт 80
  server.enableCORS(true);
  server.on("/", HTTP_GET, handle_index);
  server.on("/status", HTTP_GET, handle_status);
  server.on("/capture", HTTP_GET, handle_capture);
  server.on("/settings", HTTP_GET, handle_settings);
  server.on("/control", HTTP_GET, handle_control);
  server.begin();
  Serial.println("Control server started on :80");

  // Стрім — окрема задача на core 0, порт 81
  xTaskCreatePinnedToCore(streamTask, "stream", 8192, NULL, 3, NULL, 0);
}

void loop() {
  server.handleClient();

  // автовідновлення WiFi у режимі STA
  static uint32_t last_wifi_check = 0;
  if (sta_mode && millis() - last_wifi_check > 10000) {
    last_wifi_check = millis();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi відвалився — reconnect...");
      WiFi.reconnect();
    }
  }
}