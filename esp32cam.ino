/*
 * ESP32-CAM — WiFi AP + MJPEG streaming server
 *
 * Після прошивки:
 *   1. ESP32 підключається до WiFi "Dev" — IP дивіться в серійному моніторі
 *      або відкрийте http://esp32cam.local
 *   2. Якщо домашня мережа недоступна — плата підніме точку доступу
 *      "ESP32-CAM-XXXXXX" (пароль: 12345678), адреса http://192.168.4.1/
 *
 * Плата: AI Thinker ESP32-CAM
 * Бібліотеки: esp32-camera (вбудована в ядро esp32)
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>

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

WebServer server(80);

// --- MJPEG стрім ---
void handle_stream() {
  WiFiClient client = server.client();
  camera_fb_t *fb = NULL;

  String head = "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace;boundary=frame\r\n\r\n";
  server.sendContent(head);

  while (client.connected()) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      delay(100);
      continue;
    }

    String part = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                  String(fb->len) + "\r\n\r\n";
    server.sendContent(part);
    client.write(fb->buf, fb->len);
    server.sendContent("\r\n");

    esp_camera_fb_return(fb);
    fb = NULL;
  }
}

void handle_index() {
  String html = "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>ESP32-CAM</title>"
                "<style>body{font-family:sans-serif;background:#111;color:#eee;"
                "text-align:center;margin:0;padding:20px}"
                "img{max-width:100%;border-radius:8px}"
                "a.btn{display:inline-block;margin:10px;padding:12px 24px;"
                "background:#2a7;color:#111;border-radius:6px;text-decoration:none;"
                "font-weight:bold}</style></head><body>"
                "<h2>ESP32-CAM Stream</h2>"
                "<img src='/stream' alt='stream'>"
                "<p><a class='btn' href='/status'>Status JSON</a></p>"
                "</body></html>";
  server.send(200, "text/html", html);
}

void handle_status() {
  String ip = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP().toString()
                                              : WiFi.softAPIP().toString();
  String json = "{\"chip\":\"" + String(ESP.getChipModel()) + "\"" +
                ",\"revision\":" + String(ESP.getChipRevision()) +
                ",\"flash\":" + String(ESP.getFlashChipSize() / 1048576) + "MB" +
                ",\"heap_free\":" + String(ESP.getFreeHeap()) +
                ",\"uptime_s\":" + String(millis() / 1000) +
                ",\"wifi_mode\":\"" + (WiFi.status() == WL_CONNECTED ? "STA" : "AP") + "\"" +
                ",\"ssid\":\"" + (WiFi.status() == WL_CONNECTED ? sta_ssid : "ESP32-CAM-AP") + "\"" +
                ",\"ip\":\"" + ip + "\"}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n=== ESP32-CAM boot ===");

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

  server.on("/", HTTP_GET, handle_index);
  server.on("/stream", HTTP_GET, handle_stream);
  server.on("/status", HTTP_GET, handle_status);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}