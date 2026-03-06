/*
  Rui Santos
  Complete project details at https://RandomNerdTutorials.com/esp32-cam-post-image-photo-server/
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files.
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
*/

#include <Arduino.h>
#include <WiFi.h>
// #include <WiFiClientSecure.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_camera.h"

// ===========================
// Enter your WiFi credentials
// ===========================
const char* ssid     = "iPhone-YJL";
const char* password = "12345678";

String serverName = "http://172.20.10.2:5000";   // REPLACE WITH YOUR Raspberry Pi IP ADDRESS
//String serverName = "example.com";   // OR REPLACE WITH YOUR DOMAIN NAME


const int cameraInitRetry = 10;

String payloadPrefix = "--RandomNerdTutorials\r\nContent-Disposition: form-data; name=\"file\"; filename=\"esp32-cam.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
String payloadSurfix = "\r\n--RandomNerdTutorials--\r\n";

// CAMERA_MODEL_ESP32S3_EYE
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5

#define Y2_GPIO_NUM 11
#define Y3_GPIO_NUM 9
#define Y4_GPIO_NUM 8
#define Y5_GPIO_NUM 10
#define Y6_GPIO_NUM 12
#define Y7_GPIO_NUM 18
#define Y8_GPIO_NUM 17
#define Y9_GPIO_NUM 16

#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13



const int httpInterval = 2000;
unsigned long previousMillis = 0;   // last time image was sent

WiFiClient *client = nullptr;

void setupCamera()
{
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
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
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG; // for streaming
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  // for larger pre-allocated frame buffer.
  if(psramFound()){
    Serial.printf("PARAM found, setting higher quality.");
    delay(2000);
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    Serial.printf("PARAM not found, setting lower quality.");
    delay(2000);
    // Limit the frame size when PSRAM is not available
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }
  
  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    delay(1000);
    ESP.restart();
  }
}

void ensureWifiConnection() {
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
}

void setup() {
  // WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); 
  Serial.begin(115200);
  for (int i=0; i<10; i++)
  {
    Serial.print("brownout test");
    delay(1000);
  }

  WiFi.mode(WIFI_STA);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  ensureWifiConnection();
  
  Serial.println();
  Serial.print("ESP32-CAM IP Address: ");
  Serial.println(WiFi.localIP());
  client = new WiFiClient;

  // 設定並初始化相機
  setupCamera();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    ensureWifiConnection();
  }
  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    ESP.restart();
  }

  size_t prefixLen = payloadPrefix.length();
  uint8_t *prefix = (uint8_t *)payloadPrefix.c_str();
  size_t fbLen = fb->len;
  uint8_t *fbBuf = fb->buf;
  size_t surfixLen = payloadSurfix.length();
  uint8_t *surfix = (uint8_t *)payloadSurfix.c_str();

  // Step 1: Calculate the total length
  size_t totalLength = fbLen + prefixLen + surfixLen;
  
  // Step 2: Allocate new buffer
  uint8_t* concatenated = (uint8_t*)malloc(totalLength);

  // Step 3: Copy each array into the buffer
  memcpy(concatenated, prefix, prefixLen);
  memcpy(concatenated + prefixLen, fbBuf, fbLen);
  memcpy(concatenated + prefixLen + fbLen, surfix, surfixLen);
    
  if(client) {
    String query = serverName + "/esp32/capture";
    HTTPClient https;
    Serial.println("[POST] " + query);
    if (https.begin(*client, query)) {
      https.addHeader("Content-Type", "multipart/form-data;boundary=RandomNerdTutorials");
      int httpCode = https.POST(concatenated, totalLength);
      if (httpCode > 0) {
       Serial.printf("[STATUS CODE] %d\n", httpCode);
        if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_MOVED_PERMANENTLY) {
          String response = https.getString();
          Serial.printf("[RESPONSE] :  %s\n", response);
        }
      }
      else {
        Serial.printf("[ERROR]: %s\n", https.errorToString(httpCode).c_str());
      }
      https.end();
    }
  }
  else {
    Serial.printf("[HTTPS] Unable to connect\n");
  }
  esp_camera_fb_return(fb);
  free(concatenated);
  delay(httpInterval);
}