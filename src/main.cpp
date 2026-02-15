#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>

const char* WIFI_SSID = "ethanoly 13";
const char* WIFI_PASS = "ethanoly";

WebServer server(80);

// Motor control pins
const int PIN1 = 2;   // Speed controller for motor 1
const int PIN2 = 14;   // Control pin 2
const int PIN3 = 15;  // Speed controller for motor 2
const int PIN4 = 13;  // Control pin 4

// Non-blocking delay state
unsigned long delayStartTime = 0;
bool delayActive = false;
bool delayForDetection = false;

// Predefined resolutions
static auto loRes = esp32cam::Resolution::find(320, 240);
static auto midRes = esp32cam::Resolution::find(350, 530);
static auto hiRes = esp32cam::Resolution::find(800, 600);

// Serve JPEG frame
void serveJpg() {
    auto frame = esp32cam::capture();
    if (frame == nullptr) {
        Serial.println("CAPTURE FAIL");
        server.send(503, "text/plain", "capture failed");
        return;
    }
    Serial.printf("CAPTURE OK %dx%d %db\n", frame->getWidth(), frame->getHeight(),
                  static_cast<int>(frame->size()));

    server.setContentLength(frame->size());
    server.send(200, "image/jpeg");

    WiFiClient client = server.client();
    frame->writeTo(client);  // send frame to client
}

// Handlers for different resolutions
void handleJpgLo()  { if (!esp32cam::Camera.changeResolution(loRes)) Serial.println("SET-LO-RES FAIL"); serveJpg(); }
void handleJpgMid() { if (!esp32cam::Camera.changeResolution(midRes)) Serial.println("SET-MID-RES FAIL"); serveJpg(); }
void handleJpgHi()  { if (!esp32cam::Camera.changeResolution(hiRes)) Serial.println("SET-HI-RES FAIL"); serveJpg(); }

// Handler for Python detection signal
void handlePersonDetected() {
    digitalWrite(PIN1, LOW);    // stop motor 1
    digitalWrite(PIN3, LOW);    // stop motor 2
    
    // Start non-blocking 2-second delay for PIN2 and PIN4
    delayStartTime = millis();
    delayActive = true;
    delayForDetection = true;
    
    server.send(200, "text/plain", "Person detected, motors STOP");
}

// Handler to reset the pin
void handlePersonGone() {
    digitalWrite(PIN1, HIGH);   // enable motor 1
    digitalWrite(PIN3, HIGH);   // enable motor 2
    
    // Start non-blocking 2-second delay for PIN2 and PIN4
    delayStartTime = millis();
    delayActive = true;
    delayForDetection = false;
    
    server.send(200, "text/plain", "No person, motors MOVING");
}

void setup() {
    Serial.begin(115200);
    Serial.println();

    // Initialize motor control pins
    pinMode(PIN1, OUTPUT);
    pinMode(PIN2, OUTPUT);
    pinMode(PIN3, OUTPUT);
    pinMode(PIN4, OUTPUT);
    
    // Startup config: motors running
    digitalWrite(PIN1, LOW);   // enable motor 1
    digitalWrite(PIN3, LOW);   // enable motor 2

    // Camera config
    {
        using namespace esp32cam;
        Config cfg;
        cfg.setPins(pins::AiThinker);
        cfg.setResolution(hiRes);
        cfg.setBufferCount(2);
        cfg.setJpeg(80);

        bool ok = Camera.begin(cfg);
        Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
    }

    // WiFi
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) { delay(500); }
    Serial.print("http://");
    Serial.println(WiFi.localIP());
    Serial.println("  /cam-lo.jpg");
    Serial.println("  /cam-mid.jpg");
    Serial.println("  /cam-hi.jpg");
    Serial.println("  /person-detected");
    Serial.println("  /person-gone");

    // Routes
    server.on("/cam-lo.jpg", handleJpgLo);
    server.on("/cam-mid.jpg", handleJpgMid);
    server.on("/cam-hi.jpg", handleJpgHi);

    server.on("/person-detected", handlePersonDetected);  // called by Python
    server.on("/person-gone", handlePersonGone);          // reset pin

    server.begin();
}

void loop() {
    server.handleClient();
    
    // Handle non-blocking delay for PIN2 and PIN4
    if (delayActive && (millis() - delayStartTime >= 2000)) {
        if (delayForDetection) {
            // Person detected - set PIN2 and PIN4 LOW
            digitalWrite(PIN1, LOW);
            digitalWrite(PIN3, LOW);
        } else {
            // Person gone - set PIN2 and PIN4 HIGH
            digitalWrite(PIN1, HIGH);
            digitalWrite(PIN3, HIGH);
        }
        delayActive = false;
    }
}
