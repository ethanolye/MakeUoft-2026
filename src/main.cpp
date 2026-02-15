#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>

const char* WIFI_SSID = "ethanoly 13";
const char* WIFI_PASS = "ethanoly";

WebServer server(80);

// Motor control pins
const int PIN1 = 2;   // PWM motor 1 speed
const int PIN2 = 14;   // Motor 1 direction
const int PIN3 = 15;  // PWM motor 2 speed
const int PIN4 = 13;  // Motor 2 direction

// PWM settings
const int PWM_FREQ = 5000;  // 5kHz PWM frequency
const int PWM_BITS = 8;     // 8-bit resolution (0-255)
const int PWM_CHANNEL1 = 0;
const int PWM_CHANNEL2 = 1;
const int MOTOR_SPEED = 255; // Full speed

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
        server.send(503, "text/plain", "capture failed");
        return;
    }

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
    ledcWrite(PWM_CHANNEL1, 0);     // Stop motor 1
    ledcWrite(PWM_CHANNEL2, 0);     // Stop motor 2
    
    // Start non-blocking 2-second delay for PIN2 and PIN4
    delayStartTime = millis();
    delayActive = true;
    delayForDetection = true;
    
    server.send(200, "text/plain", "Person detected, motors STOP");
}

// Handler to reset the pin
void handlePersonGone() {
    ledcWrite(PWM_CHANNEL1, MOTOR_SPEED);   // Enable motor 1 at full speed
    ledcWrite(PWM_CHANNEL2, MOTOR_SPEED);   // Enable motor 2 at full speed
    
    // Start non-blocking 2-second delay for PIN2 and PIN4
    delayStartTime = millis();
    delayActive = true;
    delayForDetection = false;
    
    server.send(200, "text/plain", "No person, motors MOVING");
}

void setup() {
    Serial.begin(115200);
    Serial.println();

    // Configure PWM for motor speed control
    ledcSetup(PWM_CHANNEL1, PWM_FREQ, PWM_BITS);
    ledcSetup(PWM_CHANNEL2, PWM_FREQ, PWM_BITS);
    ledcAttachPin(PIN1, PWM_CHANNEL1);
    ledcAttachPin(PIN3, PWM_CHANNEL2);
    
    // Initialize direction pins
    pinMode(PIN2, OUTPUT);
    pinMode(PIN4, OUTPUT);
    
    // Startup config: motors running
    ledcWrite(PWM_CHANNEL1, MOTOR_SPEED);  // Motor 1 at full speed
    ledcWrite(PWM_CHANNEL2, MOTOR_SPEED);  // Motor 2 at full speed
    digitalWrite(PIN2, LOW);   // Direction 1
    digitalWrite(PIN4, LOW);   // Direction 2

    // Camera config
    {
        using namespace esp32cam;
        Config cfg;
        cfg.setPins(pins::AiThinker);
        cfg.setResolution(midRes);    // Start with mid-res for faster capture
        cfg.setBufferCount(4);        // Increased from 2 for smoother streaming
        cfg.setJpeg(60);              // Reduced from 80 for faster encoding (smaller files)

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
    
    // Handle non-blocking delay for direction pins
    if (delayActive && (millis() - delayStartTime >= 2000)) {
        if (delayForDetection) {
            // Person detected - stop by reducing speed
            digitalWrite(PIN2, LOW);
            digitalWrite(PIN4, LOW);
        } else {
            // Person gone - resume full speed
            digitalWrite(PIN2, LOW);
            digitalWrite(PIN4, LOW);
        }
        delayActive = false;
    }
}
