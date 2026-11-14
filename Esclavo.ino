#include <esp_now.h>
#include <WiFi.h>

// ===== CONFIGURACIÓN DE PINES =====
const int ENA = 4;   // PWM canal izquierdo
const int IN1 = 17;
const int IN2 = 16;
const int ENB = 23;  // PWM canal derecho
const int IN3 = 22;
const int IN4 = 19;
const int STBY = 18;

// ===== CONFIGURACIÓN DEL PWM (v3.x) =====
const int pwmFreq = 5000;     // 5 kHz
const int pwmResolution = 8;  // 0-255 (8 bits)

// ===== SEGURIDAD (WATCHDOG) =====
const unsigned long TIMEOUT_MS = 250;   // 250 ms sin datos → parar motores
bool isMoving = false;

// ===== DATOS RECIBIDOS =====
// El callback de ESP-NOW corre en la tarea de WiFi: solo copia los valores
// y el loop() mueve los motores. Así no se usa Serial ni String dentro del
// callback (bloqueaba la tarea de WiFi y podía desbordar su pila).
portMUX_TYPE rxMux = portMUX_INITIALIZER_UNLOCKED;
volatile unsigned long lastRecvTime = 0;
volatile int targetPwmL = 0;
volatile int targetPwmR = 0;
volatile bool newData = false;

// ===== FUNCIÓN DE CONTROL DE MOTORES =====
void controlMotors(int pwmL, int pwmR) {
  digitalWrite(STBY, HIGH);

  // Motor izquierdo
  if (pwmL >= 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  }
  ledcWrite(ENA, constrain(abs(pwmL), 0, 255));

  // Motor derecho
  if (pwmR >= 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  ledcWrite(ENB, constrain(abs(pwmR), 0, 255));

  isMoving = (pwmL != 0 || pwmR != 0);
}

// ===== CALLBACK QUE SE EJECUTA AL RECIBIR UN PAQUETE =====
// Trama esperada: "pwmL,pwmR" (la envía el Maestro).
void OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  char data[32];
  int n = min(len, (int)sizeof(data) - 1);
  memcpy(data, incomingData, n);
  data[n] = '\0';

  int l = 0, r = 0;
  char *coma = strchr(data, ',');
  if (coma != NULL && coma != data && coma[1] != '\0') {
    *coma = '\0';
    l = atoi(data);
    r = atoi(coma + 1);
  }
  // Si el formato es incorrecto se manda 0,0 (detener)

  portENTER_CRITICAL(&rxMux);
  targetPwmL = l;
  targetPwmR = r;
  newData = true;
  lastRecvTime = millis();
  portEXIT_CRITICAL(&rxMux);
}

void setup() {
  Serial.begin(115200);
  Serial.println("Iniciando esclavo ESP-NOW...");

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, LOW);   // driver apagado hasta que todo esté listo

  // WiFi y ESP-NOW primero: ledcAttach debe ir DESPUÉS de WiFi.mode()
  // para que WiFi no reinicie el periférico LEDC y desconecte el PWM.
  WiFi.mode(WIFI_STA);
  WiFi.setChannel(2);   // Importante: debe coincidir con el maestro

  delay(200);
  Serial.print("MAC: ");
  Serial.println(WiFi.macAddress());   // copiarla en Maestro.ino

  if (esp_now_init() == ESP_OK) {
    esp_now_register_recv_cb(OnDataRecv);
    Serial.println("ESP-NOW inicializado correctamente, esperando datos...");
  } else {
    Serial.println("ERROR: No se pudo inicializar ESP-NOW");
  }

  if (!ledcAttach(ENA, pwmFreq, pwmResolution)) {
    Serial.println("Error al configurar PWM en ENA");
  }
  if (!ledcAttach(ENB, pwmFreq, pwmResolution)) {
    Serial.println("Error al configurar PWM en ENB");
  }

  controlMotors(0, 0);
}

void loop() {
  int l = 0, r = 0;
  bool hayDato;
  unsigned long ultimo;

  portENTER_CRITICAL(&rxMux);
  hayDato = newData;
  if (hayDato) {
    l = targetPwmL;
    r = targetPwmR;
    newData = false;
  }
  ultimo = lastRecvTime;
  portEXIT_CRITICAL(&rxMux);

  if (hayDato) {
    controlMotors(l, r);
  }

  // Watchdog: si se perdió la conexión con el maestro, frenar por seguridad.
  // (ultimo se lee antes que millis(), así la resta nunca da negativo.)
  if (isMoving && (millis() - ultimo > TIMEOUT_MS)) {
    Serial.println("TIMEOUT: pérdida de conexión con el maestro. Frenando.");
    controlMotors(0, 0);
  }
  delay(5);
}
