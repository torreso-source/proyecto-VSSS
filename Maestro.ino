#include <WiFi.h>
#include <esp_now.h>

// ==========================================
// DIRECCIONES MAC DE LOS 3 ROBOTS (ESCLAVOS)
// ==========================================
uint8_t mac_DEL[] = {0x70, 0x4b, 0xca, 0x83, 0x48, 0xa0};   // 70:4B:CA:83:48:A0
uint8_t mac_POR[] = {0x70, 0x4b, 0xca, 0x83, 0x4f, 0xa4};   // 70:4B:CA:83:4F:A4
uint8_t mac_MED[] = {0x30, 0x76, 0xf5, 0xa7, 0x68, 0xb4};   // 30:76:F5:A7:68:B4

esp_now_peer_info_t peerInfo;

// Buffer de lectura serial (evita String char-a-char)
static char rxBuf[64];
static uint8_t rxIdx = 0;

// Contadores para depuración
unsigned long lastPrintTime = 0;
int packetsSent = 0;
int packetsFailed = 0;

// Callback para confirmación de envío (v3.x)
void OnDataSent(const wifi_tx_info_t *tx_info, esp_now_send_status_t status) {
  if (status == ESP_NOW_SEND_SUCCESS) {
    packetsSent++;
  } else {
    packetsFailed++;
  }
}

// Función para registrar esclavo
void registrarEsclavo(uint8_t *mac) {
  memcpy(peerInfo.peer_addr, mac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  peerInfo.ifidx = WIFI_IF_STA;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Error añadiendo peer");
  } else {
    Serial.print("Peer registrado: ");
    for (int i = 0; i < 6; i++) {
      Serial.printf("%02X", mac[i]);
      if (i < 5) Serial.print(":");
    }
    Serial.println();
  }
}

void setup() {
  Serial.begin(921600);
  WiFi.mode(WIFI_STA);
  WiFi.setChannel(2);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ERR:INIT_ESPNOW");
    return;
  }

  registrarEsclavo(mac_POR);
  registrarEsclavo(mac_MED);
  registrarEsclavo(mac_DEL);

  esp_now_register_send_cb(OnDataSent);

  Serial.println("Maestro listo. Esperando PWMs desde Python...");
  Serial.println("Formato esperado: <pwmL,pwmR|pwmL,pwmR|pwmL,pwmR>");
}

void loop() {
  // --- LEER DATOS DEL PUERTO SERIE (desde Python) ---
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      rxBuf[rxIdx] = '\0';
      rxIdx = 0;

      // --- PROCESAR TRAMA ---
      char* buf = rxBuf;
      int len = strlen(buf);
      // Trim espacios/\r
      while (len > 0 && (buf[len-1] == '\r' || buf[len-1] == ' ')) buf[--len] = '\0';

      if (len > 2 && buf[0] == '<' && buf[len-1] == '>') {
        buf[len-1] = '\0';
        char* datos = buf + 1;

        char* split1 = strchr(datos, '|');
        char* split2 = split1 ? strchr(split1 + 1, '|') : nullptr;

        if (split1 && split2) {
          *split1 = '\0';
          *split2 = '\0';
          char* partes[3] = {datos, split1 + 1, split2 + 1};
          uint8_t* macs[3] = {mac_POR, mac_MED, mac_DEL};

          for (int i = 0; i < 3; i++) {
            if (strlen(partes[i]) > 0) {
              esp_now_send(macs[i], (uint8_t*)partes[i], strlen(partes[i]) + 1);
            }
          }
        }
      }

      // Estadísticas cada 5 segundos
      if (millis() - lastPrintTime > 5000) {
        Serial.printf("Paquetes OK: %d | Fallos: %d\n", packetsSent, packetsFailed);
        lastPrintTime = millis();
      }

    } else if (rxIdx < sizeof(rxBuf) - 1) {
      rxBuf[rxIdx++] = inChar;
    }
  }
}
