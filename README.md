# Proyecto VSSS – Torneo Zacatecano de Fútbol Robótico

Sistema de control para un equipo de 3 robots en la categoría **VSSS** (Very Small Size Soccer) del Torneo Zacatecano de Fútbol Robótico, organizado por el COZCYT y el Club de Robótica RobTai-UAZ.

La plataforma base fue desarrollada por M. en C. José Manuel Mercado Blanco y entregada como parte del kit del torneo. **`MGApp.py` es mi versión modificada** para mi ordenador y sistema operativo (macOS), con ajustes de estrategia y corrección de errores (ver encabezado del archivo).

## Arquitectura

```
Cámara cenital ──► MGApp.py (visión + estrategia) ──USB serial 921600──► Maestro (ESP32)
                                                                          │ ESP-NOW
                                                         ┌────────────────┼────────────────┐
                                                      Esclavo POR     Esclavo MED     Esclavo DEL
```

| Archivo | Descripción |
|---|---|
| `MGApp.py` | App principal (Tkinter + OpenCV): corrección de perspectiva, detección en espacio Lab polar, filtro Kalman de la pelota, estrategias POR/MED/DEL y envío de la trama `<L,R\|L,R\|L,R>` |
| `calibrar_colores.py` | Calibrador interactivo de colores; genera `lab_calibration.json` |
| `cancha_config.json` | Esquinas de la cancha en la imagen de la cámara |
| `lab_calibration.json` | Rangos Lab polar de pelota, bases y marcadores |
| `Maestro.ino` | ESP32 conectado a la computadora; reenvía PWM a cada robot por ESP-NOW |
| `Esclavo.ino` | ESP32 de cada robot; controla motores (driver TB6612) con watchdog de 250 ms |
| `REGLAS_TFRZAC.pdf` | Reglamento del torneo |

## Requisitos

- Python 3.10+ con `opencv-python`, `numpy`, `pillow`, `pyserial`
- Arduino IDE con core ESP32 v3.x

```bash
pip install opencv-python numpy pillow pyserial
python3 MGApp.py
```

## Uso

1. Cargar `Maestro.ino` y `Esclavo.ino` (ajustar las MAC de los robots en el Maestro).
2. Ejecutar `MGApp.py`, elegir la cámara y el puerto serial.
3. Calibrar colores con **Auto colores** (usa la misma imagen procesada que el juego).
4. Elegir portería propia y color de uniforme, y **Conectar**.
