#!/usr/bin/env python3
"""
calibrar_colores.py  –  Calibrador interactivo de colores para VSSS.

Muestra la vista cenital de la cancha en tiempo real con las detecciones de
color superpuestas.  El usuario hace clic sobre una región para muestrear el
rango Lab polar automáticamente y puede ajustarlo con el teclado.

Controles
─────────
  Clic izquierdo  Muestrea el rango Lab polar en la región clicada
  1-5             Selecciona el color activo
  Tab             Cicla el parámetro activo (l_min → l_max → r_min → … → t_max)
  ← / →           Disminuye / Aumenta el parámetro activo en 1
  Shift+← / →     Ajuste rápido en 5
  D               Cicla el panel derecho: Máscara | solo-L | solo-R | solo-T
  C               Abre la pantalla de selección de cámara
  S               Guarda los rangos actuales en lab_calibration.json
  R               Restaura los rangos por defecto
  Q / Esc         Cierra la ventana

Selección de cámara
───────────────────
  Al iniciar se muestran todas las cámaras detectadas (incluido iPhone via
  Continuity Camera) con previsualización en vivo.
  Presiona el número de la cámara deseada y Enter para confirmar.

Archivos
────────
  Entrada : cancha_config.json   (esquinas del campo para warp)
  E/S     : lab_calibration.json (rangos Lab polar)
"""

import cv2
import numpy as np
import json
import os
import math
import subprocess
import platform

# ─── Configuración ───────────────────────────────────────────────────────────
OUT_W, OUT_H  = 750, 650
PANEL_W       = 500
PANEL_H       = int(PANEL_W * OUT_H / OUT_W)   # 433 px
BOTTOM_H      = 155
WIN_W         = PANEL_W * 2
WIN_H         = PANEL_H + BOTTOM_H
SAMPLE_R      = 25              # radio de muestreo al hacer clic (px en warp)
MARGIN_L      = 22              # margen sobre L y croma tras muestrear
MARGIN_T      = 20              # margen sobre ángulo tras muestrear
WIN_NAME      = "VSSS - Calibrador de Colores"
SEL_WIN       = "VSSS - Seleccion de Camara"

SEL_THUMB_W   = 320
SEL_THUMB_H   = 240
SEL_LABEL_H   = 28
SEL_FOOTER_H  = 44
SEL_COLS      = 2

CALIB_FILE = os.path.join(os.path.dirname(__file__), "lab_calibration.json")
CANCHA_FILE = os.path.join(os.path.dirname(__file__), "cancha_config.json")

# ─── Parámetros ajustables ───────────────────────────────────────────────────
PARAM_NAMES  = ["l_min", "l_max", "r_min", "r_max", "t_min", "t_max"]
PARAM_LIMITS = {
    "l_min": (0, 253), "l_max": (2, 255),
    "r_min": (0, 253), "r_max": (2, 255),
    "t_min": (0, 359), "t_max": (0, 360),
}

# ─── Colores de visualización (BGR) ─────────────────────────────────────────
VIZ_BGR = {
    "naranja":  (0,   128, 255),
    "amarillo": (0,   220, 220),
    "rosa":     (180,  80, 220),
    "verde":    (0,   200,  50),
    "azul":     (230,  80,  30),
    "rojo":     (0,    0,  220),
}

ROL_DESC = {
    "naranja":  "Pelota",
    "amarillo": "Base equipo AMARILLO  |  Marcador POR (Uniforme Azul)",
    "rosa":     "Marcador POR + DEL",
    "verde":    "Marcador MED + DEL",
    "azul":     "Base equipo AZUL  |  Marcador POR (Uniforme Amarillo)",
    "rojo":     "Marcador MED (ambos uniformes)  |  Marcador POR (Uniforme Azul)",
}

COLORES = ["naranja", "amarillo", "rosa", "verde", "azul", "rojo"]

# Rojo Lab polar: ángulo ≈ 340-20° (envuelve 0°; t_min > t_max es válido)
DEFAULTS = {
    "naranja":  {"l_min":104,"l_max":204,"r_min":59, "r_max":104,"t_min":41,  "t_max":89 },
    "amarillo": {"l_min":166,"l_max":234,"r_min":50, "r_max":101,"t_min":83,  "t_max":97 },
    "rosa":     {"l_min":100,"l_max":220,"r_min":78, "r_max":134,"t_min":302, "t_max":360},
    "verde":    {"l_min":111,"l_max":231,"r_min":11, "r_max":111,"t_min":121, "t_max":158},
    "azul":     {"l_min":62, "l_max":127,"r_min":27, "r_max":129,"t_min":274, "t_max":292},
    "rojo":     {"l_min":40, "l_max":220,"r_min":30, "r_max":150,"t_min":340, "t_max":20 },
}

DIAG_LABELS = ["Mascara completa", "Solo L (luminosidad)", "Solo R (croma)", "Solo T (angulo)"]

# ─── Detección de cámaras ────────────────────────────────────────────────────
def get_camera_names_macos():
    if platform.system() != "Darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=6
        )
        data = json.loads(result.stdout)
        return [cam.get("_name", "Camara") for cam in data.get("SPCameraDataType", [])]
    except Exception:
        return []

def scan_cameras(max_idx=8):
    sys_names = get_camera_names_macos()
    found = []
    for idx in range(max_idx):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and frame is not None:
            pos  = len(found)
            name = sys_names[pos] if pos < len(sys_names) else f"Camara {idx}"
            found.append({"idx": idx, "name": name, "cap": cap})
        else:
            cap.release()
    return found

def select_camera_screen(cams):
    n    = len(cams)
    cols = min(n, SEL_COLS)
    rows = (n + SEL_COLS - 1) // SEL_COLS
    win_w = cols * SEL_THUMB_W
    win_h = rows * (SEL_THUMB_H + SEL_LABEL_H) + SEL_FOOTER_H

    cv2.namedWindow(SEL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SEL_WIN, win_w, win_h)

    sel_pos = 0
    while True:
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        for i, cam in enumerate(cams):
            row = i // SEL_COLS
            col = i % SEL_COLS
            x0  = col * SEL_THUMB_W
            y0  = row * (SEL_THUMB_H + SEL_LABEL_H)
            ok, frame = cam["cap"].read()
            if ok and frame is not None:
                canvas[y0:y0+SEL_THUMB_H, x0:x0+SEL_THUMB_W] = \
                    cv2.resize(frame, (SEL_THUMB_W, SEL_THUMB_H))
            else:
                cv2.rectangle(canvas, (x0, y0),
                              (x0+SEL_THUMB_W, y0+SEL_THUMB_H), (40,40,40), -1)
            ly = y0 + SEL_THUMB_H
            cv2.rectangle(canvas, (x0, ly),
                          (x0+SEL_THUMB_W, ly+SEL_LABEL_H), (40,40,40), -1)
            cv2.putText(canvas, f"[{i}] {cam['name']}", (x0+6, ly+19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)
            border = (0,230,0) if i == sel_pos else (70,70,70)
            cv2.rectangle(canvas, (x0, y0),
                          (x0+SEL_THUMB_W-1, ly+SEL_LABEL_H-1), border, 2)
        fy = rows * (SEL_THUMB_H + SEL_LABEL_H)
        cv2.rectangle(canvas, (0, fy), (win_w, win_h), (20,20,20), -1)
        hint = "  |  ".join(f"{i}: {cams[i]['name'][:18]}" for i in range(n))
        cv2.putText(canvas, hint, (6, fy+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140,140,140), 1)
        cv2.putText(canvas,
                    f"Presiona 0-{n-1} para elegir   Enter: confirmar   Q: cancelar",
                    (6, fy+36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
        cv2.imshow(SEL_WIN, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):
            break
        elif key in (ord('q'), ord('Q'), 27):
            sel_pos = 0
            break
        elif ord('0') <= key <= ord('0') + n - 1:
            sel_pos = key - ord('0')
    cv2.destroyWindow(SEL_WIN)
    return cams[sel_pos]["idx"]

# ─── Utilidades Lab polar ─────────────────────────────────────────────────────
def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts  = np.array(pts, dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_image(frame, src_pts):
    sp  = order_points_clockwise(src_pts)
    ul, ur, lr, ll = sp
    src = np.array([ul, ur, lr, ll], dtype="float32")
    dst = np.array([[0,0],[OUT_W-1,0],[OUT_W-1,OUT_H-1],[0,OUT_H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)

def precompute_lab(bgr, clahe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    ll, la, lb = cv2.split(lab)
    ll = clahe.apply(ll)
    enhanced = cv2.cvtColor(cv2.merge((ll, la, lb)), cv2.COLOR_Lab2BGR)
    a = la.astype(np.float32) - 128.0
    b = lb.astype(np.float32) - 128.0
    mag, ang = cv2.cartToPolar(a, b, angleInDegrees=True)
    return enhanced, (ll, mag, ang)

def _mask_l(l_ch, rng):
    return cv2.inRange(l_ch, int(rng["l_min"]), int(rng["l_max"]))

def _mask_r(mag, rng):
    return cv2.inRange(mag, float(rng["r_min"]), float(rng["r_max"]))

def _mask_t(ang, rng):
    t0, t1 = rng["t_min"], rng["t_max"]
    if t0 <= t1:
        return cv2.inRange(ang, float(t0), float(t1))
    return cv2.bitwise_or(
        cv2.inRange(ang, float(t0), 360.0),
        cv2.inRange(ang, 0.0, float(t1))
    )

def lab_polar_mask(l_ch, mag, ang, rng):
    return cv2.bitwise_and(_mask_l(l_ch, rng),
           cv2.bitwise_and(_mask_r(mag, rng), _mask_t(ang, rng)))

def compute_range_from_region(l_ch, mag, ang, cx, cy, radius):
    h, w = l_ch.shape
    ys, xs = np.ogrid[:h, :w]
    circle = ((xs-cx)**2 + (ys-cy)**2) <= radius**2
    l_vals = l_ch[circle].astype(float)
    r_vals = mag[circle]
    t_vals = ang[circle]
    if len(l_vals) < 10:
        return None

    l_lo = max(0,   int(np.percentile(l_vals, 3))  - MARGIN_L)
    l_hi = min(255, int(np.percentile(l_vals, 97)) + MARGIN_L)
    r_lo = max(0,   int(np.percentile(r_vals, 3))  - MARGIN_L)
    r_hi = min(255, int(np.percentile(r_vals, 97)) + MARGIN_L)

    sin_m = np.mean(np.sin(np.radians(t_vals)))
    cos_m = np.mean(np.cos(np.radians(t_vals)))
    mean_angle = (math.degrees(math.atan2(sin_m, cos_m)) + 360) % 360
    delta  = ((t_vals - mean_angle + 180) % 360) - 180
    spread = max(abs(float(np.percentile(delta, 3))),
                 abs(float(np.percentile(delta, 97))))
    spread = min(spread + MARGIN_T, 179)

    return {
        "l_min": l_lo, "l_max": l_hi,
        "r_min": r_lo, "r_max": r_hi,
        "t_min": round((mean_angle - spread + 360) % 360, 1),
        "t_max": round((mean_angle + spread) % 360, 1),
    }

# ─── Dibujo ───────────────────────────────────────────────────────────────────
def draw_detections(frame, l_ch, mag, ang, ranges):
    out = frame.copy()
    for name in COLORES:
        mask = lab_polar_mask(l_ch, mag, ang, ranges[name])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = VIZ_BGR[name]
        for c in cnts:
            if cv2.contourArea(c) > 15:
                cv2.drawContours(out, [c], -1, color, 2)
                (cx, cy), _ = cv2.minEnclosingCircle(c)
                cv2.putText(out, name[:3], (int(cx)-8, int(cy)+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out

def draw_diag_panel(l_ch, mag, ang, rng, diag_mode, color_bgr):
    """Genera el panel derecho según el modo diagnóstico."""
    if diag_mode == 0:
        mask = lab_polar_mask(l_ch, mag, ang, rng)
    elif diag_mode == 1:
        mask = _mask_l(l_ch, rng)
    elif diag_mode == 2:
        mask = _mask_r(mag, rng)
    else:
        mask = _mask_t(ang, rng)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    colored = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    colored[mask > 0] = color_bgr
    return cv2.resize(colored, (PANEL_W, PANEL_H))

def draw_bottom_panel(canvas, active_color_idx, ranges, active_param_idx,
                      diag_mode, last_msg):
    y0 = PANEL_H
    cv2.rectangle(canvas, (0, y0), (WIN_W, WIN_H), (28,28,28), -1)

    # ── Fila 1: botones de color ──────────────────────────────────────────
    btn_w = WIN_W // len(COLORES)
    for i, name in enumerate(COLORES):
        x1 = i * btn_w
        x2 = x1 + btn_w - 2
        bgr    = VIZ_BGR[name]
        border = (255,255,255) if i == active_color_idx else (70,70,70)
        cv2.rectangle(canvas, (x1, y0+4), (x2, y0+26), bgr, -1)
        cv2.rectangle(canvas, (x1, y0+4), (x2, y0+26), border, 2)
        cv2.putText(canvas, f"{i+1} {name}", (x1+4, y0+19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (0,0,0) if sum(bgr) > 400 else (255,255,255), 1)

    # ── Fila 2: parámetros con el activo resaltado ────────────────────────
    name = COLORES[active_color_idx]
    r    = ranges[name]
    param_x = 6
    for pi, pn in enumerate(PARAM_NAMES):
        val  = r[pn]
        text = f"{pn}:{int(val)}"
        fg   = (0, 230, 120) if pi == active_param_idx else (180, 180, 180)
        bg   = (60, 60, 60)  if pi == active_param_idx else (28, 28, 28)
        tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        if pi == active_param_idx:
            cv2.rectangle(canvas, (param_x-2, y0+30), (param_x+tw[0]+2, y0+48), bg, -1)
        cv2.putText(canvas, text, (param_x, y0+44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg, 1)
        param_x += tw[0] + 14

    cv2.putText(canvas, "Tab:param  Flechas / +- :ajustar  , . :x5",
                (WIN_W - 300, y0+44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110,110,110), 1)

    # ── Fila 3: rol + modo diagnóstico ────────────────────────────────────
    cv2.putText(canvas, ROL_DESC[name], (6, y0+64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
    diag_txt = f"D: {DIAG_LABELS[diag_mode]}"
    cv2.putText(canvas, diag_txt,
                (WIN_W - 290, y0+64), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 200, 200), 1)

    # ── Fila 4: instrucciones ─────────────────────────────────────────────
    cv2.putText(canvas,
                "CLIC: muestrear  |  S: guardar  |  R: restaurar  |  C: camara  |  Q: salir",
                (6, y0+84), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110,110,110), 1)

    # ── Fila 5: mensaje de estado ─────────────────────────────────────────
    if last_msg:
        cv2.putText(canvas, last_msg, (6, y0+108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,220,100), 1)

    # ── Fila 6: resumen de todos los rangos del color activo ─────────────
    summary = (f"L[{r['l_min']}-{r['l_max']}]  "
               f"R[{r['r_min']}-{r['r_max']}]  "
               f"T[{r['t_min']}-{r['t_max']}]")
    cv2.putText(canvas, f"{name.upper()}:  {summary}", (6, y0+130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, VIZ_BGR[name], 1)

def draw_crosshair(frame, cx, cy, radius, color=(255,255,255)):
    cv2.circle(frame, (cx, cy), radius, color, 1)
    cv2.line(frame, (cx-radius-4, cy), (cx+radius+4, cy), color, 1)
    cv2.line(frame, (cx, cy-radius-4), (cx, cy+radius+4), color, 1)

# ─── Estado del mouse ─────────────────────────────────────────────────────────
_mouse_click = None

def on_mouse(event, x, y, flags, param):
    global _mouse_click
    if event == cv2.EVENT_LBUTTONDOWN and x < PANEL_W and y < PANEL_H:
        _mouse_click = (x, y)

# ─── Carga / Guardado ─────────────────────────────────────────────────────────
def load_ranges():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            data = json.load(f)
        out = {k: dict(v) for k, v in DEFAULTS.items()}
        for k in COLORES:
            if k in data:
                out[k] = data[k]
        return out
    return {k: dict(v) for k, v in DEFAULTS.items()}

def save_ranges(ranges):
    with open(CALIB_FILE, "w") as f:
        json.dump(ranges, f, indent=2)

def open_camera_with_selection():
    print("Buscando camaras disponibles...")
    cams = scan_cameras()
    if not cams:
        return None, -1, ""
    for c in cams:
        print(f"  [{c['idx']}] {c['name']}")
    if len(cams) == 1:
        cam_idx  = cams[0]["idx"]
        cam_name = cams[0]["name"]
        cams[0]["cap"].release()
    else:
        cam_idx = select_camera_screen(cams)
        for c in cams:
            c["cap"].release()
        cam_name = next((c["name"] for c in cams if c["idx"] == cam_idx),
                        f"Camara {cam_idx}")
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, cam_idx, cam_name

# ─── Main ─────────────────────────────────────────────────────────────────────
# ─── Bucle principal ─────────────────────────────────────────────────────────
def main():
    global _mouse_click

    src_pts = None
    if os.path.exists(CANCHA_FILE):
        with open(CANCHA_FILE) as f:
            cfg = json.load(f)
        src_pts = cfg.get("puntos_cancha")
        print(f"✓ Esquinas de cancha cargadas: {src_pts}")
    else:
        print("⚠  cancha_config.json no encontrado — imagen sin warp")

    ranges = load_ranges()
    print(f"✓ Rangos cargados desde {CALIB_FILE}")

    cap, cam_idx, cam_name = open_camera_with_selection()
    if cap is None or not cap.isOpened():
        print("✗ No se encontro ninguna camara")
        return
    print(f"✓ Camara activa: [{cam_idx}] {cam_name}\n")

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    active_color = 0
    active_param = 0
    diag_mode    = 0
    last_msg     = f"Camara: {cam_name}  |  Clic sobre un color para muestrearlo."
    click_xy     = None
    scale_x      = OUT_W / PANEL_W
    scale_y      = OUT_H / PANEL_H

    print("Controles: 1-6 color | Tab param | Flechas/+- ajustar | , . ajuste x5 | D diagnostico | S guarda | Q sale")
    print("  Si las flechas no responden revisa la consola: aparecera '[DEBUG] raw_key=...' con el codigo real.\n")

    while True:
        for _ in range(2):
            cap.grab()
        ok, raw = cap.retrieve()
        if not ok:
            continue

        warped = warp_image(raw, src_pts) if src_pts is not None \
                 else cv2.resize(raw, (OUT_W, OUT_H))
        enhanced, (l_ch, mag, ang) = precompute_lab(warped, clahe)

        # Muestreo
        if _mouse_click is not None:
            dx, dy = _mouse_click
            _mouse_click = None
            click_xy = (dx, dy)
            wx, wy = int(dx * scale_x), int(dy * scale_y)
            new_range = compute_range_from_region(l_ch, mag, ang, wx, wy, SAMPLE_R)
            if new_range is not None:
                name = COLORES[active_color]
                ranges[name] = new_range
                last_msg = (f"✓ {name}: "
                            f"L[{new_range['l_min']}-{new_range['l_max']}] "
                            f"R[{new_range['r_min']}-{new_range['r_max']}] "
                            f"T[{new_range['t_min']}-{new_range['t_max']}]")
            else:
                last_msg = "⚠ Region muy pequena, intenta de nuevo"

        # Paneles
        left = draw_detections(enhanced, l_ch, mag, ang, ranges)
        if click_xy is not None:
            r_disp = max(1, int(SAMPLE_R / scale_x))
            draw_crosshair(left, click_xy[0], click_xy[1],
                           r_disp, VIZ_BGR[COLORES[active_color]])
        left_small  = cv2.resize(left, (PANEL_W, PANEL_H))
        name        = COLORES[active_color]
        right_small = draw_diag_panel(l_ch, mag, ang, ranges[name],
                                      diag_mode, VIZ_BGR[name])

        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:PANEL_H, :PANEL_W] = left_small
        canvas[:PANEL_H, PANEL_W:] = right_small
        cv2.putText(canvas, "Detecciones (clic para muestrear)",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
        cv2.putText(canvas, DIAG_LABELS[diag_mode],
                    (PANEL_W+6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, VIZ_BGR[name], 1)
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), (80,80,80), 1)
        draw_bottom_panel(canvas, active_color, ranges, active_param, diag_mode, last_msg)
        cv2.imshow(WIN_NAME, canvas)

        # cv2.waitKeyEx captura flechas correctamente (devuelve int extendido)
        raw_key = cv2.waitKeyEx(1)
        if raw_key == -1:
            continue
        key = raw_key & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break

        elif ord('1') <= key <= ord('0') + len(COLORES):
            active_color = key - ord('1')
            click_xy = None
            last_msg = f"Color activo: {COLORES[active_color]}"

        elif key == 9:   # Tab
            active_param = (active_param + 1) % len(PARAM_NAMES)
            last_msg = f"Parametro: {PARAM_NAMES[active_param]} = {int(ranges[COLORES[active_color]][PARAM_NAMES[active_param]])}"

        elif key in (ord('d'), ord('D')):
            diag_mode = (diag_mode + 1) % 4
            last_msg = f"Vista: {DIAG_LABELS[diag_mode]}"

        elif key in (ord('s'), ord('S')):
            save_ranges(ranges)
            last_msg = f"✓ Guardado en {os.path.basename(CALIB_FILE)}"
            print(f"✓ Rangos guardados en {CALIB_FILE}")
            for c in COLORES:
                r = ranges[c]
                print(f"  {c:10s}  L[{r['l_min']}-{r['l_max']}]  "
                      f"R[{r['r_min']}-{r['r_max']}]  "
                      f"T[{r['t_min']}-{r['t_max']}]")

        elif key in (ord('r'), ord('R')):
            ranges = {k: dict(v) for k, v in DEFAULTS.items()}
            last_msg = "Rangos restaurados a valores por defecto"

        elif key in (ord('c'), ord('C')):
            cap.release()
            cap, cam_idx, cam_name = open_camera_with_selection()
            if cap is None or not cap.isOpened():
                print("✗ No se pudo cambiar de camara")
                break
            cv2.setMouseCallback(WIN_NAME, on_mouse)
            last_msg = f"Camara cambiada: [{cam_idx}] {cam_name}"

        else:
            # ── Detección de flechas (multiplataforma) ────────────────────
            # Los códigos varían según SO y versión de OpenCV:
            #   macOS  : Izq=63234  Der=63235  (NSFunctionKey)
            #   Linux  : Izq=65361  Der=65363
            #   Windows: Izq=2424832 Der=2555904
            CODES_IZQ = {63234, 65361, 2424832}
            CODES_DER = {63235, 65363, 2555904}

            is_left  = raw_key in CODES_IZQ or key == ord(',')
            is_right = raw_key in CODES_DER or key == ord('.')

            # + / - : alternativa universal sin depender de códigos de flechas
            if key in (ord('+'), ord('=')):   # = sin Shift en la mayoría de teclados
                is_right = True
            elif key == ord('-'):
                is_left = True

            # Paso: 5 con , o .  |  1 con flechas o +/-
            step = 5 if key in (ord(','), ord('.')) else 1

            if is_left or is_right:
                pname = PARAM_NAMES[active_param]
                lo, hi = PARAM_LIMITS[pname]
                delta = step if is_right else -step
                cur   = ranges[COLORES[active_color]][pname]
                # t_min y t_max envuelven 360°
                if pname in ("t_min", "t_max"):
                    new_val = round((cur + delta + 360) % 360, 1)
                else:
                    new_val = int(max(lo, min(hi, cur + delta)))
                ranges[COLORES[active_color]][pname] = new_val
                last_msg = f"{COLORES[active_color]}.{pname} = {new_val}"
            elif raw_key > 0 and key not in (0, 255):
                # Tecla no reconocida: imprimir código para diagnóstico
                print(f"[DEBUG] raw_key={raw_key}  key={key}  "
                      f"(agrega raw_key a CODES_IZQ o CODES_DER si es una flecha)")

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Calibrador cerrado.")

if __name__ == "__main__":
    main()
