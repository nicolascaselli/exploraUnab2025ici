import os
import cv2
import time
import math
import threading
import numpy as np
from datetime import datetime

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit(f"ERROR: No se pudo importar MediaPipe. ¿Venv activo? Detalle: {e}")

# -------- Config institucional ----------
EVENTO = "Jornada Explora UNAB – Taller ICI"
CAMPUS = "UNAB Concepción"
LOGO_PATH = os.path.join("assets", "logo_unab.png")  # usa ruta por defecto si existiera
SAVE_DIR = "records_A"
os.makedirs(SAVE_DIR, exist_ok=True)
MIRROR = True  # mostrar en espejo (flip horizontal)


# -------- Parámetros de rendimiento -----
# Calidad: Alta=procesa todos los frames; Media/Baja=salta frames para bajar carga.
QUALITY_PRESETS = {
    "Alta": 1,    # usar cada frame
    "Media": 2,   # usar 1 de cada 2 frames
    "Baja": 3,    # usar 1 de cada 3 frames
}
quality_name = "Alta"
skip_every = QUALITY_PRESETS[quality_name]

# Reducción de resolución para detección (se reescala la mano al frame original)
DETECT_WIDTH = 320  # entre 320-480 suele ir bien

# Detección/tracking MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=0,            # 0 = más rápido
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# -------- Estado de dibujo --------------
drawing_enabled = True
colors_bgr = [
    (0, 255, 255),  # amarillo
    (0, 165, 255),  # naranja
    (0, 0, 255),    # rojo
    (0, 255, 0),    # verde
    (255, 0, 0),    # azul
    (255, 255, 255) # blanco
]
color_idx = 0
thickness = 6

# Lienzo RGBA (se creará tras conocer el tamaño de la cámara)
canvas_rgba = None
prev_point = None

# Guardado silencioso
SAVE_EVERY_SEC = 8
_last_save_t = 0.0

def load_logo(logo_path: str, target_h: int = 48):
    if not os.path.isfile(logo_path):
        return None
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        return None
    # Asegurar 4 canales
    if logo.shape[2] == 3:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
    h, w = logo.shape[:2]
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    logo = cv2.resize(logo, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return logo

logo_rgba = load_logo(LOGO_PATH, target_h=48)

def overlay_rgba(bg_bgr: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    """Alpha-blend de fg_rgba sobre bg_bgr (vectorizado). Devuelve BGR."""
    if fg_rgba is None:
        return bg_bgr
    # Convertir bg a BGRA
    if bg_bgr.shape[2] == 3:
        bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2BGRA)
    else:
        bg = bg_bgr.copy()

    fg = fg_rgba
    # Asegurar tamaños coinciden
    if fg.shape[:2] != bg.shape[:2]:
        fg = cv2.resize(fg, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_LINEAR)

    alpha = (fg[:, :, 3:4].astype(np.float32) / 255.0)
    inv_alpha = 1.0 - alpha
    out = bg.astype(np.float32)
    out[:, :, :3] = alpha * fg[:, :, :3].astype(np.float32) + inv_alpha * out[:, :, :3]
    out[:, :, 3] = 255.0
    out = out.astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)

def make_bottom_banner(frame_bgr: np.ndarray) -> np.ndarray:
    """Genera barra inferior semitransparente con texto y logo institucional."""
    h, w = frame_bgr.shape[:2]
    bar_h = 56
    banner = np.zeros((bar_h, w, 4), dtype=np.uint8)

    # Fondo negro semitransparente
    banner[:, :, :3] = (0, 0, 0)
    banner[:, :, 3] = 150  # alpha

    # Pegar logo si existe
    x = 10
    if logo_rgba is not None:
        lh, lw = logo_rgba.shape[:2]
        lw = min(lw, w // 5)  # evita sobrepasar
        scaled_logo = cv2.resize(logo_rgba, (lw, lh), interpolation=cv2.INTER_AREA)
        # Composición del logo sobre banner
        roi = banner[4:4+lh, x:x+lw]
        if roi.shape[0] == lh and roi.shape[1] == lw:
            alpha = (scaled_logo[:, :, 3:4].astype(np.float32) / 255.0)
            inv = 1.0 - alpha
            roi[:, :, :3] = (alpha * scaled_logo[:, :, :3] + inv * roi[:, :, :3]).astype(np.uint8)
            roi[:, :, 3] = np.clip((alpha * 255 + inv * roi[:, :, 3:4]), 0, 255).astype(np.uint8)
            banner[4:4+lh, x:x+lw] = roi
        x += lw + 12

    # Texto
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text1 = f"{EVENTO} — {CAMPUS}"
    text2 = f"{now}"
    cv2.putText(banner, text1, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner, text2, (x, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200,255), 1, cv2.LINE_AA)

    # Componer sobre frame (abajo)
    out = frame_bgr.copy()
    y0 = h - bar_h
    # Expandir banner a ancho exacto si hiciera falta
    if banner.shape[1] != w:
        banner = cv2.resize(banner, (w, bar_h), interpolation=cv2.INTER_LINEAR)
    roi = out[y0:h, 0:w]
    blended = overlay_rgba(roi, banner)
    out[y0:h, 0:w] = blended
    return out

def save_with_banner_async(frame_bgr: np.ndarray, save_path: str):
    def _save():
        img = make_bottom_banner(frame_bgr)
        cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    threading.Thread(target=_save, daemon=True).start()

def draw_line_on_canvas(canvas: np.ndarray, p1, p2, color_bgr, thickness_px):
    """Dibuja una línea con alpha-blend correcto (sin errores de broadcasting) sobre el canvas RGBA."""
    if p1 is None or p2 is None:
        return

    h, w = canvas.shape[:2]
    # Capa temporal para el trazo
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # Dibuja el trazo en BGR (sin alpha)
    cv2.line(layer_bgr, p1, p2, color_bgr, thickness_px, lineType=cv2.LINE_AA)

    # Convierte ese BGR a RGBA con alpha=255 solo donde hubo dibujo
    mask = cv2.cvtColor(layer_bgr, cv2.COLOR_BGR2GRAY)
    layer[:, :, :3] = layer_bgr
    layer[:, :, 3] = np.where(mask > 0, 255, 0).astype(np.uint8)

    # ---- Blending correcto RGBA (vectorizado) ----
    # Normaliza a [0,1]
    src_rgb = layer[:, :, :3].astype(np.float32) / 255.0
    src_a   = (layer[:, :, 3].astype(np.float32) / 255.0)               # (H,W)
    dst_rgb = canvas[:, :, :3].astype(np.float32) / 255.0
    dst_a   = (canvas[:, :, 3].astype(np.float32) / 255.0)               # (H,W)

    src_a3 = src_a[..., None]                                            # (H,W,1)
    dst_a3 = dst_a[..., None]                                            # (H,W,1)

    out_a = src_a + dst_a * (1.0 - src_a)                                # (H,W)
    # Evita división por cero
    out_a_safe = np.clip(out_a, 1e-6, 1.0)[..., None]                    # (H,W,1)

    out_rgb = (src_rgb * src_a3 + dst_rgb * dst_a3 * (1.0 - src_a3)) / out_a_safe

    canvas[:, :, :3] = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
    canvas[:, :, 3]  = np.clip(out_a  * 255.0, 0, 255).astype(np.uint8)

def main():
    global canvas_rgba, prev_point, color_idx, thickness, drawing_enabled, skip_every, quality_name, _last_save_t

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("ERROR: No se pudo abrir la cámara. Revisa permisos en Windows / cierra otras apps.")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise SystemExit("ERROR: No se pudo leer el primer frame de la cámara.")
    if MIRROR:
        frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    canvas_rgba = np.zeros((H, W, 4), dtype=np.uint8)

    frame_count = 0
    fps_avg = 0.0
    last_t = time.time()

    help_lines = [
        "Controles: Q=Salir | D=On/Off dibujo | C=Cambiar color | +/-=Grosor | 1/2/3=Calidad | R=Reset | S=Guardar ahora",
    ]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR:
            frame_bgr = cv2.flip(frame, 1)
        else:
            frame_bgr = frame
        show_bgr = frame_bgr.copy()

        # --- Downscale para detección ---
        frame_count += 1
        process_this = (frame_count % skip_every == 0)

        index_tip_pixel = None  # (x, y)
        if process_this:
            scale = DETECT_WIDTH / float(W)
            new_h = int(H * scale)
            small = cv2.resize(frame_bgr, (DETECT_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            res = hands.process(small_rgb)
            if res.multi_hand_landmarks:
                # Tomar la primera mano detectada
                hand = res.multi_hand_landmarks[0]
                # Índice: landmark 8
                lm = hand.landmark[8]
                x = int(lm.x * DETECT_WIDTH)
                y = int(lm.y * new_h)
                # Reescalar a tamaño original
                rx = int(x / scale)
                ry = int(y / scale)
                index_tip_pixel = (rx, ry)

        # --- Dibujo ---
        if drawing_enabled and index_tip_pixel is not None:
            if prev_point is None:
                prev_point = index_tip_pixel
            else:
                draw_line_on_canvas(canvas_rgba, prev_point, index_tip_pixel, colors_bgr[color_idx], thickness)
                prev_point = index_tip_pixel
        else:
            prev_point = None

        # Componer canvas sobre frame
        composed_bgr = overlay_rgba(show_bgr, canvas_rgba)

        # HUD
        color_bgr = colors_bgr[color_idx]
        cv2.rectangle(composed_bgr, (10, 10), (220, 110), (0, 0, 0), thickness=-1)
        cv2.putText(composed_bgr, f"Dibujo: {'ON' if drawing_enabled else 'OFF'}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(composed_bgr, f"Color: {color_bgr}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(composed_bgr, f"Grosor: {thickness}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # FPS aproximado
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 1.0 / dt
            fps_avg = 0.9 * fps_avg + 0.1 * fps if fps_avg > 0 else fps
        cv2.putText(composed_bgr, f"FPS ~ {fps_avg:.1f} | Calidad: {quality_name} (skip={skip_every})",
                    (10, H - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        for i, t in enumerate(help_lines):
            cv2.putText(composed_bgr, t, (10, H - 45 + i*20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("AirDraw UNAB (Escritorio)", composed_bgr)

        # Guardado silencioso cada SAVE_EVERY_SEC
        if (now - _last_save_t) >= SAVE_EVERY_SEC:
            _last_save_t = now
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"airdraw_{ts}.png")
            save_with_banner_async(composed_bgr, out_path)

        # Teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('d') or key == ord('D'):
            drawing_enabled = not drawing_enabled
        elif key == ord('c') or key == ord('C'):
            color_idx = (color_idx + 1) % len(colors_bgr)
        elif key == ord('+'):
            thickness = min(40, thickness + 1)
        elif key == ord('-') or key == ord('_'):
            thickness = max(1, thickness - 1)
        elif key == ord('r') or key == ord('R'):
            canvas_rgba[:] = 0
            prev_point = None
        elif key == ord('s') or key == ord('S'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"airdraw_{ts}.png")
            save_with_banner_async(composed_bgr, out_path)
        elif key in (ord('1'), ord('2'), ord('3')):
            if key == ord('1'):
                quality_name = "Alta"
            elif key == ord('2'):
                quality_name = "Media"
            elif key == ord('3'):
                quality_name = "Baja"
            skip_every = QUALITY_PRESETS[quality_name]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
