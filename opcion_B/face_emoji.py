import os
import cv2
import time
import math
import threading
import numpy as np
from datetime import datetime

# ===== Config institucional =====
EVENTO = "Jornada Explora UNAB – Taller ICI"
CAMPUS = "UNAB Concepción"
LOGO_PATH = os.path.join("assets", "logo_unab.png")  # opcional; si no está, no falla
SAVE_DIR = "records_B"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== Parámetros de rendimiento =====
MIRROR = True
DETECT_WIDTH = 320  # 320–480 recomendado
QUALITY_PRESETS = {"Alta": 1, "Media": 2, "Baja": 3}
quality_name = "Alta"
skip_every = QUALITY_PRESETS[quality_name]

# ===== MediaPipe FaceMesh =====
try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit(f"ERROR: No se pudo importar MediaPipe. Detalle: {e}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,   # True mejora detalles (labios/ojos) pero baja FPS
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ===== Utilidades de overlay =====
def load_logo_rgba(path, target_h=48):
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = img.shape[:2]
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

LOGO_RGBA = load_logo_rgba(LOGO_PATH, target_h=48)

def overlay_rgba(bg_bgr: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    if fg_rgba is None:
        return bg_bgr
    if bg_bgr.shape[2] == 3:
        bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2BGRA)
    else:
        bg = bg_bgr.copy()
    fg = fg_rgba
    if fg.shape[:2] != bg.shape[:2]:
        fg = cv2.resize(fg, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = (fg[:, :, 3:4].astype(np.float32) / 255.0)
    inv = 1.0 - alpha
    out = bg.astype(np.float32)
    out[:, :, :3] = alpha * fg[:, :, :3].astype(np.float32) + inv * out[:, :, :3]
    out[:, :, 3] = 255.0
    out = out.astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)

def make_bottom_banner(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    bar_h = 56
    banner = np.zeros((bar_h, w, 4), dtype=np.uint8)
    banner[:, :, :3] = (0, 0, 0)
    banner[:, :, 3] = 150
    x = 10
    if LOGO_RGBA is not None:
        lh, lw = LOGO_RGBA.shape[:2]
        lw = min(lw, w // 5)
        scaled_logo = cv2.resize(LOGO_RGBA, (lw, lh), interpolation=cv2.INTER_AREA)
        roi = banner[4:4+lh, x:x+lw]
        if roi.shape[0] == lh and roi.shape[1] == lw:
            a = (scaled_logo[:, :, 3:4].astype(np.float32) / 255.0)
            roi[:, :, :3] = (a * scaled_logo[:, :, :3] + (1.0 - a) * roi[:, :, :3]).astype(np.uint8)
            roi[:, :, 3] = np.clip(a*255 + (1.0 - a) * roi[:, :, 3:4], 0, 255).astype(np.uint8)
            banner[4:4+lh, x:x+lw] = roi
        x += lw + 12
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text1 = f"{EVENTO} — {CAMPUS}"
    text2 = f"{now}"
    cv2.putText(banner, text1, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner, text2, (x, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200,255), 1, cv2.LINE_AA)
    out = frame_bgr.copy()
    y0 = h - bar_h
    if banner.shape[1] != w:
        banner = cv2.resize(banner, (w, bar_h), interpolation=cv2.INTER_LINEAR)
    roi = out[y0:h, 0:w]
    out[y0:h, 0:w] = overlay_rgba(roi, banner)
    return out

def save_with_banner_async(frame_bgr: np.ndarray, save_path: str):
    def _save():
        img = make_bottom_banner(frame_bgr)
        cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    threading.Thread(target=_save, daemon=True).start()

# ===== Emoji vectorial =====
def draw_emoji(canvas_bgr: np.ndarray, center, radius: int, kind: str):
    """Dibuja un emoji simple (🙂 😮 😐 😉 😑 🙁 😠 😁)"""
    x, y = center
    r = max(12, radius)

    # cara
    cv2.circle(canvas_bgr, (x, y), r, (0, 215, 255), -1)  # amarillo
    cv2.circle(canvas_bgr, (x, y), r, (0, 180, 255), 2)

    # ojos base
    eye_off_x = int(r * 0.35)
    eye_off_y = int(r * 0.25)
    eye_r = max(2, int(r * 0.10))

    def eye_left_closed():
        cv2.line(canvas_bgr, (x - eye_off_x - eye_r, y - eye_off_y),
                 (x - eye_off_x + eye_r, y - eye_off_y), (0, 0, 0), max(2, r // 12))
    def eye_right_closed():
        cv2.line(canvas_bgr, (x + eye_off_x - eye_r, y - eye_off_y),
                 (x + eye_off_x + eye_r, y - eye_off_y), (0, 0, 0), max(2, r // 12))

    # ojos según tipo
    if kind in ("wink_left", "eyes_closed"):
        eye_left_closed()
    else:
        cv2.circle(canvas_bgr, (x - eye_off_x, y - eye_off_y), eye_r, (0, 0, 0), -1)

    if kind in ("wink_right", "eyes_closed"):
        eye_right_closed()
    else:
        cv2.circle(canvas_bgr, (x + eye_off_x, y - eye_off_y), eye_r, (0, 0, 0), -1)

    # boca según tipo
    if kind == "neutral":      # 😐
        yb = y + int(r * 0.35)
        cv2.line(canvas_bgr, (x - int(r * 0.35), yb), (x + int(r * 0.35), yb), (0, 0, 0), max(2, r // 12))
    elif kind == "smile":      # 🙂
        yb = y + int(r * 0.25)
        axes = (int(r * 0.5), int(r * 0.35))
        cv2.ellipse(canvas_bgr, (x, yb), axes, 0, 15, 165, (0, 0, 0), max(2, r // 12))
    elif kind == "surprise":   # 😮
        mouth_r = max(3, int(r * 0.20))
        cv2.circle(canvas_bgr, (x, y + int(r * 0.30)), mouth_r, (0, 0, 0), -1)
    elif kind == "sad":        # 🙁 (arco invertido)
        yb = y + int(r * 0.35)
        axes = (int(r * 0.45), int(r * 0.25))
        cv2.ellipse(canvas_bgr, (x, yb), axes, 0, 200, 340, (0, 0, 0), max(2, r // 12))
    elif kind == "angry":      # 😠 (boca recta + cejas en ángulo)
        yb = y + int(r * 0.35)
        cv2.line(canvas_bgr, (x - int(r * 0.35), yb), (x + int(r * 0.35), yb), (0, 0, 0), max(2, r // 12))
        # cejas en ángulo
        cv2.line(canvas_bgr, (x - eye_off_x - eye_r, y - eye_off_y - eye_r),
                 (x - eye_off_x + eye_r, y - eye_off_y - 2*eye_r), (0, 0, 0), max(2, r // 12))
        cv2.line(canvas_bgr, (x + eye_off_x + eye_r, y - eye_off_y - eye_r),
                 (x + eye_off_x - eye_r, y - eye_off_y - 2*eye_r), (0, 0, 0), max(2, r // 12))
    elif kind == "grin":       # 😁 (sonrisa más ancha)
        yb = y + int(r * 0.23)
        axes = (int(r * 0.6), int(r * 0.32))
        cv2.ellipse(canvas_bgr, (x, yb), axes, 0, 10, 170, (0, 0, 0), max(2, r // 12))

# ===== Landmarks usados =====
# Boca: comisuras 61 (izq), 291 (der), labio sup 13, labio inf 14
LM_LEFT_CORNER  = 61
LM_RIGHT_CORNER = 291
LM_UPPER_LIP    = 13
LM_LOWER_LIP    = 14
# Ojos: esquinas (para ancho) y párpados (para apertura)
# Izquierdo: esquinas 33–133, párpados 159 (sup) y 145 (inf)
# Derecho:  263–362, 386 (sup) y 374 (inf)
LE_LEFT, LE_RIGHT = 33, 133
LE_UP, LE_DOWN    = 159, 145
RE_LEFT, RE_RIGHT = 263, 362
RE_UP, RE_DOWN    = 386, 374
# Cejas (aprox)
LBROW_TOP, RBROW_TOP = 70, 300  # puntos sobre ceja izq/der

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def eye_metrics(pts, left=True):
    if left:
        pL, pR = pts.get(LE_LEFT), pts.get(LE_RIGHT)
        pU, pD = pts.get(LE_UP), pts.get(LE_DOWN)
    else:
        pL, pR = pts.get(RE_LEFT), pts.get(RE_RIGHT)
        pU, pD = pts.get(RE_UP), pts.get(RE_DOWN)
    if not all([pL, pR, pU, pD]):
        return None, None
    w = dist(pL, pR) + 1e-6
    h = dist(pU, pD)
    return h / w, w

def classify_emoji(pts, face_bbox_w):
    """Heurísticas simples para 8 gestos."""
    pL = pts.get(LM_LEFT_CORNER)
    pR = pts.get(LM_RIGHT_CORNER)
    pU = pts.get(LM_UPPER_LIP)
    pD = pts.get(LM_LOWER_LIP)
    if not all([pL, pR, pU, pD]):
        return "neutral"

    mouth_w = dist(pL, pR) + 1e-6
    mouth_h = dist(pU, pD)

    ratio_open  = mouth_h / mouth_w                # apertura vertical / ancho boca
    ratio_smile = mouth_w / max(1.0, face_bbox_w)  # ancho boca / ancho cara aprox

    # Ojos
    le_ratio, le_w = eye_metrics(pts, left=True)
    re_ratio, re_w = eye_metrics(pts, left=False)

    # Cejas (distancia ceja-ojo): menor = ceja baja (enojo)
    lbrow = pts.get(LBROW_TOP)
    rbrow = pts.get(RBROW_TOP)
    # centro ojo aprox
    le_center = None
    re_center = None
    if all([pts.get(LE_LEFT), pts.get(LE_RIGHT), pts.get(LE_UP), pts.get(LE_DOWN)]):
        le_center = ((pts[LE_LEFT][0] + pts[LE_RIGHT][0]) // 2,
                     (pts[LE_UP][1] + pts[LE_DOWN][1]) // 2)
    if all([pts.get(RE_LEFT), pts.get(RE_RIGHT), pts.get(RE_UP), pts.get(RE_DOWN)]):
        re_center = ((pts[RE_LEFT][0] + pts[RE_RIGHT][0]) // 2,
                     (pts[RE_UP][1] + pts[RE_DOWN][1]) // 2)

    # ---- UMBRALES (ajustables según cámara/luz) ----
    OPEN_SURPRISE = 0.42     # 😮
    OPEN_MIN_SMILE = 0.18    # mínima apertura para considerar 🙂/😁
    SMILE_WIDTH   = 0.42     # boca relativamente ancha vs cara para 🙂
    GRIN_WIDTH    = 0.50     # más ancha -> 😁
    EYE_CLOSED    = 0.20     # ratio ojo (h/w) bajo -> ojo cerrado
    BROW_ANGRY_D  = 0.28     # distancia ceja-ojo (normalizada por ancho de ojo) baja -> 😠
    SAD_DROP      = 0.10     # comisuras más bajas que el centro de labios -> 🙁

    # Sorpresa: boca muy abierta verticalmente
    if ratio_open > OPEN_SURPRISE:
        return "surprise"

    # Ojos cerrados / guiños
    if le_ratio is not None and re_ratio is not None:
        le_closed = le_ratio < EYE_CLOSED
        re_closed = re_ratio < EYE_CLOSED
        if le_closed and re_closed:
            return "eyes_closed"
        if le_closed and not re_closed:
            return "wink_left"
        if re_closed and not le_closed:
            return "wink_right"

    # Triste: comisuras por debajo del centro de boca
    mouth_cy = (pU[1] + pD[1]) / 2.0
    corners_drop = ((pL[1] - mouth_cy) + (pR[1] - mouth_cy)) / 2.0  # >0 si comisuras más abajo
    if corners_drop > SAD_DROP * mouth_w:
        return "sad"

    # Enojado: cejas bajas (cerca del ojo)
    if lbrow and le_center and le_w and rbrow and re_center and re_w:
        lbrow_d = abs(lbrow[1] - le_center[1]) / max(1.0, le_w)
        rbrow_d = abs(rbrow[1] - re_center[1]) / max(1.0, re_w)
        if lbrow_d < BROW_ANGRY_D and rbrow_d < BROW_ANGRY_D:
            return "angry"

    # Sonrisa / Grin (ancho relativo de boca)
    if ratio_smile > GRIN_WIDTH and ratio_open > OPEN_MIN_SMILE:
        return "grin"
    if ratio_smile > SMILE_WIDTH and ratio_open > OPEN_MIN_SMILE:
        return "smile"

    return "neutral"

def main():
    global quality_name, skip_every  # usamos las globales (evita UnboundLocalError)

    save_every_sec = 8
    last_save_t = 0.0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("ERROR: No se pudo abrir la cámara.")

    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("ERROR: No se pudo leer el primer frame.")

    if MIRROR:
        frame0 = cv2.flip(frame0, 1)

    H, W = frame0.shape[:2]
    frame_count = 0
    fps_avg = 0.0
    last_t = time.time()

    help_lines = [
        "Controles: Q=Salir | 1/2/3=Calidad | S=Guardar ahora",
        "Gestos: 🙂 😮 😐 😉 😑 🙁 😠 😁",
    ]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)

        composed_bgr = frame.copy()
        frame_count += 1
        process_this = (frame_count % skip_every == 0)

        if process_this:
            # Downscale
            scale = DETECT_WIDTH / float(W)
            new_h = int(H * scale)
            small = cv2.resize(composed_bgr, (DETECT_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            res = face_mesh.process(small_rgb)
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark

                # Reescalar algunos landmarks a coords del frame original
                idxs = [
                    LM_LEFT_CORNER, LM_RIGHT_CORNER, LM_UPPER_LIP, LM_LOWER_LIP,
                    LE_LEFT, LE_RIGHT, LE_UP, LE_DOWN,
                    RE_LEFT, RE_RIGHT, RE_UP, RE_DOWN,
                    LBROW_TOP, RBROW_TOP
                ]
                pts = {}
                min_x, min_y = 1e9, 1e9
                max_x, max_y = -1e9, -1e9
                for idx in idxs:
                    lm = lms[idx]
                    x = int(lm.x * DETECT_WIDTH)
                    y = int(lm.y * new_h)
                    rx = int(x / scale)
                    ry = int(y / scale)
                    pts[idx] = (rx, ry)
                    min_x, min_y = min(min_x, rx), min(min_y, ry)
                    max_x, max_y = max(max_x, rx), max(max_y, ry)

                face_w = max(20, (max_x - min_x) * 2)
                kind = classify_emoji(pts, face_w)

                # Centro cerca de la boca
                mouth_cx = (pts[LM_LEFT_CORNER][0] + pts[LM_RIGHT_CORNER][0]) // 2
                mouth_cy = (pts[LM_UPPER_LIP][1] + pts[LM_LOWER_LIP][1]) // 2
                radius = max(30, int(face_w * 0.35))
                draw_emoji(composed_bgr, (mouth_cx, mouth_cy - int(radius * 0.6)), radius, kind)

        # HUD y FPS
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 1.0 / dt
            fps_avg = 0.9 * fps_avg + 0.1 * fps if fps_avg > 0 else fps

        cv2.rectangle(composed_bgr, (10, 10), (360, 95), (0, 0, 0), -1)
        cv2.putText(composed_bgr, f"FPS ~ {fps_avg:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(composed_bgr, f"Calidad: {quality_name} (skip={skip_every})", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(composed_bgr, "Gestos: 🙂 😮 😐 😉 😑 🙁 😠 😁", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # Guardado silencioso
        if (now - last_save_t) >= save_every_sec:
            last_save_t = now
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"faceemoji_{ts}.png")
            save_with_banner_async(composed_bgr, out_path)

        cv2.imshow("Face→Emoji UNAB (Escritorio)", composed_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('1'), ord('2'), ord('3')):
            quality_name = {"1": "Alta", "2": "Media", "3": "Baja"}[chr(key)]
            skip_every = QUALITY_PRESETS[quality_name]
        elif key in (ord('s'), ord('S')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"faceemoji_{ts}.png")
            save_with_banner_async(composed_bgr, out_path)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Silenciar algunos logs de TF/absl si molestan
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass
    main()
