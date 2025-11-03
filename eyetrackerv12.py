import cv2, socket, threading, time, csv, os, sys
import numpy as np
from collections import deque

# ========= CONFIG (edit if needed) =========
VIDEO_PATH   = r"C:\Users\ryleg\OneDrive\Desktop\2D_classroom_fixed_v2.mp4"
AUDIO_WAV    = r"C:\Users\ryleg\OneDrive\Desktop\2D_classroom_fixed_v2.wav"
HOST, PORT   = "127.0.0.1", 4242
START_DELAY  = 0.15
# ==========================================

BASE_NAME   = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
DESKTOP_DIR = os.path.dirname(VIDEO_PATH)

# ---------- AUDIO (WAV via winsound) ----------
def audio_start(wav_path):
    if os.path.exists(wav_path):
        try:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_ASYNC | winsound.SND_FILENAME)
            print("Audio: WAV started")
            return True
        except Exception as e:
            print("Audio: WAV failed:", e)
    else:
        print("Audio: WAV not found; running silent")
    return False

def audio_stop():
    try:
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass

# ---------- GAZEPOINT ----------
def send(sock, cmd): sock.sendall((cmd + "\r\n").encode())

gp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
gp.connect((HOST, PORT))

# Enable OpenGaze streams
for cmd in [
    '<SET ID="ENABLE_SEND_TIME" STATE="1" />',
    '<SET ID="ENABLE_SEND_TIME_TICK" STATE="1" />',
    '<SET ID="ENABLE_SEND_POG_BEST" STATE="1" />',
    '<SET ID="ENABLE_SEND_POG_LEFT" STATE="1" />',
    '<SET ID="ENABLE_SEND_POG_RIGHT" STATE="1" />',
    '<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1" />',
    '<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1" />',
    '<SET ID="ENABLE_SEND_EYE_LEFT" STATE="1" />',
    '<SET ID="ENABLE_SEND_EYE_RIGHT" STATE="1" />',
    '<SET ID="ENABLE_SEND_DATA" STATE="1" />',
    '<SET ID="ENABLE_SEND_GSR" STATE="1" />',
    '<SET ID="ENABLE_SEND_HR" STATE="1" />',
    '<SET ID="ENABLE_SEND_HR_PULSE" STATE="1" />',
    '<SET ID="ENABLE_SEND_DIAL" STATE="1" />'
]:
    send(gp, cmd)

gaze = dict(
    TIME=0.0, TIME_TICK=0.0,
    LPOGX=0.5, LPOGY=0.5, LPOGV=0,
    RPOGX=0.5, RPOGY=0.5, RPOGV=0,
    BPOGX=0.5, BPOGY=0.5, BPOGV=0,
    LPD_px=0.0, RPD_px=0.0,
    LPUPILD_m=0.0, RPUPILD_m=0.0,
    LPUPILV=0, RPUPILV=0,
    GSR=0.0, GSRV=0,
    HR=0.0, HRV=0,
    HRP=0.0,
    DIAL=0.0, DIALV=0
)
_run_gaze = True
gaze_queue = deque()

def _f(s, key):
    if key in s:
        try: return float(s.split(f'{key}="')[1].split('"')[0])
        except: return None
    return None

def gaze_loop():
    buf = b""
    while _run_gaze:
        try:
            chunk = gp.recv(4096)
            if not chunk: break
            buf += chunk
            while b"\r\n" in buf:
                line, buf = buf.split(b"\r\n", 1)
                s = line.decode(errors="ignore")
                if "<REC" not in s: continue

                v = _f(s,"TIME");        gaze["TIME"]      = v if v is not None else gaze["TIME"]
                v = _f(s,"TIME_TICK");   gaze["TIME_TICK"] = v if v is not None else gaze["TIME_TICK"]
                v = _f(s,"BPOGX");       gaze["BPOGX"]     = v if v is not None else gaze["BPOGX"]
                v = _f(s,"BPOGY");       gaze["BPOGY"]     = v if v is not None else gaze["BPOGY"]
                v = _f(s,"BPOGV");       gaze["BPOGV"]     = int(v) if v is not None else gaze["BPOGV"]
                v = _f(s,"LPOGX");       gaze["LPOGX"]     = v if v is not None else gaze["LPOGX"]
                v = _f(s,"LPOGY");       gaze["LPOGY"]     = v if v is not None else gaze["LPOGY"]
                v = _f(s,"LPOGV");       gaze["LPOGV"]     = int(v) if v is not None else gaze["LPOGV"]
                v = _f(s,"RPOGX");       gaze["RPOGX"]     = v if v is not None else gaze["RPOGX"]
                v = _f(s,"RPOGY");       gaze["RPOGY"]     = v if v is not None else gaze["RPOGY"]
                v = _f(s,"RPOGV");       gaze["RPOGV"]     = int(v) if v is not None else gaze["RPOGV"]
                v = _f(s,"LPD");         gaze["LPD_px"]    = v if v is not None else gaze["LPD_px"]
                v = _f(s,"RPD");         gaze["RPD_px"]    = v if v is not None else gaze["RPD_px"]
                v = _f(s,"LPUPILD");     gaze["LPUPILD_m"] = v if v is not None else gaze["LPUPILD_m"]
                v = _f(s,"RPUPILD");     gaze["RPUPILD_m"] = v if v is not None else gaze["RPUPILD_m"]
                v = _f(s,"LPUPILV");     gaze["LPUPILV"]   = int(v) if v is not None else gaze["LPUPILV"]
                v = _f(s,"RPUPILV");     gaze["RPUPILV"]   = int(v) if v is not None else gaze["RPUPILV"]
                v = _f(s,"GSR");         gaze["GSR"]       = v if v is not None else gaze["GSR"]
                v = _f(s,"GSRV");        gaze["GSRV"]      = int(v) if v is not None else gaze["GSRV"]
                v = _f(s,"HR");          gaze["HR"]        = v if v is not None else gaze["HR"]
                v = _f(s,"HRV");         gaze["HRV"]       = int(v) if v is not None else gaze["HRV"]
                v = _f(s,"HRP");         gaze["HRP"]       = v if v is not None else gaze["HRP"]
                v = _f(s,"DIAL");        gaze["DIAL"]      = v if v is not None else gaze["DIAL"]
                v = _f(s,"DIALV");       gaze["DIALV"]     = int(v) if v is not None else gaze["DIALV"]

                gaze_queue.append({
                    "ts": time.time(),
                    "TIME": gaze["TIME"], "TIME_TICK": gaze["TIME_TICK"],
                    "LPOGX": gaze["LPOGX"], "LPOGY": gaze["LPOGY"], "LPOGV": gaze["LPOGV"],
                    "RPOGX": gaze["RPOGX"], "RPOGY": gaze["RPOGY"], "RPOGV": gaze["RPOGV"],
                    "BPOGX": gaze["BPOGX"], "BPOGY": gaze["BPOGY"], "BPOGV": gaze["BPOGV"],
                    "LPD_px": gaze["LPD_px"], "RPD_px": gaze["RPD_px"],
                    "LPUPILD_m": gaze["LPUPILD_m"], "RPUPILD_m": gaze["RPUPILD_m"],
                    "LPUPILV": gaze["LPUPILV"], "RPUPILV": gaze["RPUPILV"],
                    "GSR": gaze["GSR"], "GSRV": gaze["GSRV"],
                    "HR": gaze["HR"], "HRV": gaze["HRV"], "HRP": gaze["HRP"],
                    "DIAL": gaze["DIAL"], "DIALV": gaze["DIALV"]
                })
        except Exception:
            break

threading.Thread(target=gaze_loop, daemon=True).start()

# ---------- VIDEO ----------
def open_cap(path):
    for backend in (cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(path, backend)
        if cap.isOpened():
            print("Using backend:", "FFMPEG" if backend==cv2.CAP_FFMPEG else "MSMF" if backend==cv2.CAP_MSMF else "DSHOW")
            return cap
    return cv2.VideoCapture()

cap = open_cap(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
print("fps:", fps, "size:", w, "x", h)

# ---------- SPLASH / UI ----------
WIN = "Gaze Overlay"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

participant = "001"
researcher_mode = True
start_clicked = False

btn_start = btn_researcher = pid_box = btn_minus = btn_plus = None
btn_calibrate = None

def draw_button(img, rect, label, filled=True, active=True, font_scale=1.1):
    x1,y1,x2,y2 = rect
    color = (255,255,255) if active else (120,120,120)
    if filled: cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)
    else:      cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
    tx = x1 + (x2-x1 - tw)//2
    ty = y1 + (y2-y1 + th)//2
    cv2.putText(img, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0) if filled else color, 3, cv2.LINE_AA)

def format_pid(pid_str):
    s = ''.join([c for c in pid_str if c.isdigit()])[:3]
    return s.zfill(3) if s else "001"

def draw_overlay(img):
    if gaze["BPOGV"] == 1:
        bx, by = int(gaze["BPOGX"]*w), int(gaze["BPOGY"]*h)
        cv2.circle(img, (bx,by), 20, (255,0,0), 3)

def draw_pupil_hud(img):
    lp_mm = gaze["LPUPILD_m"]*1000 if gaze["LPUPILD_m"]>0 else None
    rp_mm = gaze["RPUPILD_m"]*1000 if gaze["RPUPILD_m"]>0 else None
    def fmt_mm(val): return f"{val:.2f}mm" if isinstance(val,(int,float)) else "—"
    ltxt, rtxt = fmt_mm(lp_mm), fmt_mm(rp_mm)
    bar_w, bar_h = 340, 28
    cv2.rectangle(img, (20, 20), (20+bar_w, 20+bar_h), (0,0,0), -1)
    cv2.putText(img, f"L {ltxt}   |   R {rtxt}", (28, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Valid L:{gaze['LPUPILV']} R:{gaze['RPUPILV']}", (28, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

def _draw_engagement_wheel(img, x_center, y_center, radius, dial_value, valid):
    cv2.circle(img, (x_center, y_center), radius, (255,255,255) if valid else (120,120,120), 2)
    if valid and isinstance(dial_value, (int, float)):
        angle = 225.0 - (dial_value * 270.0)
        ang_rad = np.deg2rad(angle)
        nx = int(x_center + radius * 0.85 * np.cos(ang_rad))
        ny = int(y_center - radius * 0.85 * np.sin(ang_rad))
        cv2.line(img, (x_center, y_center), (nx, ny), (255,255,255), 3)
    cv2.putText(img, "Engagement", (x_center - radius, y_center + radius + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

def draw_biometrics_hud(img):
    panel_w, panel_h = 420, 120
    x0 = img.shape[1] - panel_w - 20
    y0 = 20
    cv2.rectangle(img, (x0, y0), (x0+panel_w, y0+panel_h), (0,0,0), -1)

    hr_bpm  = gaze["HR"] if gaze["HRV"]==1 else None
    gsr_ohm = gaze["GSR"] if gaze["GSRV"]==1 else None
    hr_txt  = f"HR: {hr_bpm:.0f} bpm" if isinstance(hr_bpm,(int,float)) else "HR: —"
    gsr_txt = f"GSR: {gsr_ohm:.0f} Ω"  if isinstance(gsr_ohm,(int,float)) else "GSR: —"
    cv2.putText(img, hr_txt,  (x0+8, y0+26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255) if hr_bpm is not None else (160,160,160), 2, cv2.LINE_AA)
    cv2.putText(img, gsr_txt, (x0+8, y0+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255) if gsr_ohm is not None else (160,160,160), 2, cv2.LINE_AA)

    dial_val = gaze["DIAL"]
    dial_valid = (gaze["DIALV"] == 1) and isinstance(dial_val, (int,float))
    cx = x0 + panel_w - 70
    cy = y0 + 60
    _draw_engagement_wheel(img, cx, cy, 45, max(0.0, min(1.0, float(dial_val))) if dial_valid else 0.0, dial_valid)

def start_calibration():
    try:
        send(gp, '<SET ID="CALIBRATE_SHOW" STATE="1" />')
        send(gp, '<SET ID="CALIBRATE_START" STATE="1" />')
        print("Calibration started via OpenGaze.")
    except Exception as e:
        print("Calibration error:", e)

def draw_splash(w,h, participant, researcher_mode):
    global btn_start, btn_researcher, pid_box, btn_minus, btn_plus, btn_calibrate
    img = np.zeros((h,w,3), dtype=np.uint8)

    for x in range(0,w,80): cv2.line(img, (x,0), (x,h), (20,20,20), 1)
    for y in range(0,h,80): cv2.line(img, (0,y), (w,y), (20,20,20), 1)

    cv2.putText(img, "Eye-Tracked Playback", (int(w*0.20), int(h*0.20)),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 4, cv2.LINE_AA)

    pid_box = (int(w*0.24), int(h*0.30), int(w*0.56), int(h*0.40))
    cv2.rectangle(img, (pid_box[0],pid_box[1]), (pid_box[2],pid_box[3]), (255,255,255), 2)
    cv2.putText(img, f"Participant ID:  {participant}", (pid_box[0]+15, pid_box[3]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3, cv2.LINE_AA)

    btn_minus = (pid_box[0]-int(w*0.06), pid_box[1], pid_box[0]-10, pid_box[3])
    btn_plus  = (pid_box[2]+10, pid_box[1], pid_box[2]+int(w*0.06), pid_box[3])
    draw_button(img, btn_minus, "-", filled=False)
    draw_button(img, btn_plus,  "+", filled=False)

    btn_researcher = (int(w*0.24), int(h*0.46), int(w*0.56), int(h*0.56))
    draw_button(img, btn_researcher, f"Researcher Mode: {'ON' if researcher_mode else 'OFF'}",
                filled=True, active=True)

    btn_start = (int(w*0.24), int(h*0.66), int(w*0.52), int(h*0.82))
    btn_calibrate = (int(w*0.54), int(h*0.66), int(w*0.76), int(h*0.82))
    draw_button(img, btn_start, "Start Video", filled=True, active=True, font_scale=1.3)
    draw_button(img, btn_calibrate, "Calibrate", filled=False, active=True, font_scale=1.1)

    helper_y = int(h*0.90)
    for i, t in enumerate([
        "Type digits to set Participant ID (Backspace deletes, Up/Down change).",
        "Click buttons or press R to toggle Researcher Mode.",
        "Click Calibrate to run Gazepoint Control 9-point calibration.",
        "Press ENTER or click Start to begin. ESC or Q exits."
    ]):
        cv2.putText(img, t, (int(w*0.10), helper_y + i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)

    if researcher_mode:
        overlay = img.copy()
        draw_overlay(overlay)
        draw_pupil_hud(overlay)
        draw_biometrics_hud(overlay)
        return overlay
    else:
        return img

def pid_inc(pid, d=1):
    n = int(pid)
    n = max(0, min(999, n + d))
    return f"{n:03d}"

def on_mouse(event, x, y, flags, param):
    global start_clicked, researcher_mode, participant
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_start and btn_start[0]<=x<=btn_start[2] and btn_start[1]<=y<=btn_start[3]:
            start_clicked = True
        elif btn_calibrate and btn_calibrate[0]<=x<=btn_calibrate[2] and btn_calibrate[1]<=y<=btn_calibrate[3]:
            start_calibration()
        elif btn_researcher and btn_researcher[0]<=x<=btn_researcher[2] and btn_researcher[1]<=y<=btn_researcher[3]:
            researcher_mode = not researcher_mode
        elif btn_minus and btn_minus[0]<=x<=btn_minus[2] and btn_minus[1]<=y<=btn_minus[3]:
            participant = pid_inc(participant, -1)
        elif btn_plus and btn_plus[0]<=x<=btn_plus[2] and btn_plus[1]<=y<=btn_plus[3]:
            participant = pid_inc(participant, +1)

cv2.setMouseCallback(WIN, on_mouse)

# interactive splash
while True:
    splash = draw_splash(w, h, participant, researcher_mode)
    cv2.imshow(WIN, splash)
    k = cv2.waitKey(20) & 0xFF
    if k in (27, ord('q')):
        cap.release(); gp.close(); cv2.destroyAllWindows(); sys.exit(0)
    if k == ord('r'):
        researcher_mode = not researcher_mode
    elif k == 13:
        start_clicked = True
    elif k == 8:
        digits = ''.join([c for c in participant if c.isdigit()])[:-1]
        participant = format_pid(digits)
    elif k == 2490368:
        participant = pid_inc(participant, +1)
    elif k == 2621440:
        participant = pid_inc(participant, -1)
    elif 48 <= k <= 57:
        s = ''.join([c for c in participant if c.isdigit()])
        s = (s + chr(k))[-3:]
        participant = format_pid(s)
    if start_clicked:
        break

# ---------- RECORDING PHASE SETUP ----------
part_dir = os.path.join(DESKTOP_DIR, f"participant_{participant}")
os.makedirs(part_dir, exist_ok=True)
CSV_PATH   = os.path.join(part_dir, BASE_NAME + "_gaze.csv")
OUTPUT_MP4 = os.path.join(part_dir, BASE_NAME + "_gaze.mp4")

time.sleep(START_DELAY)
cap.release()
cap = open_cap(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_MP4, fourcc, fps, (w, h))

csv_f = open(CSV_PATH, "w", newline="", encoding="utf-8")
wr = csv.writer(csv_f)
wr.writerow([
    "PARTICIPANT","TIME_s","TIME_TICK",
    "LPOGX","LPOGY","LPOGV",
    "RPOGX","RPOGY","RPOGV",
    "BPOGX","BPOGY","BPOGV",
    "LPD_px","RPD_px","LPUPIL_mm","RPUPIL_mm","LPUPILV","RPUPILV",
    "GSR_ohm","GSRV","HR_bpm","HRV","HRP","DIAL","DIALV"
])

# --------- PLAYBACK WITH FRAME-ACCURATE TIMING ----------
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print(f"▶ Participant {participant} | Researcher Mode (display): {'ON' if researcher_mode else 'OFF'}")
print("▶ Playing and recording. ESC or Q stops early.")

gaze_queue.clear()

# Start audio first, then set t0 after brief delay for audio startup
audio_start(AUDIO_WAV)
time.sleep(0.32)  # Small delay for audio system to start
t0 = time.time()

frame_interval = 1.0 / fps
frame_number = 0

# Use video timestamp when available for better sync
use_video_pts = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Try to get actual video timestamp (more accurate than frame counting)
    video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    
    # Calculate target display time
    if use_video_pts and video_timestamp > 0:
        target_time = t0 + video_timestamp
    else:
        target_time = t0 + (frame_number * frame_interval)
    
    record_frame  = frame.copy()
    display_frame = frame.copy()

    # Drain all gaze samples to CSV
    while gaze_queue:
        s = gaze_queue.popleft()
        t_rel = s["ts"] - t0
        if t_rel < 0:
            continue

        lp_mm = s["LPUPILD_m"]*1000 if s["LPUPILD_m"]>0 else ""
        rp_mm = s["RPUPILD_m"]*1000 if s["RPUPILD_m"]>0 else ""

        wr.writerow([
            participant, round(t_rel,6), s["TIME_TICK"],
            s["LPOGX"], s["LPOGY"], s["LPOGV"],
            s["RPOGX"], s["RPOGY"], s["RPOGV"],
            s["BPOGX"], s["BPOGY"], s["BPOGV"],
            s["LPD_px"], s["RPD_px"],
            round(lp_mm,3) if lp_mm!="" else "",
            round(rp_mm,3) if rp_mm!="" else "",
            s["LPUPILV"], s["RPUPILV"],
            s["GSR"], s["GSRV"], s["HR"], s["HRV"], s["HRP"], s["DIAL"], s["DIALV"]
        ])

    # Apply overlays
    draw_overlay(record_frame)
    if researcher_mode:
        draw_overlay(display_frame)
        draw_pupil_hud(display_frame)
        draw_biometrics_hud(display_frame)

    out.write(record_frame)
    
    # Wait until target time BEFORE displaying
    now = time.time()
    sleep_time = target_time - now
    
    if sleep_time > 0:
        time.sleep(sleep_time)
    elif sleep_time < -0.1:  # More than 100ms behind
        print(f"Warning: Frame {frame_number} is {-sleep_time:.3f}s behind schedule")
    
    # Display frame at the precise moment
    cv2.imshow(WIN, display_frame)

    # Check for exit key (1ms wait to process events)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        print("⏹ Stopped early by user.")
        break
    
    frame_number += 1

# ---------- CLEANUP ----------
cap.release()
out.release()
audio_stop()
csv_f.close()
_run_gaze = False
try: gp.close()
except: pass
cv2.destroyAllWindows()

print(f"✅ Saved in {part_dir}\n  Video: {os.path.join(part_dir, BASE_NAME + '_gaze.mp4')}\n  CSV:   {os.path.join(part_dir, BASE_NAME + '_gaze.csv')}")