# Save as ~/lerobot/record_wrist_cam.py
import cv2
import time
from datetime import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path  = f'/media/precag/ARM101/eval_data/wrist_raw_{timestamp}.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, 30, (640, 480))

print(f'Recording to: {out_path}')
print('Press Ctrl+C to stop')

start = time.time()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Apply same rotation as rollout
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        writer.write(frame)
        elapsed = time.time() - start
        if int(elapsed) % 10 == 0:
            print(f'  Recording: {elapsed:.0f}s', end='\r')
except KeyboardInterrupt:
    pass

cap.release()
writer.release()
print(f'\nSaved: {out_path}  ({time.time()-start:.0f}s)')
