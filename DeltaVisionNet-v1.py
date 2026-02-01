# Welcome to the birdhouse of the DeltaVisionNet algorithm
# Note I am still working on this algorithm and some parts aren't visible

import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

class NoseRbdetector:
    def __init__(self, rub_factor=0.38, gap_frames=8):
        base_face = python.BaseOptions(model_asset_path="face_landmarker.task")
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base_face,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
        )
        base_hands = python.BaseOptions(model_asset_path="hand_landmarker.task")
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base_hands, num_hands=2)
        )
        self.rub_factor = rub_factor
        self.gap_frames = gap_frames
        self.face_width = None
        self.rub_threshold = None
        self.state = "IDLE"

        self.farr_count = 0
        self._rub_start_frame = None
    def _dist(self, a, b):
        return math.dist(a, b)
    def _update_face_width(self, face_landmarks, w, h):
        lm = face_landmarks[0]
        left_eye = lm[33]
        right_eye = lm[263]
        left = (left_eye.x * w, left_eye.y * h)
        right = (right_eye.x * w, right_eye.y * h)
        self.face_width = self._dist(left, right)
        self.rub_threshold = self.rub_factor * self.face_width
    def process(self, frame, frame_idx=None):
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_res = self.face.detect(mp_image)
        if not face_res.face_landmarks:
            return None
        self._update_face_width(face_res.face_landmarks, w, h)
        nose_lm = face_res.face_landmarks[0][1]
        nose = (nose_lm.x * w, nose_lm.y * h)

        hands_res = self.hands.detect(mp_image)
        if not hands_res.hand_landmarks:
            return self._handle_farr(frame_idx=frame_idx)
        tips = [(hand[8].x * w, hand[8].y * h) for hand in hands_res.hand_landmarks]
        tip = min(tips, key=lambda t: self._dist(nose, t))
        dist = self._dist(nose, tip)
        if dist < self.rub_threshold:
            return self._handle_close(frame_idx=frame_idx)
        return self._handle_farr(frame_idx=frame_idx)
    def _handle_close(self, frame_idx=None):
        if self.state == "IDLE":
            self.state = "RUBBING"
            self.farr_count = 0
            self._rub_start_frame = frame_idx
            return {"event": "start", "frame": frame_idx}
        self.farr_count = 0
        return None
    def _handle_farr(self, frame_idx=None):
        if self.state == "RUBBING":
            self.farr_count += 1
            if self.farr_count >= self.gap_frames:
                self.state = "IDLE"

                self.farr_count = 0
                if self._rub_start_frame is not None and frame_idx is not None:
                    time_frames = frame_idx - self._rub_start_frame
                else:
                    time_frames = None
                self._rub_start_frame = None
                return {"event": "end", "frame": frame_idx, "time_frames": time_frames}
        return None
def main():
    cap = cv2.VideoCapture("WIN_20260104_01_06_48_Pro.mp4")
    detector = NoseRbdetector()
    rub_count = 0
    frame_idx = 0
    rub_events = []
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        ret, frame = cap.retrieve()
        if not ret:
            break
        event = detector.process(frame, frame_idx=frame_idx)
        if event and isinstance(event, dict):
            if event.get("event") == "start":
                rub_count += 1
                rub_events.append({
                    "count": rub_count,
                    "start_frame": event.get("frame"),
                    "end_frame": None,

                    "time_frames": None,
                    "time_s": None,
                })
                print(f"Rub Count = {rub_count} (frame {event.get('frame')})")
            elif event.get("event") == "end":

                time_frames = event.get("time_frames")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                time_s = (float(time_frames) / fps) if time_frames else 0.0
                for ev in reversed(rub_events):
                    if ev.get("end_frame") is None:
                        ev["end_frame"] = event.get("frame")
                        ev["time_frames"] = time_frames
                        ev["time_s"] = time_s
                        break
                print(f"Rub ended. Count = {rub_count}, time = {time_s:.2f}s (frames={time_frames})")
        frame_idx += 1
    cap.release()
    if rub_events and rub_events[-1].get("end_frame") is None:
        last = rub_events[-1]
        start = last.get("start_frame")
        end_frame = max(0, frame_idx - 1)
        time_frames = end_frame - (start or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        time_s = (float(time_frames) / fps) if time_frames is not None else 0.0
        last["end_frame"] = end_frame

        last["time_frames"] = time_frames
        last["time_s"] = time_s
    if rub_events:
        print("Rub events summary:")
        for ev in rub_events:
            c = ev.get("count")
            s = ev.get("start_frame")
            e = ev.get("end_frame")
            df = ev.get("time_frames")

            ds = ev.get("time_s")
            if e is None:
                print(f"  {c}: start={s}, (no end detected)")
            else:
                print(f"  {c}: start={s} end={e} time={ds:.2f}s(frames={df})")
    try:
        import json as _json
        out_path = Path(__file__).parent.joinpath("rub_events.json")
        
        with open(out_path, "w", encoding="utf-8") as _fh:
            _json.dump(rub_events, _fh, indent=2)
        print(f"This is saved in {out_path}")
    except Exception:
        pass
    print("Total rubss=", rub_count)
if __name__ == "__main__":
    main()



