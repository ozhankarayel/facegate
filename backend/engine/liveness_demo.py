import collections

import cv2

from liveness import EarExtractor, detect_blink

cap = cv2.VideoCapture(0)
window = collections.deque(maxlen=15)

with EarExtractor() as ex:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        window.append(ex.ear(frame))
        info = detect_blink(list(window))
        ear = window[-1]
        color = (0, 255, 0) if info["blink"] else (0, 0, 255)
        cv2.putText(frame, f"EAR: {ear:.3f}" if ear else "EAR: -", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "BLINK" if info["blink"] else "no blink", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()