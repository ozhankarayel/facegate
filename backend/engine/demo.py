import cv2
from pipeline import FaceEngine, best_match

engine = FaceEngine(model_name="buffalo_s")
gallery = {}
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    faces = engine.extract(frame)
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        name, score = best_match(f["embedding"], gallery)
        label = f"{name or 'Bilinmiyor'} {score:.2f}" if gallery else "Kayitli kisi yok"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("FaceGate demo", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("e") and len(faces) == 1:
        name = input("Isim: ")
        gallery[name] = faces[0]["embedding"]
        print(f"{name} kaydedildi.")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()