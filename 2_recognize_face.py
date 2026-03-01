import os
import cv2
import numpy as np
import argparse
from insightface.app import FaceAnalysis
import chromadb

CHROMA_DB_DIR = "/home/apurva/Downloads/face_embeddings/chroma_db"
COLLECTION_NAME = "student_faces"
SIMILARITY_THRESHOLD = 0.45

def load_model():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def extract_embedding(app, img):
    faces = app.get(img)
    if not faces:
        return None, None
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding, faces[0].bbox

def query_database(collection, embedding):
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=1)
    name = results["metadatas"][0][0]["name"]
    distance = results["distances"][0][0]
    return name, distance

def draw_result(img, bbox, name, distance, recognized):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if recognized:
        label = f"{name}  {(1 - distance) * 100:.1f}%"
        color = (0, 220, 0)
    else:
        label = f"Unknown  (dist: {distance:.2f})"
        color = (0, 0, 220)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return img

def recognize_from_image(app, collection, image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return
    embedding, bbox = extract_embedding(app, img)
    if embedding is None:
        print("[WARN] No face detected in the image.")
        return
    name, distance = query_database(collection, embedding)
    recognized = distance < SIMILARITY_THRESHOLD
    if recognized:
        print(f"[MATCH]   Student   : {name}")
        print(f"          Confidence: {(1 - distance) * 100:.1f}%")
        print(f"          Distance  : {distance:.4f}")
    else:
        print(f"[UNKNOWN] No match found.")
        print(f"          Closest   : {name}")
        print(f"          Distance  : {distance:.4f}  (threshold: {SIMILARITY_THRESHOLD})")
    img = draw_result(img, bbox, name, distance, recognized)
    cv2.imshow("Face Recognition Result — press any key to close", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_from_webcam(app, collection):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    print("Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        embedding, bbox = extract_embedding(app, frame)
        if embedding is not None and bbox is not None:
            name, distance = query_database(collection, embedding)
            recognized = distance < SIMILARITY_THRESHOLD
            frame = draw_result(frame, bbox, name, distance, recognized)
        else:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        cv2.putText(frame, f"DB: {collection.count()} students", (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Face Recognition — Live  (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Face Recognition using ArcFace + ChromaDB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to image file for recognition")
    group.add_argument("--webcam", action="store_true", help="Use webcam for real-time recognition")
    args = parser.parse_args()

    app = load_model()

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"\n[ERROR] Collection '{COLLECTION_NAME}' not found. Run 1_build_database.py first!\n")
        return

    print(f"Database loaded — {collection.count()} student(s) registered.\n")

    if args.image:
        recognize_from_image(app, collection, args.image)
    elif args.webcam:
        recognize_from_webcam(app, collection)

if __name__ == "__main__":
    main()