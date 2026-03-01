"""
STEP 1 - BUILD FACE DATABASE
=============================
Reads student folders → extracts ArcFace embeddings for each image →
averages all embeddings per student → stores in ChromaDB.

Your folder structure:
    /home/apurva/Downloads/face_embeddings/data/students/
        nithin/
            labels/   ← skipped automatically (no images)
            nithin/   ← actual face images (.jpg/.png etc.)
        student2/
            labels/
            student2/
        ...
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import chromadb

# ─────────────────────────────────────────────
# CONFIG — edit only these if paths change
# ─────────────────────────────────────────────
STUDENTS_DIR = "/home/apurva/Downloads/face_embeddings/data/students"
CHROMA_DB_DIR = "/home/apurva/Downloads/face_embeddings/chroma_db"
COLLECTION_NAME = "student_faces"
# ─────────────────────────────────────────────


def load_insightface_model():
    print("Loading InsightFace ArcFace model...")
    print("(First run will download ~300MB — please wait)\n")
    app = FaceAnalysis(
        name="buffalo_l",                        # ArcFace R100 — most accurate
        providers=["CPUExecutionProvider"]        # change to CUDAExecutionProvider if you have GPU
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✓ Model loaded!\n")
    return app


def get_embedding(app, image_path):
    """
    Read an image, detect face, return 512-d normalized ArcFace embedding.
    Returns None if image can't be read or no face is found.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"    [WARN] Could not read image: {os.path.basename(image_path)}")
        return None

    faces = app.get(img)

    if len(faces) == 0:
        print(f"    [WARN] No face detected in: {os.path.basename(image_path)}")
        return None

    if len(faces) > 1:
        # If multiple faces, pick the largest (most prominent)
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )

    return faces[0].normed_embedding  # already L2-normalized, shape: (512,)


def collect_image_paths(student_path):
    """
    Handles two folder structures:
      1. Images directly in student_path/
      2. Images in a subfolder (your case: students/nithin/nithin/*.jpg)
         Labels/ and other non-image subfolders are automatically skipped.
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = []

    # Check if images are directly inside student_path
    direct_images = [
        f for f in os.listdir(student_path)
        if f.lower().endswith(image_extensions)
    ]

    if direct_images:
        image_files = [os.path.join(student_path, f) for f in direct_images]
    else:
        # Look one level deeper — skip subfolders with no images (like labels/)
        for subfolder in sorted(os.listdir(student_path)):
            subfolder_path = os.path.join(student_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            imgs = [
                f for f in os.listdir(subfolder_path)
                if f.lower().endswith(image_extensions)
            ]
            if imgs:
                print(f"    Found {len(imgs)} images in: {subfolder}/")
                image_files += [os.path.join(subfolder_path, f) for f in sorted(imgs)]
            else:
                print(f"    Skipping (no images): {subfolder}/")

    return image_files


def process_student(app, student_path):
    """
    Extract embedding from each image, then average them all.
    Returns (averaged_embedding, number_of_frames_used) or (None, 0).
    """
    image_files = collect_image_paths(student_path)

    if not image_files:
        print(f"    [WARN] No images found in {student_path}")
        return None, 0

    print(f"    Processing {len(image_files)} images...")

    embeddings = []
    for img_path in image_files:
        emb = get_embedding(app, img_path)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        print(f"    [WARN] Could not extract any face embeddings.")
        return None, 0

    print(f"    Successfully embedded {len(embeddings)}/{len(image_files)} images.")

    # ── CORE: Average all embeddings into one representative vector ──
    avg_embedding = np.mean(embeddings, axis=0)

    # Re-normalize after averaging (important for cosine similarity to work correctly)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    return avg_embedding, len(embeddings)


def build_database():
    # ── Load ArcFace model ──
    app = load_insightface_model()

    # ── Connect to ChromaDB (auto-creates the folder) ──
    print(f"Connecting to ChromaDB at: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete existing collection so we rebuild fresh each time
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection — rebuilding fresh.\n")
    except Exception:
        print("No existing collection found — creating new one.\n")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for face matching
    )

    # ── Find all student folders ──
    student_folders = sorted([
        d for d in os.listdir(STUDENTS_DIR)
        if os.path.isdir(os.path.join(STUDENTS_DIR, d))
    ])

    if not student_folders:
        print(f"\n[ERROR] No student folders found in:\n  {STUDENTS_DIR}")
        print("Make sure your images are placed correctly.")
        return

    print(f"Found {len(student_folders)} student(s): {student_folders}\n")
    print("=" * 55)

    success_count = 0

    for student_name in tqdm(student_folders, desc="Overall progress"):
        student_path = os.path.join(STUDENTS_DIR, student_name)
        print(f"\n→ Student: {student_name}")

        avg_embedding, num_frames = process_student(app, student_path)

        if avg_embedding is None:
            print(f"  [SKIP] {student_name} — no valid embeddings found.")
            continue

        # ── Store averaged embedding in ChromaDB ──
        collection.add(
            ids=[student_name],                        # unique ID = folder name
            embeddings=[avg_embedding.tolist()],       # 512-d averaged embedding
            metadatas=[{
                "name": student_name,
                "frames_used": num_frames
            }]
        )

        print(f"  ✓ Stored '{student_name}' using {num_frames} frame(s).")
        success_count += 1

    print("\n" + "=" * 55)
    print(f"✓ Database built: {success_count}/{len(student_folders)} students stored.")
    print(f"✓ ChromaDB location: {os.path.abspath(CHROMA_DB_DIR)}")


if __name__ == "__main__":
    build_database()