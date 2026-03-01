# FaceVault 🎯
### Student Face Recognition using ArcFace + ChromaDB

A lightweight, offline face recognition pipeline that registers students from multi-angle images and identifies them in real time or from static photos — no cloud, no GPU required.

---

## How It Works

Traditional face recognition fails when a person turns their head slightly. FaceVault solves this by capturing **10 images per student from different angles**, extracting a 512-dimensional ArcFace embedding from each, and **averaging them into a single robust representation**.

```
10 images × different angles
        ↓
ArcFace extracts 512-d embedding per image
        ↓
All embeddings averaged → 1 mean vector per student
        ↓
Re-normalized and stored in ChromaDB
        ↓
Recognition = cosine similarity against stored vectors
```

This "mean embedding" acts as a centroid in embedding space, making recognition robust to pose variation, lighting changes, and partial occlusion.

---

## Project Structure

```
facevault/
├── data/
│   └── students/
│       ├── student_a/
│       │   └── student_a/       ← face images (.jpg / .png)
│       └── student_b/
│           └── student_b/
├── chroma_db/                   ← auto-created on first run
├── 1_build_database.py          ← register students into DB
├── 2_recognize_face.py          ← recognize from image or webcam
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AP0707/FaceVault.git
cd FaceVault
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Build the database

Add student images to `data/students/STUDENT_NAME/STUDENT_NAME/` then run:

```bash
python3 1_build_database.py
```

What happens:
- ArcFace `buffalo_l` model downloads automatically (~300MB, first run only)
- Each student's images are processed and embedded
- Averaged embeddings are stored in ChromaDB

Example output:
```
Found 2 student(s)
→ Processing: student_a
  Found 10 images | Embedded 10/10
  ✓ Stored using 10 frames
→ Processing: student_b
  Found 10 images | Embedded 9/10
  ✓ Stored using 9 frames
Database built: 2/2 students stored.
```

### Step 2 — Recognize a face

**From an image file:**
```bash
python3 2_recognize_face.py --image /path/to/photo.jpg
```

**From webcam (real-time):**
```bash
python3 2_recognize_face.py --webcam
```

---

## Configuration

| Variable | File | Description | Default |
|---|---|---|---|
| `STUDENTS_DIR` | both | Path to student image folders | `data/students` |
| `CHROMA_DB_DIR` | both | ChromaDB storage path | `chroma_db` |
| `SIMILARITY_THRESHOLD` | `2_recognize_face.py` | Cosine distance cutoff | `0.45` |

**Tuning the threshold:**
- Lower (e.g. `0.3`) → stricter, fewer false positives
- Higher (e.g. `0.5`) → more lenient, fewer unknowns
- Start at `0.45` and adjust based on your results

---

## Tech Stack

| Component | Tool |
|---|---|
| Face Detection | InsightFace `buffalo_l` |
| Face Embedding | ArcFace R100 — 512-d vectors |
| Vector Database | ChromaDB — cosine similarity |
| Image Processing | OpenCV |
| Language | Python 3.10+ |

---

## Requirements

- Python 3.10+
- Ubuntu 20.04+ (tested on Ubuntu 24)
- No GPU required — runs fully on CPU

---

## Why Average Embeddings?

| Approach | Weakness |
|---|---|
| Single image per student | Fails if pose/lighting differs at recognition time |
| Multiple images stored separately | Slower lookup, redundant comparisons |
| **Averaged embedding (FaceVault)** | ✅ Robust centroid, fast single comparison |

Averaging creates a mean face representation that is more tolerant of real-world variation than any single image alone.

---

## License

MIT