import cv2
import pytesseract
from flask import Flask, render_template, request, send_file
from PIL import Image
import spacy
import re
import os
import time
import numpy as np
from pdf2image import convert_from_path
from docx import Document

# -------------------------------
# LOAD AI MODEL
# -------------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# PATH SETTINGS
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\dharini\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ["PATH"]

# -------------------------------
# 🔥 FACE DETECTION
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def mask_faces(img, mode="black"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        if mode == "blur":
            face = img[y:y+h, x:x+w]
            face = cv2.GaussianBlur(face, (51, 51), 30)
            img[y:y+h, x:x+w] = face
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)

    return img

# -------------------------------
# 🔥 NEW: MERGE DIGITS (AADHAAR FIX)
# -------------------------------
def merge_digits(words):
    merged = []
    temp = ""
    coords = None

    for w in words:
        if re.fullmatch(r"\d+", w["text"]):
            if temp == "":
                temp = w["text"]
                coords = w.copy()
            else:
                temp += w["text"]
                coords["w"] = (w["x"] + w["w"]) - coords["x"]
                coords["h"] = max(coords["h"], w["h"])
        else:
            if temp != "":
                coords["text"] = temp
                merged.append(coords)
                temp = ""
            merged.append(w)

    if temp != "":
        coords["text"] = temp
        merged.append(coords)

    return merged

# -------------------------------
# FLASK SETUP
# -------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# MASKING FUNCTIONS
# -------------------------------
def mask_text(text, label):
    if label == "PERSON":
        return "XXXX"
    elif label == "EMAIL":
        parts = text.split("@")
        return parts[0][:2] + "***@" + parts[1] if len(parts) > 1 else "****"
    elif label == "PHONE":
        return text[:3] + "****" + text[-3:]
    elif label == "AADHAAR":
        return text[:4] + " **** ****"
    elif label == "PAN":
        return text[:3] + "****" + text[-1]
    elif label == "ACCOUNT":
        return text[:2] + "******" + text[-2:]
    elif label == "ORG":
        return "*****"
    return text

# -------------------------------
# COMMON DETECTION
# -------------------------------
def detect_common(text):
    if re.fullmatch(r"\d{10}", text):
        return "PHONE"
    elif re.fullmatch(r"\d{4}\s?\d{4}\s?\d{4}", text):
        return "AADHAAR"
    elif re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", text):
        return "PAN"
    elif re.fullmatch(r"\d{9,18}", text):
        return "ACCOUNT"
    elif "@" in text:
        return "EMAIL"
    return None

# -------------------------------
# AMOUNT DETECTION
# -------------------------------
def is_amount(text):
    return bool(re.fullmatch(r"\d{1,3}(,\d{3})*(\.\d+)?", text))

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bank')
def bank():
    return render_template('bank.html')

@app.route('/healthcare')
def healthcare():
    return render_template('healthcare.html')

@app.route('/corporate')
def corporate():
    return render_template('corporate.html')

@app.route('/government')
def government():
    return render_template('government.html')

# -------------------------------
# MAIN PROCESS
# -------------------------------
@app.route('/upload/<industry>', methods=['POST'])
def upload(industry):

    file = request.files['file']

    if file.filename == "":
        return "No file selected"

    filename = file.filename.lower()
    images = []

    # IMAGE
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        images.append(img)

    # PDF
    elif filename.endswith('.pdf'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            pages = convert_from_path(filepath, poppler_path=POPPLER_PATH)
        except Exception as e:
            return f"PDF Error: {str(e)}"

        for page in pages:
            img = np.array(page)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)

    # WORD
    elif filename.endswith('.docx'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        doc = Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])

        img = np.ones((900, 1400, 3), dtype=np.uint8) * 255

        y = 40
        for line in text.split("\n"):
            cv2.putText(img, line, (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)
            y += 30

        images.append(img)

    else:
        return "Unsupported file type"

    # -------------------------------
    # PROCESS
    # -------------------------------
    processed_images = []

    for img in images:

        if img is None:
            continue

        img = mask_faces(img, mode="black")

        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6"
        )

        # GROUP BY LINE
        lines = {}

        for i in range(len(data['text'])):
            txt = data['text'][i].strip()

            if txt == "":
                continue

            line_id = (data['block_num'][i], data['line_num'][i])

            if line_id not in lines:
                lines[line_id] = []

            lines[line_id].append({
                "text": txt,
                "x": data['left'][i],
                "y": data['top'][i],
                "w": data['width'][i],
                "h": data['height'][i]
            })

        # PROCESS TEXT
        for line in lines.values():

            # 🔥 FIX APPLIED HERE
            line = merge_digits(line)

            for word in line:

                text = word["text"]
                x, y, w, h = word["x"], word["y"], word["w"], word["h"]

                label = None

                if is_amount(text):
                    continue

                if industry == "bank":
                    if re.fullmatch(r"\d{9,18}", text):
                        label = "ACCOUNT"

                elif industry == "government":
                    if re.fullmatch(r"\d{12}", text):
                        label = "AADHAAR"
                    elif re.fullmatch(r"\d{4}\s?\d{4}\s?\d{4}", text):
                        label = "AADHAAR"
                    elif re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", text):
                        label = "PAN"

                common = detect_common(text)
                if common:
                    label = common

                doc_nlp = nlp(text)
                for ent in doc_nlp.ents:
                    if ent.label_ in ["PERSON", "ORG"]:
                        label = ent.label_

                if label:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

                    masked = mask_text(text, label)
                    cv2.putText(img, masked, (x, y + h),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)

        processed_images.append(img)

    # SAVE OUTPUT
    filename = f"output_{int(time.time())}.pdf"
    pdf_path = os.path.join(OUTPUT_FOLDER, filename)

    pil_images = []
    for img in processed_images:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_images.append(pil_img)

    pil_images[0].save(pdf_path, save_all=True, append_images=pil_images[1:])

    return send_file(pdf_path, as_attachment=True)

# RUN
if __name__ == "__main__":
    app.run(debug=True)