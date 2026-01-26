import cv2
import numpy as np
import random


def log(msg):
    print(f"      ↳ {msg}")


# -------------------------------
# Bounding box helpers
# -------------------------------

def _yolo_to_corners(label, w, h):
    cls, xc, yc, bw, bh = label
    xc, yc, bw, bh = xc * w, yc * h, bw * w, bh * h
    x1, y1 = xc - bw / 2, yc - bh / 2
    x2, y2 = xc + bw / 2, yc + bh / 2
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32
    )
    return cls, corners


def _corners_to_yolo(cls, pts, w, h):
    x = pts[:, 0]
    y = pts[:, 1]

    x1, x2 = np.clip([x.min(), x.max()], 0, w)
    y1, y2 = np.clip([y.min(), y.max()], 0, h)

    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return [cls, xc, yc, bw, bh]


# -------------------------------
# Augmentations
# -------------------------------

def _perspective(image, labels, max_shift=0.12):
    log("Perspective distortion")
    h, w = image.shape[:2]

    sw, sh = int(w * max_shift), int(h * max_shift)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, sw), random.randint(0, sh)],
        [w - random.randint(0, sw), random.randint(0, sh)],
        [w - random.randint(0, sw), h - random.randint(0, sh)],
        [random.randint(0, sw), h - random.randint(0, sh)],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(image, M, (w, h))

    new_labels = []
    for lbl in labels:
        cls, corners = _yolo_to_corners(lbl, w, h)
        ones = np.ones((4, 1))
        pts = np.hstack([corners, ones])
        warped_pts = (M @ pts.T).T
        warped_pts = warped_pts[:, :2] / warped_pts[:, 2:3]
        yolo = _corners_to_yolo(cls, warped_pts, w, h)
        if yolo:
            new_labels.append(yolo)

    return warped_img, new_labels


def _illumination(image):
    log("Illumination gradient")
    h, w = image.shape[:2]
    strength = random.uniform(0.2, 0.4)
    grad = np.linspace(1 - strength, 1 + strength, w)
    mask = np.tile(grad, (h, 1))

    out = image.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= mask

    return np.clip(out, 0, 255).astype(np.uint8)


def _blur(image):
    log("Gaussian blur")
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (k, k), 0)


def _brightness_contrast(image):
    log("Brightness / contrast jitter")
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-20, 20)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def _jpeg(image):
    log("JPEG compression")
    q = random.randint(40, 80)
    _, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _noise(image):
    log("Gaussian noise (optional)")
    if random.random() < 0.5:
        return image
    noise = np.random.normal(0, 6, image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------------
# Main pipeline
# -------------------------------

def augment_image_and_labels(image, labels):
    image, labels = _perspective(image, labels)
    image = _illumination(image)
    image = _blur(image)
    image = _brightness_contrast(image)
    image = _jpeg(image)
    image = _noise(image)
    return image, labels
