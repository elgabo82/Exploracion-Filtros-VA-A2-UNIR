"""fantomas.py - Imagenes sinteticas con verdad de terreno conocida.
Tres casos: (A) dos discos casi tangentes, (B) disco fragmentado por
fisuras finas, (C) estructura lineal oscura y delgada. Ruido tipo speckle
(multiplicativo Gamma) + ruido gaussiano aditivo, seguido de desenfoque leve."""
import cv2
import numpy as np

N, FONDO, OBJ = 200, 150, 55

def _ruido(img, rng, L=3, sg=5.0):
    """speckle: img * Gamma(L, 1/L); mas gaussiano aditivo; desenfoque leve."""
    f = img.astype(np.float32) * rng.gamma(L, 1.0 / L, img.shape).astype(np.float32)
    f += rng.normal(0, sg, img.shape).astype(np.float32)
    f = cv2.GaussianBlur(f, (5, 5), 1.0)
    return np.clip(f, 0, 255).astype(np.uint8)

def caso_A(rng):
    """Dos discos (r=22) unidos por un puente de 6 px x 16 px. GT: 2 discos."""
    gt = np.zeros((N, N), np.uint8)
    cv2.circle(gt, (62, 100), 22, 255, -1)
    cv2.circle(gt, (138, 100), 22, 255, -1)
    obj = gt.copy()
    cv2.rectangle(obj, (80, 97), (120, 102), 255, -1)   # puente espurio
    img = np.where(obj > 0, OBJ, FONDO).astype(np.uint8)
    return _ruido(img, rng), gt

def caso_B(rng):
    """Disco (r=45) atravesado por 3 fisuras claras de 2 px. GT: 1 objeto."""
    gt = np.zeros((N, N), np.uint8)
    cv2.circle(gt, (100, 100), 45, 255, -1)
    img = np.where(gt > 0, OBJ, FONDO).astype(np.uint8)
    for x0, y0, x1, y1 in [(55, 80, 145, 85), (60, 115, 140, 108), (98, 55, 103, 145)]:
        cv2.line(img, (x0, y0), (x1, y1), FONDO, 2)
    return _ruido(img, rng), gt

def caso_C(rng):
    """Fisura oscura de 3 px de ancho sobre fondo claro. GT: la linea."""
    gt = np.zeros((N, N), np.uint8)
    pts = np.array([[10, 60], [60, 90], [110, 110], [160, 100], [190, 140]], np.int32)
    cv2.polylines(gt, [pts], False, 255, 3)
    img = np.where(gt > 0, OBJ, FONDO).astype(np.uint8)
    return _ruido(img, rng), gt

CASOS = {"A": caso_A, "B": caso_B, "C": caso_C}
