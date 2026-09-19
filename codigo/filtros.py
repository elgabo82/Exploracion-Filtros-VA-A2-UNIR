"""filtros.py - Filtros espaciales, operaciones morfologicas y metricas.
Actividad 2 (grupal) de Vision Artificial, UNIR. Autores: G. Morejon y W. Obregon.
Cada funcion corresponde a una ecuacion o algoritmo de la seccion de metodos."""
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim_fn
from scipy import ndimage as ndi

# ---------------------------------------------------------------- carga
def cargar(ruta):
    """Devuelve (imagen BGR, imagen en grises uint8)."""
    bgr = cv2.imread(ruta, cv2.IMREAD_COLOR)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def zona_valida(g, tol=2.0):
    """Excluye filas/columnas uniformes (margenes o separadores de paneles)."""
    col = g.std(axis=0) > tol
    fil = g.std(axis=1) > tol
    return fil[:, None] & col[None, :]

# ------------------------------------------------------ filtros espaciales
def gauss(g, sigma):
    """Suavizado gaussiano; tamano de ventana 2*ceil(3*sigma)+1."""
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(g, (k, k), sigma)

def media(g, k):
    return cv2.blur(g, (k, k))

def mediana(g, k):
    return cv2.medianBlur(g, k)

def unsharp(g, sigma, alpha):
    """g + alpha * (g - G_sigma * g)."""
    k = 2 * int(np.ceil(3 * sigma)) + 1
    f = g.astype(np.float32)
    s = cv2.GaussianBlur(f, (k, k), sigma)
    return np.clip(f + alpha * (f - s), 0, 255).astype(np.uint8)

def laplaciano(g):
    """Realce laplaciano: g - Laplaciano(g)."""
    f = g.astype(np.float32)
    return np.clip(f - cv2.Laplacian(f, cv2.CV_32F, ksize=3), 0, 255).astype(np.uint8)

def sobel(g):
    """Magnitud del gradiente de Sobel normalizada a [0, 255]."""
    f = g.astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
    return cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def canny(g, t1=50, t2=150):
    return cv2.Canny(g, t1, t2)

# ---------------------------------------------- elementos estructurales
def ee(forma, r):
    """Elemento estructural de radio r (lado 2r+1): disco, cuadrado o cruz."""
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    if forma == "disco":
        m = x ** 2 + y ** 2 <= r ** 2 + 0.5
    elif forma == "cuadrado":
        m = np.ones_like(x, bool)
    else:
        m = (x == 0) | (y == 0)
    return m.astype(np.uint8)

def _m(img, op, forma, r):
    return cv2.morphologyEx(img, op, ee(forma, r))

def erosion(i, r, f="disco"): return _m(i, cv2.MORPH_ERODE, f, r)
def dilatacion(i, r, f="disco"): return _m(i, cv2.MORPH_DILATE, f, r)
def apertura(i, r, f="disco"): return _m(i, cv2.MORPH_OPEN, f, r)
def cierre(i, r, f="disco"): return _m(i, cv2.MORPH_CLOSE, f, r)
def gradiente(i, r, f="disco"): return _m(i, cv2.MORPH_GRADIENT, f, r)
def tophat(i, r, f="disco"): return _m(i, cv2.MORPH_TOPHAT, f, r)
def blackhat(i, r, f="disco"): return _m(i, cv2.MORPH_BLACKHAT, f, r)

# ------------------------------------------------------- binarizacion
def otsu(g, valida=None, oscuro=True):
    """Umbral de Otsu calculado solo sobre la zona valida. Si oscuro=True,
    las estructuras de interes son las mas oscuras (se invierte la mascara)."""
    v = g[valida] if valida is not None else g.ravel()
    t = threshold_otsu(v)
    m = (g <= t) if oscuro else (g > t)
    if valida is not None:
        m &= valida
    return (m * 255).astype(np.uint8), t

def secuencia(m, tipo, r_ap, r_ci):
    """Secuencias morfologicas evaluadas sobre una mascara binaria."""
    if tipo == "ninguna": return m
    if tipo == "apertura": return apertura(m, r_ap)
    if tipo == "cierre": return cierre(m, r_ci)
    if tipo == "ap>ci": return cierre(apertura(m, r_ap), r_ci)
    if tipo == "ci>ap": return apertura(cierre(m, r_ci), r_ap)
    raise ValueError(tipo)

# ------------------------------------------------------------ metricas
def ruido(g):
    """Estimador de ruido de Immerkaer (1996) basado en un nucleo laplaciano."""
    N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    c = cv2.filter2D(g.astype(np.float32), -1, N)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(c).sum() / (6.0 * (w - 2) * (h - 2)))

def nitidez(g):
    """Varianza del laplaciano (Pech-Pacheco et al., 2000)."""
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def contraste_rms(g):
    """Desviacion tipica de la intensidad normalizada a [0, 1]."""
    return float((g.astype(np.float64) / 255.0).std())

def contraste_local(g, k=15):
    """Media de la desviacion tipica local (ventana k x k), normalizada a [0, 1]."""
    f = g.astype(np.float32) / 255.0
    m = cv2.blur(f, (k, k)); m2 = cv2.blur(f * f, (k, k))
    return float(np.sqrt(np.maximum(m2 - m * m, 0)).mean())

def entropia(g):
    p = np.bincount(g.ravel(), minlength=256) / g.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def epi(ref, filt):
    """Indice de conservacion de bordes: correlacion de los laplacianos."""
    a = cv2.Laplacian(ref.astype(np.float32), cv2.CV_32F).ravel()
    b = cv2.Laplacian(filt.astype(np.float32), cv2.CV_32F).ravel()
    a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))

def ssim(a, b): return float(ssim_fn(a, b, data_range=255))

def borde_p95(g):
    """Percentil 95 del gradiente (fuerza de los bordes principales)."""
    f = g.astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
    return float(np.percentile(m, 95))

def dens_canny(g): return float((canny(g) > 0).mean())

def n_comp(m): return int(cv2.connectedComponents((m > 0).astype(np.uint8))[0] - 1)

def frac_pequenas(m, frac=0.001):
    """Fraccion de componentes con area menor que frac del encuadre."""
    n, _, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    a = st[1:, cv2.CC_STAT_AREA]
    return float((a < frac * m.size).mean()) if len(a) else 0.0

def dice(a, b):
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0

def prf(det, gt, tol=2):
    """Precision, exhaustividad y F1 de un mapa de bordes con tolerancia tol (px)."""
    d, g = det > 0, gt > 0
    k = ee("disco", tol)
    g_t = cv2.dilate(g.astype(np.uint8), k) > 0
    d_t = cv2.dilate(d.astype(np.uint8), k) > 0
    p = (d & g_t).sum() / max(d.sum(), 1)
    r = (g & d_t).sum() / max(g.sum(), 1)
    return float(p), float(r), float(2 * p * r / (p + r + 1e-12))
