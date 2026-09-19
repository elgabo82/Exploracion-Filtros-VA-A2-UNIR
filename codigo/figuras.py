"""figuras.py - Genera las figuras del informe (requiere haber ejecutado experimentos.py)."""
import os
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from filtros import *
from fantomas import CASOS
from experimentos import SIG, SIG_U, ALFA, R_AP, R_CI, RAIZ
from skimage.filters import threshold_otsu

FIG = os.path.join(RAIZ, "resultados", "figuras"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.5})
W = 6.3  # ancho de pagina en pulgadas

def rejilla(filas, cols, alto, nombre, titulos, imgs, filas_rot=None, ancho=W):
    fig, ax = plt.subplots(filas, cols, figsize=(ancho, alto), squeeze=False)
    for i in range(filas):
        for j in range(cols):
            a = ax[i][j]; im = imgs[i][j]
            a.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255 if im.ndim == 2 else None)
            a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(titulos[j], fontsize=6.5, pad=2)
            if j == 0 and filas_rot: a.set_ylabel(filas_rot[i], fontsize=6.5)
    fig.subplots_adjust(wspace=0.03, hspace=0.05, left=0.03 if filas_rot else 0.01, right=0.995, top=0.90 if titulos else 0.99, bottom=0.01)
    fig.savefig(f"{FIG}/{nombre}.png", dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

def contorno(g, m, r=1):
    """Superpone en rojo el gradiente morfologico de la mascara."""
    c = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB); c[gradiente(m, r) > 0] = (230, 30, 30); return c

def panel_708():
    g = cargar(f"{RAIZ}/originales/tiroides/tiroides_708.png")[1]
    return g[:, 279:512]

def mascara(g, dom, unsh=False):
    s = gauss(g, SIG[dom]); s = unsharp(s, SIG_U, ALFA) if unsh else s
    return otsu(s, zona_valida(g), dom == "tiroides")[0]

def f1_espaciales():
    t = panel_708(); c = cargar(f"{RAIZ}/originales/clima/clima_07_pasterze_glacier.jpg")[1][120:420, 120:560]
    filas, imgs = ["Tiroides (US)", "Glaciar"], []
    for g, dom in ((t, "tiroides"), (c, "clima")):
        s = gauss(g, SIG[dom]); u = unsharp(s, SIG_U, ALFA)
        imgs.append([g, media(g, 3), mediana(g, 3), s, u, sobel(u), canny(u)])
    rejilla(2, 7, 1.72, "f1_espaciales", ["Original", "Media 3×3", "Mediana 3×3", "Gauss. $\\sigma^*$", "Gauss. + unsharp", "Sobel", "Canny"], imgs, filas)

def f2_histogramas():
    t = panel_708(); c = cargar(f"{RAIZ}/originales/clima/clima_07_pasterze_glacier.jpg")[1]
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.6))
    for a, g, dom, tit in ((ax[0], t, "tiroides", "Tiroides (US), panel (b)"), (ax[1], c, "clima", "Glaciar Pasterze")):
        s = gauss(g, SIG[dom]); u = unsharp(s, SIG_U, ALFA)
        for im, lab, col in ((g, "original", "0.25"), (s, "suavizada", "tab:blue"), (u, "suavizada + unsharp", "tab:red")):
            h = np.bincount(im.ravel(), minlength=256) / im.size
            a.plot(h, lw=0.8, color=col, label=lab)
        a.set_title(tit, fontsize=6.5); a.set_xlabel("Nivel de gris"); a.set_xlim(0, 255)
    ax[0].set_ylabel("Frecuencia relativa"); ax[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout(pad=0.4); fig.savefig(f"{FIG}/f2_histogramas.png", dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

def f3_morfologia():
    t = panel_708(); m = mascara(t, "tiroides")
    g = cargar(f"{RAIZ}/originales/clima/clima_07_pasterze_glacier.jpg")[1][190:400, 220:470]
    fila1 = [m, erosion(m, 3), dilatacion(m, 3), apertura(m, R_AP), cierre(m, R_CI), gradiente(cierre(apertura(m, R_AP), R_CI), 1)]
    fila2 = [g, erosion(g, 3), dilatacion(g, 3), tophat(g, 6), blackhat(g, 6), gradiente(g, 1)]
    fila2[3] = cv2.normalize(fila2[3], None, 0, 255, cv2.NORM_MINMAX); fila2[4] = cv2.normalize(fila2[4], None, 0, 255, cv2.NORM_MINMAX)
    fila2[5] = cv2.normalize(fila2[5], None, 0, 255, cv2.NORM_MINMAX)
    rejilla(2, 6, 2.0, "f3_morfologia", ["Original / máscara", "Erosión ($r$=3)", "Dilatación ($r$=3)", "Apertura/Top-hat", "Cierre/Black-hat", "Gradiente"],
            [fila1, fila2], ["US (binaria)", "Glaciar"])

def f4_fantomas():
    filas = []
    for caso in "AB":
        img, gt = CASOS[caso](np.random.default_rng(3))
        m, _ = otsu(gauss(img, 1.0), None, True)
        filas.append([img, gt, m, secuencia(m, "apertura", 5, 5), secuencia(m, "ci>ap", 5, 5)])
    img, gt = CASOS["C"](np.random.default_rng(3)); g = gauss(img, 1.0)
    def umbral(r): return ((r > np.percentile(r, 97)) * 255).astype(np.uint8)
    filas.append([img, gt, umbral(sobel(g)), umbral(gradiente(g, 1)), umbral(blackhat(g, 6))])
    rejilla(3, 5, 3.4, "f4_fantomas", ["Sintética con ruido", "Verdad de terreno", "Sin morfología / Sobel", "Apertura / Grad. morf.", "Cierre>apertura / Black-hat"], filas,
            ["A: separar", "B: reconectar", "C: línea fina"])

def f5_casos():
    t = panel_708(); m7 = cargar(f"{RAIZ}/originales/tiroides/tiroides_707.png")[1]
    c = cargar(f"{RAIZ}/originales/clima/clima_07_pasterze_glacier.jpg")[1]
    filas = []
    for g, dom in ((t, "tiroides"), (m7, "tiroides"), (c, "clima")):
        m = mascara(g, dom); f = secuencia(m, "ap>ci", R_AP, R_CI)
        extra = sobel(unsharp(gauss(g, SIG[dom]), SIG_U, ALFA)) if dom == "tiroides" else cv2.normalize(blackhat(gauss(g, SIG[dom]), 6), None, 0, 255, cv2.NORM_MINMAX)
        filas.append([g, m, f, contorno(g, f), extra])
    rejilla(3, 5, 3.3, "f5_casos", ["Original", "Máscara (Otsu)", "Apertura>cierre ($r$=5)", "Contorno superpuesto", "Sobel / Black-hat"], filas,
            ["Ecografía 708(b)", "Resonancia 707", "Glaciar 07"])

def f6_chad():
    from experimentos import agua_cromatica
    bgr, _ = cargar(f"{RAIZ}/originales/clima/clima_06_lake_chad_1973_2007.jpg"); mid = bgr.shape[1] // 2
    P = {"1973": bgr[:, :mid], "2007": bgr[:, mid:]}; filas = []
    for k, p in P.items():
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        m1 = cierre(apertura(otsu(gauss(g, 1.1), None, True)[0], 1), 3)
        m2 = agua_cromatica(p)
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        def sup(m):
            o = rgb.copy(); o[gradiente(m, 1) > 0] = (255, 255, 0); return o
        filas.append([rgb, sup(m1), sup(m2)])
    rejilla(2, 3, 2.05, "f6_chad", ["Original", "Otsu en grises (por panel)", "Regla cromática (B, G > R)"], filas, ["1973", "2007"], ancho=4.7)

if __name__ == "__main__":
    for f in (f1_espaciales, f2_histogramas, f3_morfologia, f4_fantomas, f5_casos, f6_chad):
        f(); print("ok", f.__name__)
