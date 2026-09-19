"""experimentos.py - Genera todas las tablas (CSV) del informe.
Uso: python experimentos.py   (imagenes en ../originales/{tiroides,clima})"""
import glob, os
import numpy as np, pandas as pd, cv2
from filtros import *
from fantomas import CASOS
from skimage.filters import threshold_otsu

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "resultados", "csv"); os.makedirs(CSV, exist_ok=True)
DOM = {"tiroides": ("png", True), "clima": ("jpg", False)}   # (extension, estructura oscura)
SIG = {"tiroides": 0.8, "clima": 0.6}                         # sigma elegida (E2)
SIG_U, ALFA = 2.0, 1.0                                        # unsharp elegido (E3)
R_AP = R_CI = 5                                               # radios elegidos (E5)

def imagenes():
    for dom, (ext, _) in DOM.items():
        for f in sorted(glob.glob(f"{RAIZ}/originales/{dom}/*.{ext}")):
            yield dom, os.path.basename(f), cargar(f)[1]

def e1_suavizado():
    """Comparacion de filtros de suavizado (ruido relativo, EPI, SSIM)."""
    filas = []
    for dom, n, g in imagenes():
        n0 = ruido(g)
        v = {f"media k={k}": media(g, k) for k in (3, 5)}
        v.update({f"mediana k={k}": mediana(g, k) for k in (3, 5)})
        v.update({f"gauss s={s}": gauss(g, s) for s in (0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0)})
        for nombre, f in v.items():
            r = ruido(f) / n0; e = epi(g, f)
            filas.append(dict(dominio=dom, imagen=n, filtro=nombre, ruido_rel=r,
                              epi=e, ssim=ssim(g, f), fom=(1 - r) * e))
    pd.DataFrame(filas).to_csv(f"{CSV}/e1_suavizado.csv", index=False)

def e3_unsharp():
    """Barrido (sigma_u, alfa): ganancia de bordes, amplificacion de ruido, contraste."""
    filas = []
    for dom, n, g in imagenes():
        gs = gauss(g, SIG[dom]); n_s, p_s = ruido(gs), borde_p95(gs)
        c_s, h_s = contraste_rms(gs), entropia(gs)
        for su in (1.0, 1.5, 2.0, 3.0):
            for a in (0.5, 1.0, 1.5, 2.0):
                u = unsharp(gs, su, a)
                filas.append(dict(dominio=dom, imagen=n, sigma_u=su, alfa=a,
                                  ganancia_borde=borde_p95(u) / p_s,
                                  amp_ruido=ruido(u) / max(n_s, 1e-6),
                                  rms_rel=contraste_rms(u) / c_s, d_entropia=entropia(u) - h_s))
    pd.DataFrame(filas).to_csv(f"{CSV}/e3_unsharp.csv", index=False)

def e4_etapas():
    """Metricas de contraste, nitidez y bordes por etapa del pipeline espacial."""
    filas = []
    for dom, n, g in imagenes():
        gs = gauss(g, SIG[dom]); gu = unsharp(gs, SIG_U, ALFA)
        for etapa, im in (("original", g), ("suavizada", gs), ("suavizada+unsharp", gu)):
            filas.append(dict(dominio=dom, imagen=n, etapa=etapa, rms=contraste_rms(im),
                              c_local=contraste_local(im),
                              entropia=entropia(im), nitidez=nitidez(im), ruido=ruido(im),
                              borde_p95=borde_p95(im), dens_canny=dens_canny(im)))
    pd.DataFrame(filas).to_csv(f"{CSV}/e4_etapas.csv", index=False)

def segmenta(g, dom, unsh, seq, r_ap=R_AP, r_ci=R_CI):
    s = gauss(g, SIG[dom])
    if unsh: s = unsharp(s, SIG_U, ALFA)
    m, _ = otsu(s, zona_valida(g), DOM[dom][1])
    return secuencia(m, seq, r_ap, r_ci)

def e6_real():
    """Ablacion de secuencias sobre las 20 imagenes reales; estabilidad ante ruido."""
    filas = []
    for dom, n, g in imagenes():
        ruidosas = [np.clip(g + np.random.default_rng(s).normal(0, 8, g.shape), 0, 255).astype(np.uint8)
                    for s in range(3)]
        base = segmenta(g, dom, False, "ninguna")
        for unsh in (False, True):
            for seq in ("ninguna", "apertura", "cierre", "ap>ci", "ci>ap"):
                m = segmenta(g, dom, unsh, seq)
                est = np.mean([dice(m, segmenta(gr, dom, unsh, seq)) for gr in ruidosas])
                filas.append(dict(dominio=dom, imagen=n, unsharp=unsh, secuencia=seq,
                                  componentes=n_comp(m), pequenas=frac_pequenas(m),
                                  area=100 * (m > 0).mean(), estabilidad=est,
                                  comp_base=n_comp(base)))
    pd.DataFrame(filas).to_csv(f"{CSV}/e6_real.csv", index=False)

def e6b_radios():
    """Sensibilidad al radio (secuencia ap>ci, gauss sin unsharp)."""
    filas = []
    for dom, n, g in imagenes():
        for r in (2, 3, 5, 7, 9):
            m = segmenta(g, dom, False, "ap>ci", r, r)
            filas.append(dict(dominio=dom, imagen=n, radio=r, componentes=n_comp(m), area=100 * (m > 0).mean()))
    pd.DataFrame(filas).to_csv(f"{CSV}/e6b_radios.csv", index=False)

def e5_fantomas(semillas=30):
    """Fantomas con verdad de terreno: A (separar), B (reconectar), C (lineas)."""
    filas = []
    for caso, objetivo in (("A", 2), ("B", 1)):
        for s in range(semillas):
            img, gt = CASOS[caso](np.random.default_rng(s))
            for sig in (1.0, 2.0):
                m, _ = otsu(gauss(img, sig), None, True)
                for seq in ("ninguna", "apertura", "cierre", "ap>ci", "ci>ap"):
                    for r in ((5,) if seq == "ninguna" else (3, 5)):
                        rs = secuencia(m, seq, r, r)
                        filas.append(dict(caso=caso, sigma=sig, secuencia=seq, radio=r, semilla=s,
                                          dice=dice(rs, gt), objetos_ok=int(n_comp(rs) == objetivo)))
    pd.DataFrame(filas).to_csv(f"{CSV}/e5_fantomas_AB.csv", index=False)
    filas = []
    for s in range(semillas):
        img, gt = CASOS["C"](np.random.default_rng(s))
        g = gauss(img, 1.0)
        resp = {"Sobel": sobel(g), "Gradiente morfologico (r=1)": gradiente(g, 1),
                "Black-hat r=4": blackhat(g, 4), "Black-hat r=6": blackhat(g, 6)}
        for nombre, rp in resp.items():          # mejor umbral de percentil, igual para todos
            mejor = max((prf((rp > np.percentile(rp, q)).astype(np.uint8) * 255, gt) for q in np.arange(85, 99.9, 0.5)),
                        key=lambda t: t[2])
            filas.append(dict(metodo=nombre, semilla=s, precision=mejor[0], exhaustividad=mejor[1], f1=mejor[2]))
        mejor = max((prf(canny(g, t // 3, t), gt) for t in range(30, 240, 15)), key=lambda t: t[2])
        filas.append(dict(metodo="Canny (mejor umbral)", semilla=s, precision=mejor[0],
                          exhaustividad=mejor[1], f1=mejor[2]))
    pd.DataFrame(filas).to_csv(f"{CSV}/e5_fantomas_C.csv", index=False)

def agua_cromatica(p, delta=10):
    """Agua en composicion en falso color: azul y verde superan al rojo en delta niveles."""
    f = cv2.GaussianBlur(p, (5, 5), 0).astype(int)
    m = ((f[..., 0] - f[..., 2] > delta) & (f[..., 1] - f[..., 2] > delta)).astype(np.uint8) * 255
    return cierre(apertura(m, 1), 3)

def e7_chad():
    """Lago Chad: Otsu en grises por panel frente a regla cromatica (barrido de delta)."""
    bgr, _ = cargar(f"{RAIZ}/originales/clima/clima_06_lake_chad_1973_2007.jpg")
    mid = bgr.shape[1] // 2
    P = {"1973": bgr[:, :mid], "2007": bgr[:, mid:]}
    filas = []
    for k, p in P.items():
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        m1, t1 = otsu(gauss(g, 1.1), None, True)
        m1 = cierre(apertura(m1, 1), 3)
        B, G, R = [p[..., i].astype(float) for i in (0, 1, 2)]
        rojiza = R > 1.3 * np.maximum(B, G)
        fila = dict(panel=k, umbral_gris=t1, area_gris=100 * (m1 > 0).mean(),
                    rojizo_en_mascara=100 * (rojiza & (m1 > 0)).sum() / (m1 > 0).sum())
        for d in (0, 5, 10, 15, 20):
            fila[f"area_cromatica_d{d}"] = 100 * (agua_cromatica(p, d) > 0).mean()
        filas.append(fila)
    pd.DataFrame(filas).to_csv(f"{CSV}/e7_chad.csv", index=False)

def e8_multiotsu():
    """Ecografia 708(b): Otsu de dos clases frente a tres clases (clase mas oscura)."""
    from skimage.filters import threshold_multiotsu
    g = cargar(f"{RAIZ}/originales/tiroides/tiroides_708.png")[1][:, 279:512]
    s = gauss(g, SIG["tiroides"]); v = zona_valida(g)
    m2, _ = otsu(s, v, True)
    t3 = threshold_multiotsu(s[v], classes=3)
    m3 = ((s <= t3[0]) & v).astype(np.uint8) * 255
    filas = []
    for nombre, m in (("2 clases", m2), ("3 clases (mas oscura)", m3)):
        f = secuencia(m, "ap>ci", R_AP, R_CI)
        filas.append(dict(umbral=nombre, area=100 * (m > 0).mean(), comp_brutas=n_comp(m),
                          comp_ap_ci=n_comp(f), area_ap_ci=100 * (f > 0).mean()))
    pd.DataFrame(filas).to_csv(f"{CSV}/e8_multiotsu.csv", index=False)

if __name__ == "__main__":
    for f in (e1_suavizado, e3_unsharp, e4_etapas, e6_real, e6b_radios, e5_fantomas, e7_chad, e8_multiotsu):
        f(); print("ok", f.__name__)
