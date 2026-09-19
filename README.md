# Filtros espaciales y morfológicos en imágenes reales

Código y resultados de la Actividad 2 (grupal) de **Visión Artificial**, Máster Universitario en Inteligencia Artificial, UNIR.

**Autores:** Gabriel Eduardo Morejón López · Wendy Viviana Obregón Martínez

Comparamos filtros espaciales (suavizado y realce de bordes) y operaciones morfológicas (erosión, dilatación, apertura, cierre, gradiente, top-hat y black-hat) sobre veinte imágenes reales: diez de cáncer de tiroides (ecografía, resonancia, tomografía y gammagrafía) y diez de evidencia visual del cambio climático. Como las imágenes reales no tienen segmentación de referencia, añadimos tres fantomas sintéticos cuya solución correcta se conoce, y así podemos medir aciertos y no solo describirlos.

El razonamiento completo, las ecuaciones y la discusión están en el informe (PDF/Word) de la actividad. Este repositorio contiene lo necesario para reproducir todas sus tablas y figuras.

## Qué hay aquí

```
.
├── codigo/
│   ├── filtros.py        # filtros espaciales, morfología, segmentación y métricas
│   ├── fantomas.py       # imágenes sintéticas con verdad de terreno (casos A, B y C)
│   ├── experimentos.py   # ejecuta los experimentos y escribe los CSV
│   └── figuras.py        # genera las seis figuras del informe
├── resultados/
│   ├── csv/              # tablas numéricas ya generadas (e1 … e8)
│   └── figuras/          # figuras ya generadas (f1 … f6)
├── originales/           # aquí van las imágenes (ver «Datos»); vacío en el repositorio
├── requirements.txt
└── README.md
```

## Lo que hace el código

`filtros.py` reúne las operaciones del informe: suavizado por media, mediana y gaussiano; realce por máscara de desenfoque; gradiente de Sobel y detector de Canny; erosión, dilatación, apertura, cierre, gradiente morfológico, top-hat y black-hat con elemento estructural en forma de disco; umbral de Otsu restringido a la zona válida de la imagen; y las métricas (ruido relativo de Immerkær, índice de conservación de bordes, SSIM, contraste local, fuerza de bordes, componentes conexos, Dice, precisión, exhaustividad y F1).

`fantomas.py` genera tres casos con ruido speckle: **A**, dos discos unidos por un puente fino (hay que separarlos); **B**, un disco fisurado (hay que reconectarlo); **C**, una línea oscura fina sobre fondo claro.

`experimentos.py` ejecuta ocho bloques y guarda un CSV por bloque:

| Bloque | Qué mide | Salida |
|---|---|---|
| E1 | Barrido de σ del suavizado gaussiano | `e1_suavizado.csv` |
| E3 | Barrido del realce (σᵤ, α) y amplificación de ruido | `e3_unsharp.csv` |
| E4 | Métricas por etapa de la cadena | `e4_etapas.csv` |
| E5 | Fantomas A/B y C, 30 realizaciones | `e5_fantomas_AB.csv`, `e5_fantomas_C.csv` |
| E6 | Cinco secuencias morfológicas sobre las imágenes reales | `e6_real.csv` |
| E6b | Sensibilidad al radio del elemento estructural | `e6b_radios.csv` |
| E7 | Retracción del lago Chad, Otsu frente a regla cromática | `e7_chad.csv` |
| E8 | Otsu de dos y tres clases | `e8_multiotsu.csv` |

Los parámetros elegidos (σ = 0,8 en tiroides y 0,6 en clima, radio 5 para apertura y cierre, δ = 10 en la regla cromática) están al inicio de `experimentos.py` y se justifican en la sección 2.9 del informe.

## Datos

Las imágenes **no se distribuyen** porque sus derechos pertenecen a terceros. Para reproducir los resultados hay que descargarlas y colocarlas así:

```
originales/
├── tiroides/   tiroides_703.png … tiroides_719.png   (10 imágenes)
└── clima/      clima_01_….jpg … clima_10_….jpg       (10 imágenes)
```

- **Tiroides:** [Open-i](https://openi.nlm.nih.gov/) (Biblioteca Nacional de Medicina de EE. UU.), figuras del artículo de King (2008), *Imaging for staging and management of thyroid cancer*, Cancer Imaging 8(1), 57–69, PMC2324369.
- **Cambio climático:** [Science Source](https://www.sciencesource.com/), diez fotografías de glaciares, lago Chad, huracán Michael, escombros marinos y otros.

Los nombres de archivo deben coincidir con el patrón de arriba: el código toma todos los `*.png` de `tiroides/` y todos los `*.jpg` de `clima/`. La tabla B.1 del informe lista cada imagen con su identificador.

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd codigo
python experimentos.py             # unos 15 s; escribe resultados/csv/
python figuras.py                  # escribe resultados/figuras/
```

`figuras.py` necesita que `experimentos.py` se haya ejecutado antes. Las semillas de los fantomas están fijadas, así que los CSV salen idénticos entre ejecuciones. Los resultados incluidos se generaron con Python 3.11, OpenCV 4.13, scikit-image 0.26, NumPy 2.4, pandas 3.0 y matplotlib 3.10.

## Resultados principales

- El suavizado gaussiano conserva más bordes que la media y la mediana con el mismo ruido residual.
- El realce por máscara de desenfoque sube el contraste local un 12 % en tiroides y un 17 % en clima, pero multiplica el ruido por más de dos. Si se aplica antes de binarizar, duplica los componentes conexos.
- En los fantomas no existe un orden morfológico universal: la apertura separa objetos unidos por puentes finos, el cierre reconecta objetos fisurados, y solo las secuencias que combinan ambas resuelven los dos casos.
- El black-hat de radio 6 recupera líneas oscuras finas con F1 = 0,97; Sobel llega a 0,69.
- En las imágenes reales, apertura seguida de cierre reduce los componentes conexos un 84 % en tiroides y un 97 % en clima. Eso mide fragmentación, no calidad.
- El umbral global de Otsu no delimita las lesiones tiroideas y subestima la retracción del lago Chad (−24 % en grises frente a −91 % con la regla cromática).
- El radio del elemento estructural cambia el resultado más que el orden de las operaciones.

## Limitaciones

Son diez imágenes por dominio y sin segmentación manual, así que las conclusiones sobre imágenes reales son descriptivas; solo los fantomas permiten medir aciertos. Los umbrales de Canny (50/150) y de la regla cromática son fijos y no se adaptan a cada imagen. Las imágenes médicas se usan con fines académicos de procesamiento de señal y no tienen valor diagnóstico.

## Uso de inteligencia artificial

Usamos Claude (Anthropic) como apoyo para auditar cifras, diseñar experimentos y redactar. Ejecutamos el código, revisamos los resultados y asumimos la responsabilidad del contenido. El detalle está en la sección 5 del informe.

## Referencias del método

Canny (1986), Otsu (1979), Immerkær (1996), Wang et al. (2004), Haralick et al. (1987), Serra (1982), Soille (2003), Gonzalez y Woods (2018), McFeeters (1996) y Xu (2006). La lista completa con DOI está en el informe.

## Licencia

Código: se puede elegir la licencia al publicar el repositorio (por ejemplo MIT). Las imágenes y el texto del informe no forman parte de la licencia del código.
