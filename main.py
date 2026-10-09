import cv2
import json
import os

from src.procesamiento import (
    cargar_imagen,
    crear_mascara_papel,
    encontrar_regiones_papel,
    dibujar_candidatos,
    detectar_pliegue,
    recortar_pagina_derecha,
    crear_mascara_pagina,
    encontrar_contorno_pagina,
    analizar_mascara_por_filas,
    corregir_perspectiva,
    detectar_borde_inferior,
    detectar_borde_superior,
    obtener_esquinas_pagina,
    preparar_imagen_ocr,
    eliminar_lineas_horizontales,
    detectar_lineas_horizontales,
    dibujar_lineas_horizontales,
)

from src.ocr import(
    realizar_ocr,
    realizar_ocr_con_coordenadas,
    agrupar_palabras_por_linea,
)


# --------------------------------------------------
# 1. Cargar imagen
# --------------------------------------------------

# --------------------------------------------------
# Configuración del formato
# --------------------------------------------------

ruta_imagen = "fotos/Formato2.jpg"

nombre_formato = os.path.splitext(
    os.path.basename(ruta_imagen)
)[0]

carpeta_resultados = f"resultados/{nombre_formato}"
carpeta_procesamiento = f"{carpeta_resultados}/procesamiento"
carpeta_ocr = f"{carpeta_resultados}/ocr"

os.makedirs(carpeta_procesamiento, exist_ok=True)
os.makedirs(carpeta_ocr, exist_ok=True)

imagen = cargar_imagen(ruta_imagen)

alto, ancho = imagen.shape[:2]

print(f"Ancho de la imagen: {ancho} píxeles")
print(f"Alto de la imagen: {alto} píxeles")
# --------------------------------------------------
# 2. Detectar regiones de papel
# --------------------------------------------------

mascara_papel = crear_mascara_papel(imagen)

cv2.imwrite(
    f"{carpeta_procesamiento}/mascara_papel.jpg",
    mascara_papel
)

candidatos = encontrar_regiones_papel(
    mascara_papel,
    ancho,
    alto
)

print(f"Regiones candidatas: {len(candidatos)}")

for i, candidato in enumerate(candidatos, start=1):

    print(f"\nCandidato {i}")
    print(f"  Área: {candidato['area']}")
    print(
        f"  Posición: "
        f"({candidato['x']}, {candidato['y']})"
    )
    print(
        f"  Tamaño: "
        f"{candidato['ancho']} x {candidato['alto']}"
    )
    print(
        f"  Proporción: "
        f"{candidato['proporcion']:.2f}"
    )
    print(
        f"  Vértices aproximados: "
        f"{candidato['vertices']}"
    )


# --------------------------------------------------
# 3. Dibujar candidatos
# --------------------------------------------------

imagen_candidatos = dibujar_candidatos(
    imagen.copy(),
    candidatos
)

cv2.imwrite(
    "f{carpeta_procesamiento}/candidatos_papel.jpg",
    imagen_candidatos
)


# --------------------------------------------------
# 4. Detectar pliegue
# --------------------------------------------------

if not candidatos:
    raise RuntimeError(
        "No se encontró ninguna región candidata."
    )

candidato = candidatos[0]

x_pliegue, perfil_pliegue = detectar_pliegue(
    imagen,
    candidato
)

print(
    f"\nPliegue detectado aproximadamente en:"
    f"\nx = {x_pliegue}"
)


# --------------------------------------------------
# 5. Dibujar pliegue
# --------------------------------------------------

imagen_pliegue = imagen.copy()

cv2.line(
    imagen_pliegue,
    (x_pliegue, 0),
    (x_pliegue, alto - 1),
    (0, 0, 255),
    2
)

cv2.imwrite(
    f"{carpeta_procesamiento}/pliegue_detectado.jpg",
    imagen_pliegue
)


# --------------------------------------------------
# 6. Recortar página derecha
# --------------------------------------------------

pagina_derecha = recortar_pagina_derecha(
    imagen,
    candidato,
    x_pliegue
)

cv2.imwrite(
    f"{carpeta_procesamiento}/pagina_derecha.jpg",
    pagina_derecha
)

alto_pagina, ancho_pagina = pagina_derecha.shape[:2]

print("\nPágina derecha:")
print(f"  Ancho: {ancho_pagina} píxeles")
print(f"  Alto: {alto_pagina} píxeles")


# --------------------------------------------------
# 7. Crear máscara local de la página
# --------------------------------------------------

mascara_pagina = crear_mascara_pagina(pagina_derecha)

cv2.imwrite(
    f"{carpeta_procesamiento}/mascara_pagina.jpg",
    mascara_pagina
)

y_borde_inferior, perfil_filas = detectar_borde_inferior(
    mascara_pagina
)

print("\nBorde inferior detectado:")
print(f"  y = {y_borde_inferior}")

y_borde_superior, perfil_superior = detectar_borde_superior(
    mascara_pagina
)

print("\nBorde superior detectado:")
print(f"  y = {y_borde_superior}")
# --------------------------------------------------
# 8. Encontrar contorno principal
# --------------------------------------------------

contorno_pagina = encontrar_contorno_pagina(
    mascara_pagina
)

if contorno_pagina is None:
    raise RuntimeError(
        "No se encontró el contorno de la página."
    )


# --------------------------------------------------
# 9. Analizar geometría básica
# --------------------------------------------------

area = cv2.contourArea(
    contorno_pagina
)

esquinas, metodo = obtener_esquinas_pagina(
    contorno_pagina
)

print("\nEsquinas utilizadas:")
print(f"  Método: {metodo}")

for punto in esquinas:

    x, y = punto

    print(
        f"  ({x}, {y})"
    )


# --------------------------------------------------
# 10. Corregir perspectiva
# --------------------------------------------------

esquinas_cv = esquinas.reshape(4, 1, 2)

pagina_rectificada = corregir_perspectiva(
    pagina_derecha,
    esquinas_cv
)

cv2.imwrite(
    f"{carpeta_procesamiento}/pagina_rectificada.jpg",
    pagina_rectificada
)

# --------------------------------------------------
# 11. Preparar imagen para OCR
# --------------------------------------------------

gris, normalizada, contraste = preparar_imagen_ocr(
    pagina_rectificada
)

# ============================================================
# DETECCIÓN DE LÍNEAS HORIZONTALES
# ============================================================

lineas = detectar_lineas_horizontales(
    contraste
)

imagen_lineas = dibujar_lineas_horizontales(
    contraste,
    lineas
)

ruta_lineas = f"{carpeta_ocr}/lineas_horizontales.jpg"

cv2.imwrite(
    ruta_lineas,
    imagen_lineas
)

cantidad_lineas = 0 if lineas is None else len(lineas)

print(
    f"\nSegmentos horizontales detectados: "
    f"{cantidad_lineas}"
)

print(
    f"Imagen de líneas:"
    f"\n  {ruta_lineas}"
)

# ============================================================
# ELIMINAR LINEAS HORIZONTALES
# ============================================================

sin_lineas = eliminar_lineas_horizontales(
    contraste
)

ruta_sin_lineas = f"{carpeta_ocr}/sin_lineas.jpg"

print("Tipo de sin_lineas:", type(sin_lineas))

if sin_lineas is not None:
    print("Forma de sin_lineas:", getattr(sin_lineas, "shape", None))

cv2.imwrite(
    ruta_sin_lineas,
    sin_lineas
)

print(
    f"\nImagen OCR sin líneas creada:"
    f"\n  {ruta_sin_lineas}"
)

# ============================================================
# PRUEBA OCR POR REGIONES Y PSM
# ============================================================

alto_ocr, ancho_ocr = contraste.shape[:2]

print("\n================ PRUEBA OCR POR REGIONES ================\n")

print(
    f"Imagen OCR: {ancho_ocr} x {alto_ocr}"
)

# ------------------------------------------------------------
# CABECERA
# ------------------------------------------------------------

cabecera = contraste[0:800, 0:ancho_ocr]

ruta_cabecera = f"{carpeta_ocr}/prueba_cabecera.jpg"

cv2.imwrite(
    ruta_cabecera,
    cabecera
)

print("\n================ CABECERA ================\n")

for psm in [6, 11, 12]:

    print(f"\n--- PSM {psm} ---")

    texto = realizar_ocr(
        cabecera,
        psm=psm
    )

    print(texto)


# ------------------------------------------------------------
# PRIMERA FILA
# ------------------------------------------------------------

fila1 = contraste[850:1150, 0:ancho_ocr]

ruta_fila1 = f"{carpeta_ocr}/prueba_fila1.jpg"

cv2.imwrite(
    ruta_fila1,
    fila1
)

print("\n================ PRIMERA FILA ================\n")

for psm in [6, 11, 12]:

    print(f"\n--- PSM {psm} ---")

    texto = realizar_ocr(
        fila1,
        psm=psm
    )

    print(texto)

print("\n==========================================================\n")
print("\nTamaños de imágenes OCR:")

print(
    f"  Página rectificada: "
    f"{pagina_rectificada.shape[1]} x "
    f"{pagina_rectificada.shape[0]}"
)

print(
    f"  Gris: "
    f"{gris.shape[1]} x "
    f"{gris.shape[0]}"
)

print(
    f"  Normalizada: "
    f"{normalizada.shape[1]} x "
    f"{normalizada.shape[0]}"
)

print(
    f"  Contraste: "
    f"{contraste.shape[1]} x "
    f"{contraste.shape[0]}"
)

cv2.imwrite(
    f"{carpeta_ocr}/gris.jpg",
    gris
)

cv2.imwrite(
    f"{carpeta_ocr}/iluminacion_normalizada.jpg",
    normalizada
)

cv2.imwrite(
    f"{carpeta_ocr}/contraste_ocr.jpg",
    contraste
)

print("\nImágenes para OCR creadas:")
print(f"  {carpeta_ocr}/gris.jpg")
print(f"  {carpeta_ocr}/iluminacion_normalizada.jpg")
print(f"  {carpeta_ocr}/contraste_ocr.jpg")

# --------------------------------------------------
# 12. Realizar OCR
# --------------------------------------------------

texto = realizar_ocr(contraste)

print("\n================ OCR ================\n")
print(texto)
print("\n======================================")

palabras = realizar_ocr_con_coordenadas(contraste)

lineas = agrupar_palabras_por_linea(palabras)

print("\n========== LINEAS RECONSTRUIDAS ==========\n")

for numero, linea in enumerate(lineas, start=1):

    texto_linea = " ".join(
        palabra["texto"]
        for palabra in linea["palabras"]
    )

    print(
        f"{numero:03} "
        f"y={linea['centro_y']:7.1f} "
        f"{texto_linea}"
    )

print("\n===========================================")

with open(
    f"{carpeta_ocr}/ocr_coordenadas.json",
    "w",
    encoding="utf-8"
) as archivo:

    json.dump(
        palabras,
        archivo,
        ensure_ascii=False,
        indent=2
    )

print("\n========== OCR CON COORDENADAS ==========\n")

print(f"Palabras detectadas: {len(palabras)}")

for palabra in palabras[:30]:

    print(
        f"x={palabra['x']:4} "
        f"y={palabra['y']:4} "
        f"conf={palabra['confianza']:5.1f} "
        f"{palabra['texto']}"
    )

print("\n==========================================")

with open(
    f"{carpeta_ocr}/texto_ocr.txt",
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write(texto)

alto_rectificada, ancho_rectificada = (
    pagina_rectificada.shape[:2]
)

print("\nPerspectiva corregida:")
print(
    f"  Ancho: {ancho_rectificada} píxeles"
)

print(
    f"  Alto: {alto_rectificada} píxeles"
)

print(
    "  Archivo creado:"
    f" {carpeta_procesamiento}/pagina_rectificada.jpg"
)


# --------------------------------------------------
# 11. Mostrar geometría encontrada
# --------------------------------------------------

print("\nContorno de la página derecha:")
print(f"  Área: {area:.0f}")

print(
    f"  Esquinas utilizadas: "
    f"{len(esquinas)}"
)

for punto in esquinas:

    x, y = punto

    print(
        f"  ({x}, {y})"
    )


# --------------------------------------------------
# 12. Dibujar contorno y esquinas
# --------------------------------------------------

imagen_contorno = pagina_derecha.copy()

cv2.drawContours(
    imagen_contorno,
    [contorno_pagina],
    -1,
    (0, 0, 255),
    2
)

for punto in esquinas:

    x, y = punto

    cv2.circle(
        imagen_contorno,
        (x, y),
        6,
        (0, 255, 0),
        -1
    )

cv2.imwrite(
    f"{carpeta_procesamiento}/contorno_pagina.jpg",
    imagen_contorno
)