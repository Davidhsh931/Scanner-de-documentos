import cv2
import math
import numpy as np


def cargar_imagen(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo abrir la imagen: {ruta}"
        )

    return imagen


def crear_mascara_papel(imagen):

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    # Suavizamos para reducir el efecto
    # de letras y pequeños detalles.
    gris = cv2.GaussianBlur(
        gris,
        (15, 15),
        0
    )

    # En lugar de utilizar un valor fijo,
    # calculamos automáticamente un umbral
    # a partir de la imagen.
    _, mascara = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Unimos pequeñas interrupciones dentro
    # de las regiones claras.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (31, 31)
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mascara


def encontrar_regiones_papel(
    mascara,
    ancho_imagen,
    alto_imagen
):

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidatos = []

    area_imagen = (
        ancho_imagen * alto_imagen
    )

    for contorno in contornos:

        area = cv2.contourArea(
            contorno
        )

        # Ignoramos regiones diminutas.
        if area < area_imagen * 0.01:
            continue

        x, y, ancho, alto = cv2.boundingRect(
            contorno
        )

        if alto == 0:
            continue

        proporcion = ancho / alto

        perimetro = cv2.arcLength(
            contorno,
            True
        )

        aproximacion = cv2.approxPolyDP(
            contorno,
            0.02 * perimetro,
            True
        )

        candidatos.append({
            "contorno": contorno,
            "area": area,
            "x": x,
            "y": y,
            "ancho": ancho,
            "alto": alto,
            "proporcion": proporcion,
            "vertices": len(aproximacion),
            "aproximacion": aproximacion
        })

    candidatos.sort(
        key=lambda candidato: candidato["area"],
        reverse=True
    )

    return candidatos


def dibujar_candidatos(
    imagen,
    candidatos,
    cantidad=10
):

    resultado = imagen.copy()

    for indice, candidato in enumerate(
        candidatos[:cantidad],
        start=1
    ):

        contorno = candidato["contorno"]

        cv2.drawContours(
            resultado,
            [contorno],
            -1,
            (0, 0, 255),
            3
        )

        x = candidato["x"]
        y = candidato["y"]

        cv2.putText(
            resultado,
            str(indice),
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    return resultado

def detectar_pliegue(imagen, candidato):

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # 1. Suavizar
    # --------------------------------------------------

    gris = cv2.GaussianBlur(
        gris,
        (31, 31),
        0
    )

    alto_imagen, ancho_imagen = gris.shape

    x = candidato["x"]
    y = candidato["y"]
    ancho = candidato["ancho"]
    alto = candidato["alto"]

    # --------------------------------------------------
    # 2. Zona donde podría estar el pliegue
    # --------------------------------------------------

    # No buscamos en toda la fotografía.
    #
    # El pliegue debe estar relativamente cerca
    # del lado izquierdo de la página derecha.

    x_inicio = x + int(ancho * 0.05)
    x_fin = x + int(ancho * 0.30)

    x_inicio = max(0, x_inicio)
    x_fin = min(ancho_imagen, x_fin)

    y_inicio = y + int(alto * 0.05)
    y_fin = y + int(alto * 0.95)

    # --------------------------------------------------
    # 3. Gradiente horizontal
    # --------------------------------------------------

    gradiente = cv2.Sobel(
        gris,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gradiente = np.abs(
        gradiente
    )

    # --------------------------------------------------
    # 4. Dividir la altura en franjas
    # --------------------------------------------------

    numero_franjas = 12

    resultados = []

    altura_franja = (
        y_fin - y_inicio
    ) // numero_franjas

    for i in range(numero_franjas):

        inicio_y = (
            y_inicio +
            i * altura_franja
        )

        if i == numero_franjas - 1:

            fin_y = y_fin

        else:

            fin_y = (
                y_inicio +
                (i + 1) * altura_franja
            )

        franja = gradiente[
            inicio_y:fin_y,
            x_inicio:x_fin
        ]

        perfil = franja.mean(
            axis=0
        )

        # Suavizamos el perfil.
        perfil = cv2.GaussianBlur(
            perfil.astype(np.float32),
            (31, 1),
            0
        ).reshape(-1)

        indice = int(
            np.argmax(perfil)
        )

        x_detectado = (
            x_inicio +
            indice
        )

        fuerza = float(
            perfil[indice]
        )

        resultados.append({
            "franja": i + 1,
            "x": x_detectado,
            "fuerza": fuerza
        })

    # --------------------------------------------------
    # 5. Buscar una posición consistente
    # --------------------------------------------------

    posiciones = np.array([
        resultado["x"]
        for resultado in resultados
    ])

    # La mediana evita que unas pocas franjas
    # contaminadas por texto dominen el resultado.

    x_pliegue = int(
        np.median(posiciones)
    )

    # --------------------------------------------------
    # 6. Perfil global para diagnóstico
    # --------------------------------------------------

    zona_global = gradiente[
        y_inicio:y_fin,
        x_inicio:x_fin
    ]

    perfil_global = zona_global.mean(
        axis=0
    )

    perfil_global = cv2.GaussianBlur(
        perfil_global.astype(np.float32),
        (31, 1),
        0
    ).reshape(-1)

    return (
        x_pliegue,
        perfil_global
    )

def recortar_pagina_derecha(imagen, candidato, x_pliegue):
    alto_imagen, ancho_imagen = imagen.shape[:2]

    x_inicio = x_pliegue

    x_fin = candidato["x"] + candidato["ancho"]

    x_inicio = max(0, x_inicio)
    x_fin = min(ancho_imagen, x_fin)

    pagina = imagen[
        0:alto_imagen,
        x_inicio:x_fin
    ]

    return pagina

def detectar_esquinas_pagina(pagina):
    gris = cv2.cvtColor(pagina, cv2.COLOR_BGR2GRAY)

    gris = cv2.GaussianBlur(gris, (11, 11), 0)

    bordes = cv2.Canny(gris, 50, 150)

    contornos, _ = cv2.findContours(
        bordes,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidatos = []

    alto, ancho = pagina.shape[:2]
    area_imagen = ancho * alto

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if area < area_imagen * 0.10:
            continue

        perimetro = cv2.arcLength(contorno, True)

        aproximacion = cv2.approxPolyDP(
            contorno,
            0.03 * perimetro,
            True
        )

        if len(aproximacion) == 4:
            candidatos.append({
                "contorno": contorno,
                "aproximacion": aproximacion,
                "area": area
            })

    candidatos.sort(
        key=lambda candidato: candidato["area"],
        reverse=True
    )

    return candidatos, bordes

def detectar_bordes_pagina(pagina):
    gris = cv2.cvtColor(pagina, cv2.COLOR_BGR2GRAY)

    # Suavizamos bastante para reducir el efecto del texto
    # y de las líneas internas de la tabla.
    gris = cv2.GaussianBlur(gris, (31, 31), 0)

    alto, ancho = gris.shape

    # ---------------------------------------------------------
    # BORDE IZQUIERDO Y DERECHO
    # ---------------------------------------------------------

    # Promedio vertical:
    # para cada columna calculamos el brillo medio.
    perfil_columnas = gris.mean(axis=0)

    # ---------------------------------------------------------
    # BORDE SUPERIOR E INFERIOR
    # ---------------------------------------------------------

    # Promedio horizontal:
    # para cada fila calculamos el brillo medio.
    perfil_filas = gris.mean(axis=1)

    # Gradiente de los perfiles.
    gradiente_columnas = np.abs(
        np.gradient(perfil_columnas)
    )

    gradiente_filas = np.abs(
        np.gradient(perfil_filas)
    )

    # Suavizamos los perfiles de gradiente.
    gradiente_columnas = cv2.GaussianBlur(
        gradiente_columnas.astype(np.float32),
        (31, 1),
        0
    ).reshape(-1)

    gradiente_filas = cv2.GaussianBlur(
        gradiente_filas.astype(np.float32),
        (31, 1),
        0
    ).reshape(-1)

    # ---------------------------------------------------------
    # IGNORAMOS LOS EXTREMOS
    # ---------------------------------------------------------
    # Evitamos que el límite artificial de nuestro propio
    # recorte sea interpretado como borde de página.

    margen_x = int(ancho * 0.05)
    margen_y = int(alto * 0.05)

    gradiente_columnas[:margen_x] = 0
    gradiente_columnas[-margen_x:] = 0

    gradiente_filas[:margen_y] = 0
    gradiente_filas[-margen_y:] = 0

    # ---------------------------------------------------------
    # BUSCAR LOS BORDES
    # ---------------------------------------------------------

    x_izquierdo = int(np.argmax(gradiente_columnas))

    gradiente_derecho = gradiente_columnas.copy()

    # No queremos volver a detectar el mismo borde.
    zona_exclusion = int(ancho * 0.15)

    inicio_derecho = min(
        x_izquierdo + zona_exclusion,
        ancho - margen_x
    )

    gradiente_derecho[:inicio_derecho] = 0

    x_derecho = int(np.argmax(gradiente_derecho))

    y_superior = int(np.argmax(gradiente_filas))

    gradiente_inferior = gradiente_filas.copy()

    zona_exclusion = int(alto * 0.15)

    inicio_inferior = min(
        y_superior + zona_exclusion,
        alto - margen_y
    )

    gradiente_inferior[:inicio_inferior] = 0

    y_inferior = int(np.argmax(gradiente_inferior))

    return {
        "x_izquierdo": x_izquierdo,
        "x_derecho": x_derecho,
        "y_superior": y_superior,
        "y_inferior": y_inferior,
        "perfil_columnas": gradiente_columnas,
        "perfil_filas": gradiente_filas
    }

def analizar_borde_derecho_por_franjas(pagina):
    gris = cv2.cvtColor(pagina, cv2.COLOR_BGR2GRAY)

    gris = cv2.GaussianBlur(
        gris,
        (21, 21),
        0
    )

    alto, ancho = gris.shape

    numero_franjas = 8
    resultados = []

    altura_franja = alto // numero_franjas

    for i in range(numero_franjas):

        y_inicio = i * altura_franja
        y_fin = alto if i == numero_franjas - 1 else (i + 1) * altura_franja

        franja = gris[y_inicio:y_fin, :]

        perfil = franja.mean(axis=0)

        gradiente = np.abs(
            np.gradient(perfil)
        )

        gradiente = cv2.GaussianBlur(
            gradiente.astype(np.float32),
            (21, 1),
            0
        ).reshape(-1)

        # Ignoramos los primeros 15% porque ahí
        # estamos demasiado cerca del pliegue.
        inicio = int(ancho * 0.15)

        # También ignoramos los últimos 5%.
        fin = int(ancho * 0.95)

        zona = gradiente[inicio:fin]

        indice = int(np.argmax(zona))

        x = inicio + indice

        resultados.append({
            "franja": i + 1,
            "y_inicio": y_inicio,
            "y_fin": y_fin,
            "x": x,
            "fuerza": float(gradiente[x])
        })

    return resultados

def detectar_inclinacion_pagina(pagina):
    gris = cv2.cvtColor(pagina, cv2.COLOR_BGR2GRAY)

    gris = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    bordes = cv2.Canny(
        gris,
        50,
        150
    )

    alto, ancho = gris.shape

    # Tomamos únicamente bandas cercanas a la parte
    # superior e inferior de la página.
    altura_banda = int(alto * 0.20)

    banda_superior = bordes[
        :altura_banda,
        :
    ]

    banda_inferior = bordes[
        alto - altura_banda:,
        :
    ]

    # Hough probabilístico.
    lineas_superior = cv2.HoughLinesP(
        banda_superior,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=int(ancho * 0.30),
        maxLineGap=30
    )

    lineas_inferior = cv2.HoughLinesP(
        banda_inferior,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=int(ancho * 0.30),
        maxLineGap=30
    )

    angulos = []

    for conjunto in [lineas_superior, lineas_inferior]:

        if conjunto is None:
            continue

        for linea in conjunto:
            x1, y1, x2, y2 = linea

            angulo = np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )

            # Solo nos interesan líneas
            # aproximadamente horizontales.
            if abs(angulo) < 20:
                angulos.append(angulo)

    if not angulos:
        return None, []

    angulo_medio = float(np.median(angulos))

    return angulo_medio, angulos

def corregir_inclinacion(pagina, angulo):
    alto, ancho = pagina.shape[:2]

    centro = (ancho // 2, alto // 2)

    matriz = cv2.getRotationMatrix2D(
        centro,
        angulo,
        1.0
    )

    pagina_rotada = cv2.warpAffine(
        pagina,
        matriz,
        (ancho, alto),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return pagina_rotada

def crear_mascara_pagina(pagina):

    gris = cv2.cvtColor(
        pagina,
        cv2.COLOR_BGR2GRAY
    )

    gris = cv2.GaussianBlur(
        gris,
        (15, 15),
        0
    )

    _, mascara = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (31, 31)
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mascara

def encontrar_contorno_pagina(mascara):

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return None

    contorno = max(
        contornos,
        key=cv2.contourArea
    )

    return contorno

def analizar_contorno_derecho(contorno):

    puntos = contorno.reshape(-1, 2)

    x_max = puntos[:, 0].max()

    # Nos quedamos con los puntos que están
    # cerca del extremo derecho del contorno.
    limite = x_max - 15

    puntos_derechos = puntos[
        puntos[:, 0] >= limite
    ]

    if len(puntos_derechos) == 0:
        return []

    resultados = []

    ys = np.linspace(
        0,
        puntos[:, 1].max(),
        10
    )

    for i in range(len(ys) - 1):

        y_inicio = ys[i]
        y_fin = ys[i + 1]

        franja = puntos_derechos[
            (puntos_derechos[:, 1] >= y_inicio) &
            (puntos_derechos[:, 1] < y_fin)
        ]

        if len(franja) == 0:
            continue

        x_promedio = float(
            franja[:, 0].mean()
        )

        y_promedio = float(
            franja[:, 1].mean()
        )

        resultados.append({
            "y": y_promedio,
            "x": x_promedio
        })

    return resultados

def analizar_mascara_por_filas(mascara):
    alto, ancho = mascara.shape

    resultados = []

    for y in range(alto):
        fila = mascara[y]

        pixeles_blancos = cv2.countNonZero(fila)

        resultados.append({
            "y": y,
            "blancos": pixeles_blancos
        })

    return resultados

def ordenar_puntos(puntos):
    puntos = puntos.reshape(4, 2)

    suma = puntos.sum(axis=1)
    diferencia = puntos[:, 0] - puntos[:, 1]

    superior_izquierdo = puntos[suma.argmin()]
    inferior_derecho = puntos[suma.argmax()]

    superior_derecho = puntos[diferencia.argmax()]
    inferior_izquierdo = puntos[diferencia.argmin()]

    return (
        superior_izquierdo,
        superior_derecho,
        inferior_derecho,
        inferior_izquierdo
    )


def corregir_perspectiva(imagen, puntos):
    superior_izquierdo, superior_derecho, \
    inferior_derecho, inferior_izquierdo = ordenar_puntos(
        puntos
    )

    ancho_superior = (
        ((superior_derecho[0] - superior_izquierdo[0]) ** 2 +
         (superior_derecho[1] - superior_izquierdo[1]) ** 2)
        ** 0.5
    )

    ancho_inferior = (
        ((inferior_derecho[0] - inferior_izquierdo[0]) ** 2 +
         (inferior_derecho[1] - inferior_izquierdo[1]) ** 2)
        ** 0.5
    )

    ancho_salida = int(
        max(ancho_superior, ancho_inferior)
    )

    alto_derecho = (
        ((inferior_derecho[0] - superior_derecho[0]) ** 2 +
         (inferior_derecho[1] - superior_derecho[1]) ** 2)
        ** 0.5
    )

    alto_izquierdo = (
        ((inferior_izquierdo[0] - superior_izquierdo[0]) ** 2 +
         (inferior_izquierdo[1] - superior_izquierdo[1]) ** 2)
        ** 0.5
    )

    alto_salida = int(
        max(alto_derecho, alto_izquierdo)
    )

    destino = np.array([
        [0, 0],
        [ancho_salida - 1, 0],
        [ancho_salida - 1, alto_salida - 1],
        [0, alto_salida - 1]
    ], dtype="float32")

    origen = np.array([
        superior_izquierdo,
        superior_derecho,
        inferior_derecho,
        inferior_izquierdo
    ], dtype="float32")

    matriz = cv2.getPerspectiveTransform(
        origen,
        destino
    )

    rectificada = cv2.warpPerspective(
        imagen,
        matriz,
        (ancho_salida, alto_salida)
    )

    return rectificada

def detectar_borde_inferior(mascara):
    alto, ancho = mascara.shape

    perfil_filas = []

    for y in range(alto):
        blancos = cv2.countNonZero(mascara[y])
        perfil_filas.append(blancos)

    perfil_filas = np.array(perfil_filas, dtype=np.float32)

    # Suavizamos para evitar pequeñas variaciones
    perfil_suavizado = cv2.GaussianBlur(
        perfil_filas.reshape(-1, 1),
        (1, 31),
        0
    ).reshape(-1)

    # Empezamos a buscar solamente en el 70% inferior
    inicio = int(alto * 0.70)

    # Buscamos la primera zona donde la cantidad de píxeles
    # blancos comienza a caer fuertemente hacia el fondo.
    diferencias = np.diff(perfil_suavizado)

    mejor_y = None
    mejor_caida = 0

    for y in range(inicio, alto - 1):

        caida = perfil_suavizado[y] - perfil_suavizado[y + 1]

        if caida > mejor_caida:
            mejor_caida = caida
            mejor_y = y

    if mejor_y is None:
        mejor_y = alto - 1

    return mejor_y, perfil_filas

def detectar_borde_superior(mascara):
    alto, ancho = mascara.shape

    perfil_filas = []

    for y in range(alto):
        blancos = cv2.countNonZero(mascara[y])
        perfil_filas.append(blancos)

    perfil_filas = np.array(
        perfil_filas,
        dtype=np.float32
    )

    # Buscamos la primera fila donde la máscara
    # ocupa prácticamente toda la anchura de la página.
    umbral = ancho * 0.90

    for y in range(alto):
        if perfil_filas[y] >= umbral:
            return y, perfil_filas

    return 0, perfil_filas

def obtener_esquinas_pagina(contorno):

    perimetro = cv2.arcLength(
        contorno,
        True
    )

    aproximacion = cv2.approxPolyDP(
        contorno,
        0.02 * perimetro,
        True
    )

    if len(aproximacion) == 4:

        return (
            aproximacion.reshape(4, 2),
            "approxPolyDP"
        )

    rectangulo = cv2.minAreaRect(
        contorno
    )

    puntos = cv2.boxPoints(
        rectangulo
    )

    puntos = np.int32(puntos)

    return (
        puntos,
        "minAreaRect"
    )

def preparar_imagen_ocr(imagen):
    """
    Prepara una imagen rectificada para las siguientes
    etapas de OCR.

    Por ahora solamente:
    1. Convierte a escala de grises.
    2. Estima la iluminación del fondo.
    3. Normaliza la iluminación.
    4. Mejora ligeramente el contraste.

    No realiza todavía binarización agresiva.
    """

    imagen_ampliada = cv2.resize(
        imagen,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------
    # 1. Escala de grises
    # --------------------------------------------------

    gris = cv2.cvtColor(
        imagen_ampliada,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # 2. Estimar iluminación del documento
    # --------------------------------------------------

    iluminacion = cv2.GaussianBlur(
        gris,
        (0, 0),
        35
    )

    # --------------------------------------------------
    # 3. Evitar división por cero
    # --------------------------------------------------

    iluminacion = np.maximum(
        iluminacion,
        1
    )

    # --------------------------------------------------
    # 4. Normalizar iluminación
    # --------------------------------------------------

    normalizada = cv2.divide(
        gris,
        iluminacion,
        scale=180
    )

    # --------------------------------------------------
    # 5. Mejorar contraste local
    # --------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contraste = clahe.apply(
        normalizada
    )

    return (
        gris,
        normalizada,
        contraste
    )

def eliminar_lineas_horizontales(imagen):
    """
    Detecta líneas horizontales largas del formulario
    y las elimina mediante inpainting.

    La imagen original no se modifica.
    """

    gris = imagen.copy()

    # Si por alguna razón llega una imagen a color
    if len(gris.shape) == 3:
        gris = cv2.cvtColor(gris, cv2.COLOR_BGR2GRAY)

    # Convertimos a blanco/negro para detectar las líneas
    _, binaria = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Kernel horizontal.
    # 120 píxeles porque estamos trabajando con la imagen
    # ampliada 2x (4448 x 3312).
    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (120, 1)
    )

    # Conserva solamente estructuras horizontales largas
    lineas = cv2.morphologyEx(
        binaria,
        cv2.MORPH_OPEN,
        kernel_horizontal
    )

    # Engrosamos ligeramente la máscara para cubrir
    # completamente el grosor de las líneas.
    kernel_dilatacion = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    lineas = cv2.dilate(
        lineas,
        kernel_dilatacion,
        iterations=1
    )

    # Rellenamos las zonas donde estaban las líneas
    imagen_sin_lineas = cv2.inpaint(
        gris,
        lineas,
        3,
        cv2.INPAINT_TELEA
    )

    return imagen_sin_lineas


def detectar_lineas_horizontales(imagen):
    """
    Detecta segmentos aproximadamente horizontales.

    Filtra por longitud e inclinación para reducir
    los segmentos irrelevantes.
    """

    gris = imagen.copy()

    if len(gris.shape) == 3:
        gris = cv2.cvtColor(
            gris,
            cv2.COLOR_BGR2GRAY
        )

    alto, ancho = gris.shape[:2]

    # Reducir ruido sin borrar demasiado los caracteres.
    gris = cv2.GaussianBlur(
        gris,
        (3, 3),
        0
    )

    bordes = cv2.Canny(
        gris,
        50,
        150
    )

    lineas = cv2.HoughLinesP(
        bordes,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=max(100, int(ancho * 0.08)),
        maxLineGap=30
    )

    if lineas is None:
        return []

    lineas_filtradas = []

    for linea in lineas:
        # HoughLinesP entrega normalmente [[x1, y1, x2, y2]].
        x1, y1, x2, y2 = linea.reshape(-1, 4)[0]

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angulo = abs(
            math.degrees(math.atan2(dy, dx))
        )

        longitud = math.hypot(dx, dy)

        # Conservar segmentos largos y casi horizontales.
        if angulo <= 5 and longitud >= ancho * 0.08:
            lineas_filtradas.append(
                (int(x1), int(y1), int(x2), int(y2))
            )

    return lineas_filtradas


def dibujar_lineas_horizontales(imagen, lineas):
    """
    Dibuja los segmentos detectados para inspección visual.
    Acepta una lista de tuplas (x1, y1, x2, y2).
    """

    if len(imagen.shape) == 2:
        resultado = cv2.cvtColor(
            imagen.copy(),
            cv2.COLOR_GRAY2BGR
        )
    else:
        resultado = imagen.copy()

    if lineas is None:
        return resultado

    for linea in lineas:
        x1, y1, x2, y2 = linea

        cv2.line(
            resultado,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

    return resultado


def eliminar_lineas_horizontales(imagen):
    """
    Detecta líneas horizontales largas y las elimina
    de una imagen destinada al OCR.
    """
    import cv2
    import numpy as np

    # Convertir a escala de grises si la imagen es a color.
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    # Crear una imagen binaria: texto y trazos oscuros en blanco.
    binaria = cv2.adaptiveThreshold(
        gris,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12
    )

    # Buscar trazos horizontales largos.
    ancho_kernel = max(40, gris.shape[1] // 35)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (ancho_kernel, 1)
    )

    lineas = cv2.morphologyEx(
        binaria,
        cv2.MORPH_OPEN,
        kernel
    )

    # Engrosar ligeramente la máscara para cubrir los trazos.
    kernel_dilatacion = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 1)
    )

    lineas = cv2.dilate(
        lineas,
        kernel_dilatacion,
        iterations=1
    )

    # Eliminar las líneas detectadas, rellenando sus píxeles en blanco.
    resultado = gris.copy()
    resultado[lineas > 0] = 255

    return resultado, lineas