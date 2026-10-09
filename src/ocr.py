import pytesseract


def realizar_ocr(imagen, psm=6):
    texto = pytesseract.image_to_string(
        imagen,
        lang="spa",
        config=f"--psm {psm}"
    )
    return texto


def realizar_ocr_con_coordenadas(imagen):

    datos = pytesseract.image_to_data(
        imagen,
        lang="spa",
        output_type=pytesseract.Output.DICT
    )

    palabras = []

    total = len(datos["text"])

    for i in range(total):

        texto = datos["text"][i].strip()

        if not texto:
            continue

        try:
            confianza = float(datos["conf"][i])
        except (ValueError, TypeError):
            confianza = -1

        palabras.append({
            "texto": texto,
            "confianza": confianza,

            "x": int(datos["left"][i]),
            "y": int(datos["top"][i]),
            "ancho": int(datos["width"][i]),
            "alto": int(datos["height"][i]),

            "bloque": int(datos["block_num"][i]),
            "parrafo": int(datos["par_num"][i]),
            "linea": int(datos["line_num"][i]),
            "palabra": int(datos["word_num"][i])
        })

    return palabras

def agrupar_palabras_por_linea(palabras, tolerancia_y=15):

    palabras_ordenadas = sorted(
        palabras,
        key=lambda palabra: (
            palabra["y"],
            palabra["x"]
        )
    )

    lineas = []

    for palabra in palabras_ordenadas:

        centro_y = palabra["y"] + palabra["alto"] / 2

        linea_encontrada = None

        for linea in lineas:

            diferencia_y = abs(
                centro_y - linea["centro_y"]
            )

            if diferencia_y <= tolerancia_y:
                linea_encontrada = linea
                break

        if linea_encontrada is None:

            nueva_linea = {
                "centro_y": centro_y,
                "palabras": [palabra]
            }

            lineas.append(nueva_linea)

        else:

            linea_encontrada["palabras"].append(palabra)

            centros = [
                p["y"] + p["alto"] / 2
                for p in linea_encontrada["palabras"]
            ]

            linea_encontrada["centro_y"] = sum(centros) / len(centros)

    for linea in lineas:

        linea["palabras"].sort(
            key=lambda palabra: palabra["x"]
        )

    return lineas