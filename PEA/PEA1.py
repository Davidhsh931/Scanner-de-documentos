from flask import Flask, request, jsonify
import cv2
import pytesseract
import numpy as np
import re
import os

app = Flask(__name__)

# --- OPTIMIZACIÓN (Trabajo del Lunes 31/08/2026) ---
def preprocesar_imagen(img):
    """Mejora el contraste de la imagen para que pytesseract lea mejor."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar un threshold (umbral) para resaltar el texto negro sobre fondo blanco
    _, binaria = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binaria

# --- REFINAMIENTO DE EXTRACCIÓN (Trabajo del Miércoles 09/09/2026) ---
def extraer_datos(texto, anio_default=2026, mes_default=9, moneda_default="PEN"):
    """Usa expresiones regulares mejoradas para buscar montos y texto."""
    # Busca números con formato de moneda (ej. 1500.00 o 1500,00)
    montos = re.findall(r'\b\d+[\.,]\d{2}\b', texto)
    
    total_ingresos = float(montos[0].replace(',', '.')) if len(montos) > 0 else 0.0
    total_descuentos = float(montos[1].replace(',', '.')) if len(montos) > 1 else 0.0
    monto_liquido = float(montos[2].replace(',', '.')) if len(montos) > 2 else 0.0

    # Intento básico de buscar un nombre (se puede mejorar si la planilla tiene formato fijo)
    nombres_encontrados = re.findall(r'NOMBRE:\s*([A-Za-z\s]+)', texto)
    nombre_completo = nombres_encontrados[0].strip() if nombres_encontrados else 'POR_VERIFICAR'

    return {
        'anio': anio_default,
        'mes': mes_default,
        'nombres': nombre_completo,
        'apellidos': 'POR_VERIFICAR', # Se completará en la UI de Android (Jueves 17/09)
        'total_ingresos': total_ingresos,
        'total_descuentos': total_descuentos,
        'monto_liquido': monto_liquido,
        'moneda': moneda_default
    }

# --- CREACIÓN DE API REST (Trabajo del Viernes 04/09/2026) ---
@app.route('/api/v1/procesar-planilla', methods=['POST'])
def procesar_imagen_planilla():
    # 1. Verificar si la petición trae una imagen
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen en la petición'}), 400
    
    file = request.files['imagen']
    
    try:
        # 2. Leer la imagen directamente desde la memoria (sin guardarla en el disco)
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'El archivo enviado no es una imagen válida'}), 400

        # 3. Procesar OCR
        img_optimizada = preprocesar_imagen(img)
        texto_detectado = pytesseract.image_to_string(img_optimizada, lang='spa')
        
        print("--- Texto Detectado en Servidor ---")
        print(texto_detectado)

        # 4. Extraer información
        datos = extraer_datos(texto_detectado)

        # 5. Respuesta JSON a la aplicación Android (Trabajo del Jueves 10/09/2026)
        return jsonify({
            'status': 'success',
            'mensaje': 'Planilla procesada correctamente',
            'datos': datos
        }), 200

    except Exception as e:
        return jsonify({'error': f'Ocurrió un error en el servidor: {str(e)}'}), 500

if __name__ == '__main__':
    # Ejecuta el servidor en la red local (0.0.0.0 permite que el celular se conecte)
    # Puerto 5000 por defecto
    print("Iniciando API REST de Tesorería...")
    app.run(host='0.0.0.0', port=5000, debug=True)