# reconocimiento.py
import cv2
import numpy as np
from deepface import DeepFace
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SistemaReconocimientoFacial:
    def __init__(self, ruta_imagen_referencia='my_face.jpg'):
        self.ruta_imagen_referencia = ruta_imagen_referencia
        self.modelo = 'Facenet'
        self.detector_backend = 'opencv'
        
        # Verificar que existe la imagen de referencia
        if not os.path.exists(ruta_imagen_referencia):
            logger.warning(f"⚠️ Imagen de referencia no encontrada: {ruta_imagen_referencia}")
        else:
            logger.info(f"✅ Imagen de referencia cargada: {ruta_imagen_referencia}")
    
    def verificar_rostro(self, imagen):
        """
        Recibe una imagen (np.array BGR) y devuelve resultado de verificación facial.
        """
        try:
            # Verificar que la imagen de referencia existe
            if not os.path.exists(self.ruta_imagen_referencia):
                return {
                    "verified": False,
                    "distance": 1.0,
                    "error": f"Imagen de referencia no encontrada: {self.ruta_imagen_referencia}"
                }
            
            # Validar imagen de entrada
            if imagen is None or imagen.size == 0:
                return {
                    "verified": False,
                    "distance": 1.0,
                    "error": "Imagen de entrada inválida"
                }
            
            # Opcional: redimensionar imagen para acelerar procesamiento
            h, w = imagen.shape[:2]
            if w > 150 or h > 150:
                factor = min(150/w, 150/h)
                nuevo_w = int(w * factor)
                nuevo_h = int(h * factor)
                imagen = cv2.resize(imagen, (nuevo_w, nuevo_h))
                logger.info(f"Imagen redimensionada a {nuevo_w}x{nuevo_h}")
            
            # Realizar verificación facial
            resultado = DeepFace.verify(
                imagen,
                self.ruta_imagen_referencia,
                model_name=self.modelo,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            logger.info(f"Verificación completada: {resultado['verified']}, distancia: {resultado['distance']}")
            
            return {
                "verified": resultado["verified"],
                "distance": resultado["distance"],
                "error": None
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error en verificación facial: {error_msg}")
            return {
                "verified": False,
                "distance": 1.0,
                "error": error_msg
            }