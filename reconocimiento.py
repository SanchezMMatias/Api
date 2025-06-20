# reconocimiento.py
import cv2
import numpy as np
from deepface import DeepFace
import os
import logging
import time
from functools import lru_cache
import pickle

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SistemaReconocimientoFacial:
    def __init__(self, ruta_imagen_referencia='my_face.jpg'):
        self.ruta_imagen_referencia = ruta_imagen_referencia
        self.modelo = 'Facenet'  # Modelo más rápido que VGG-Face
        self.detector_backend = 'opencv'  # Detector más rápido
        self.threshold = 0.4  # Umbral de similitud
        
        # Cache para embeddings
        self.embedding_cache_file = 'embedding_cache.pkl'
        self.embedding_referencia = None
        
        # Verificar que existe la imagen de referencia
        if not os.path.exists(ruta_imagen_referencia):
            logger.warning(f"⚠️ Imagen de referencia no encontrada: {ruta_imagen_referencia}")
            raise FileNotFoundError(f"Imagen de referencia no encontrada: {ruta_imagen_referencia}")
        else:
            logger.info(f"✅ Imagen de referencia encontrada: {ruta_imagen_referencia}")
            
        # Pre-cargar embedding de la imagen de referencia
        self._cargar_embedding_referencia()
        
    def _cargar_embedding_referencia(self):
        """Cargar o generar embedding de la imagen de referencia"""
        try:
            # Intentar cargar embedding desde cache
            if os.path.exists(self.embedding_cache_file):
                with open(self.embedding_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data.get('ruta_imagen') == self.ruta_imagen_referencia:
                        self.embedding_referencia = cache_data['embedding']
                        logger.info("✅ Embedding de referencia cargado desde cache")
                        return
            
            # Generar nuevo embedding
            logger.info("Generando embedding de imagen de referencia...")
            start_time = time.time()
            
            # Usar represent en lugar de verify para obtener solo el embedding
            embedding = DeepFace.represent(
                img_path=self.ruta_imagen_referencia,
                model_name=self.modelo,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            self.embedding_referencia = np.array(embedding[0]["embedding"])
            
            # Guardar en cache
            cache_data = {
                'ruta_imagen': self.ruta_imagen_referencia,
                'embedding': self.embedding_referencia,
                'modelo': self.modelo,
                'timestamp': time.time()
            }
            
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Embedding generado y guardado en cache ({elapsed_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Error al cargar embedding de referencia: {e}")
            raise
    
    def _calcular_distancia_euclidiana(self, embedding1, embedding2):
        """Calcular distancia euclidiana entre dos embeddings"""
        return np.linalg.norm(embedding1 - embedding2)
    
    def _preprocesar_imagen(self, imagen):
        """Preprocesar imagen para acelerar el análisis"""
        if imagen is None or imagen.size == 0:
            return None
            
        # Redimensionar para acelerar procesamiento
        h, w = imagen.shape[:2]
        max_size = 300  # Reducido para mayor velocidad
        
        if max(h, w) > max_size:
            if w > h:
                nuevo_w = max_size
                nuevo_h = int(h * (max_size / w))
            else:
                nuevo_h = max_size
                nuevo_w = int(w * (max_size / h))
            
            imagen = cv2.resize(imagen, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Imagen redimensionada a {nuevo_w}x{nuevo_h}")
        
        return imagen
    
    def verificar_rostro(self, imagen):
        """
        Verificación facial optimizada usando embeddings pre-calculados
        """
        start_time = time.time()
        
        try:
            # Validar entrada
            if imagen is None or imagen.size == 0:
                return {
                    "verified": False,
                    "distance": 1.0,
                    "error": "Imagen de entrada inválida",
                    "processing_time": 0
                }
            
            if self.embedding_referencia is None:
                return {
                    "verified": False,
                    "distance": 1.0,
                    "error": "Embedding de referencia no disponible",
                    "processing_time": 0
                }
            
            # Preprocesar imagen
            imagen_procesada = self._preprocesar_imagen(imagen)
            if imagen_procesada is None:
                return {
                    "verified": False,
                    "distance": 1.0,
                    "error": "Error al preprocesar imagen",
                    "processing_time": time.time() - start_time
                }
            
            logger.info("Generando embedding de imagen de entrada...")
            
            # Generar embedding de la imagen de entrada
            embedding_entrada = DeepFace.represent(
                img_path=imagen_procesada,
                model_name=self.modelo,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            if not embedding_entrada:
                return {
                    "verified": False,
                    "distance": 1.0,
                    "error": "No se pudo detectar rostro en la imagen",
                    "processing_time": time.time() - start_time
                }
            
            # Calcular distancia
            embedding_array = np.array(embedding_entrada[0]["embedding"])
            distancia = self._calcular_distancia_euclidiana(
                self.embedding_referencia, 
                embedding_array
            )
            
            # Verificar si la distancia está dentro del umbral
            verificado = distancia <= self.threshold
            
            processing_time = time.time() - start_time
            
            logger.info(f"Verificación completada: {verificado}, distancia: {distancia:.4f}, tiempo: {processing_time:.2f}s")
            
            return {
                "verified": verificado,
                "distance": float(distancia),
                "error": None,
                "processing_time": processing_time,
                "threshold": self.threshold
            }
            
        except Exception as e:
            error_msg = str(e)
            processing_time = time.time() - start_time
            logger.error(f"Error en verificación facial: {error_msg} (tiempo: {processing_time:.2f}s)")
            
            return {
                "verified": False,
                "distance": 1.0,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def verificar_rostro_rapido(self, imagen):
        """
        Versión ultra-rápida usando solo OpenCV para detección básica
        Útil para pruebas rápidas o cuando DeepFace es muy lento
        """
        try:
            # Detector de rostros de OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convertir a escala de grises
            if len(imagen.shape) == 3:
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagen
            
            # Detectar rostros
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Si se detecta al menos un rostro, considerarlo como "verificado"
            # (Esto es muy básico, solo para pruebas de velocidad)
            rostro_detectado = len(faces) > 0
            
            return {
                "verified": rostro_detectado,
                "distance": 0.5 if rostro_detectado else 1.0,
                "error": None,
                "method": "opencv_basic",
                "faces_detected": len(faces)
            }
            
        except Exception as e:
            return {
                "verified": False,
                "distance": 1.0,
                "error": str(e),
                "method": "opencv_basic"
            }
    
    def limpiar_cache(self):
        """Limpiar cache de embeddings"""
        try:
            if os.path.exists(self.embedding_cache_file):
                os.remove(self.embedding_cache_file)
                logger.info("Cache de embeddings limpiado")
        except Exception as e:
            logger.error(f"Error al limpiar cache: {e}")
    
    def obtener_info_sistema(self):
        """Obtener información del sistema de reconocimiento"""
        return {
            "modelo": self.modelo,
            "detector_backend": self.detector_backend,
            "threshold": self.threshold,
            "imagen_referencia": self.ruta_imagen_referencia,
            "imagen_referencia_existe": os.path.exists(self.ruta_imagen_referencia),
            "embedding_cache_existe": os.path.exists(self.embedding_cache_file),
            "embedding_referencia_cargado": self.embedding_referencia is not None
        }