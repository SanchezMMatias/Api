# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import time
import logging
from reconocimiento import SistemaReconocimientoFacial

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Reconocimiento Facial",
    version="1.0.0"
)

# Configurar CORS para Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verificar que existe la imagen de referencia
imagen_referencia = 'my_face.jpg'

# Inicializar el sistema con manejo de errores
try:
    sistema = SistemaReconocimientoFacial(imagen_referencia)
    logger.info("Sistema de reconocimiento inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar sistema: {e}")
    sistema = None

@app.get("/")
async def root():
    return {
        "message": "API de Reconocimiento Facial activa", 
        "status": "OK",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "imagen_referencia_existe": os.path.exists(imagen_referencia),
        "sistema_inicializado": sistema is not None,
        "timestamp": time.time()
    }

@app.post("/analizar_frame")
async def analizar_frame(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        logger.info(f"Iniciando análisis de imagen: {file.filename}")
        
        # Verificar que el sistema esté inicializado
        if sistema is None:
            raise HTTPException(
                status_code=500, 
                detail="Sistema de reconocimiento no inicializado"
            )
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.warning(f"Tipo de archivo inválido: {file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"El archivo debe ser una imagen. Recibido: {file.content_type}"
            )
        
        # Leer contenido del archivo
        contenido = await file.read()
        logger.info(f"Archivo leído: {len(contenido)} bytes")
        
        if len(contenido) == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        
        # Optimización: Limitar tamaño máximo de imagen (5MB)
        max_size = 5 * 1024 * 1024  # 5MB
        if len(contenido) > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Imagen demasiado grande. Máximo: {max_size//1024//1024}MB"
            )
        
        # Decodificar imagen
        try:
            npimg = np.frombuffer(contenido, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error al decodificar imagen: {e}")
            raise HTTPException(
                status_code=400, 
                detail="Error al decodificar la imagen"
            )
        
        if img is None:
            raise HTTPException(
                status_code=400, 
                detail="No se pudo decodificar la imagen. Verifica que sea un formato válido (JPEG, PNG)"
            )
        
        # Optimización: Redimensionar imagen si es muy grande
        height, width = img.shape[:2]
        max_dimension = 800  # Máximo 800px en cualquier dimensión
        
        if max(height, width) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Imagen redimensionada de {width}x{height} a {new_width}x{new_height}")
        
        logger.info(f"Imagen procesada: {img.shape}")
        
        # Realizar reconocimiento facial
        proceso_start = time.time()
        resultado = sistema.verificar_rostro(img)
        proceso_time = time.time() - proceso_start
        
        logger.info(f"Procesamiento completado en {proceso_time:.2f}s")
        
        if resultado["error"]:
            logger.error(f"Error en reconocimiento: {resultado['error']}")
            raise HTTPException(status_code=500, detail=resultado["error"])
        
        total_time = time.time() - start_time
        
        response = {
            "verified": resultado["verified"],
            "distance": float(resultado["distance"]),
            "status": "success",
            "processing_time": round(proceso_time, 2),
            "total_time": round(total_time, 2),
            "image_size": f"{img.shape[1]}x{img.shape[0]}",
            "message": "Rostro verificado exitosamente" if resultado["verified"] else "Rostro no reconocido"
        }
        
        logger.info(f"Respuesta enviada: verified={resultado['verified']}, distance={resultado['distance']:.3f}, time={total_time:.2f}s")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Error interno después de {total_time:.2f}s: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno del servidor: {str(e)}"
        )

# Endpoint adicional para test rápido
@app.get("/test")
async def test():
    return {
        "status": "API funcionando",
        "sistema_listo": sistema is not None,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Iniciando servidor en puerto {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        # Configuraciones para producción
        access_log=True,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )