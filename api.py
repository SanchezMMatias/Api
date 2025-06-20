# api_optimizada.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from reconocimiento_optimizado import SistemaReconocimientoFacial

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Reconocimiento Facial Optimizada",
    version="2.0.0",
    description="API optimizada para reconocimiento facial con DeepFace"
)

# Configurar CORS para Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
imagen_referencia = 'my_face.jpg'
sistema = None
executor = ThreadPoolExecutor(max_workers=2)  # Pool de threads para procesamiento

# Inicializar el sistema con manejo de errores
async def inicializar_sistema():
    global sistema
    try:
        sistema = SistemaReconocimientoFacial(imagen_referencia)
        logger.info("✅ Sistema de reconocimiento inicializado correctamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error al inicializar sistema: {e}")
        sistema = None
        return False

# Inicializar al arranque
@app.on_event("startup")
async def startup_event():
    await inicializar_sistema()

@app.get("/")
async def root():
    return {
        "message": "API de Reconocimiento Facial Optimizada", 
        "status": "OK",
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    sistema_info = sistema.obtener_info_sistema() if sistema else {}
    
    return {
        "status": "healthy" if sistema is not None else "unhealthy",
        "imagen_referencia_existe": os.path.exists(imagen_referencia),
        "sistema_inicializado": sistema is not None,
        "sistema_info": sistema_info,
        "timestamp": time.time()
    }

@app.get("/info")
async def sistema_info():
    """Obtener información detallada del sistema"""
    if sistema is None:
        raise HTTPException(status_code=500, detail="Sistema no inicializado")
    
    return sistema.obtener_info_sistema()

@app.post("/reinicializar")
async def reinicializar_sistema():
    """Reinicializar el sistema de reconocimiento"""
    global sistema
    try:
        if sistema:
            sistema.limpiar_cache()
        
        success = await inicializar_sistema()
        if success:
            return {"message": "Sistema reinicializado correctamente", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Error al reinicializar sistema")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reinicializar: {str(e)}")

def procesar_imagen_sync(contenido_imagen):
    """Función síncrona para procesamiento en thread separado"""
    try:
        # Decodificar imagen
        npimg = np.frombuffer(contenido_imagen, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "No se pudo decodificar la imagen"}
        
        # Realizar reconocimiento
        resultado = sistema.verificar_rostro(img)
        return resultado
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/analizar_frame")
async def analizar_frame(
    file: UploadFile = File(...),
    modo_rapido: bool = Query(False, description="Usar modo rápido (menos preciso pero más veloz)")
):
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Iniciando análisis - Archivo: {file.filename}, Modo rápido: {modo_rapido}")
        
        # Verificar que el sistema esté inicializado
        if sistema is None:
            raise HTTPException(
                status_code=500, 
                detail="Sistema de reconocimiento no inicializado. Intenta /reinicializar"
            )
        
        # Validar archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"Archivo debe ser imagen. Recibido: {file.content_type}"
            )
        
        # Leer contenido
        contenido = await file.read()
        if len(contenido) == 0:
            raise HTTPException(status_code=400, detail="Archivo vacío")
        
        # Límite de tamaño (reducido para mayor velocidad)
        max_size = 2 * 1024 * 1024  # 2MB
        if len(contenido) > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Imagen muy grande. Máximo: {max_size//1024//1024}MB"
            )
        
        logger.info(f"📁 Archivo procesado: {len(contenido)} bytes")
        
        if modo_rapido:
            # Modo rápido: procesamiento síncrono básico
            npimg = np.frombuffer(contenido, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail="Error al decodificar imagen")
            
            resultado = sistema.verificar_rostro_rapido(img)
        else:
            # Modo normal: procesamiento en thread separado para evitar bloqueos
            loop = asyncio.get_event_loop()
            resultado = await loop.run_in_executor(
                executor, 
                procesar_imagen_sync, 
                contenido
            )
        
        if "error" in resultado and resultado["error"]:
            logger.error(f"❌ Error en procesamiento: {resultado['error']}")
            raise HTTPException(status_code=500, detail=resultado["error"])
        
        total_time = time.time() - start_time
        processing_time = resultado.get("processing_time", 0)
        
        response = {
            "verified": resultado["verified"],
            "distance": float(resultado["distance"]),
            "status": "success",
            "processing_time": round(processing_time, 2),
            "total_time": round(total_time, 2),
            "modo_rapido": modo_rapido,
            "threshold": resultado.get("threshold", 0.4),
            "message": "Rostro verificado exitosamente" if resultado["verified"] else "Rostro no reconocido",
            "timestamp": time.time()
        }
        
        # Agregar información adicional según el modo
        if modo_rapido:
            response["method"] = resultado.get("method", "opencv_basic")
            response["faces_detected"] = resultado.get("faces_detected", 0)
        
        logger.info(f"✅ Respuesta: verified={resultado['verified']}, distance={resultado['distance']:.3f}, total_time={total_time:.2f}s")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Error interno ({total_time:.2f}s): {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno: {str(e)}"
        )

@app.post("/analizar_frame_rapido")
async def analizar_frame_rapido(file: UploadFile = File(...)):
    """Endpoint específico para análisis rápido (solo detección básica)"""
    return await analizar_frame(file, modo_rapido=True)

@app.get("/test")
async def test():
    return {
        "status": "API funcionando",
        "sistema_listo": sistema is not None,
        "version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/test_velocidad")
async def test_velocidad():
    """Test rápido de velocidad del sistema"""
    if sistema is None:
        return {"error": "Sistema no inicializado"}
    
    try:
        # Crear imagen de prueba simple
        img_test = np.zeros((100, 100, 3), dtype=np.uint8)
        img_test.fill(128)  # Imagen gris
        
        start_time = time.time()
        resultado = sistema.verificar_rostro_rapido(img_test)
        elapsed_time = time.time() - start_time
        
        return {
            "test_completed": True,
            "elapsed_time": round(elapsed_time, 3),
            "resultado": resultado,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Iniciando servidor optimizado en puerto {port}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        # Configuraciones optimizadas
        access_log=True,
        timeout_keep_alive=60,  # Aumentado
        timeout_graceful_shutdown=60,  # Aumentado
        workers=1,  # Un solo worker para evitar problemas de memoria
        limit_max_requests=100,  # Reiniciar worker cada 100 requests
    )