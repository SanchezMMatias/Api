from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from reconocimiento import SistemaReconocimientoFacial

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Reconocimiento Facial Optimizada",
    version="2.1.0",
    description="API optimizada para Render sin GPU"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

imagen_referencia = 'my_face.jpg'
sistema = None
executor = ThreadPoolExecutor(max_workers=2)

async def inicializar_sistema():
    global sistema
    try:
        sistema = SistemaReconocimientoFacial(imagen_referencia)
        logger.info("✅ Sistema inicializado correctamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error al inicializar sistema: {e}")
        sistema = None
        return False

@app.on_event("startup")
async def startup_event():
    await inicializar_sistema()

@app.get("/")
async def root():
    return {
        "message": "API de Reconocimiento Facial Optimizada para Render",
        "status": "OK",
        "version": "2.1.0",
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

@app.post("/reinicializar")
async def reinicializar_sistema():
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
    try:
        npimg = np.frombuffer(contenido_imagen, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "No se pudo decodificar la imagen"}
        resultado = sistema.verificar_rostro(img)
        return resultado
    except Exception as e:
        return {"error": str(e)}

@app.post("/analizar_frame")
async def analizar_frame(
    file: UploadFile = File(...),
    modo_rapido: bool = Query(True, description="Usar modo rápido (por defecto en Render sin GPU)")
):
    start_time = time.time()
    try:
        logger.info(f"📸 Analizando imagen: {file.filename}, modo_rapido={modo_rapido}")

        if sistema is None:
            raise HTTPException(status_code=500, detail="Sistema no inicializado. Usa /reinicializar")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

        contenido = await file.read()
        if len(contenido) == 0:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        max_size = 2 * 1024 * 1024  # 2MB
        if len(contenido) > max_size:
            raise HTTPException(status_code=400, detail="Imagen demasiado grande. Máximo 2MB")

        if modo_rapido:
            npimg = np.frombuffer(contenido, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Error al decodificar imagen")
            resultado = sistema.verificar_rostro_rapido(img)
        else:
            loop = asyncio.get_event_loop()
            resultado = await loop.run_in_executor(executor, procesar_imagen_sync, contenido)

        if "error" in resultado and resultado["error"]:
            raise HTTPException(status_code=500, detail=resultado["error"])

        total_time = time.time() - start_time
        return {
            "verified": resultado["verified"],
            "distance": float(resultado["distance"]),
            "status": "success",
            "processing_time": round(resultado.get("processing_time", 0), 2),
            "total_time": round(total_time, 2),
            "modo_rapido": modo_rapido,
            "threshold": resultado.get("threshold", 0.4),
            "faces_detected": resultado.get("faces_detected", 0),
            "method": resultado.get("method", "opencv_basic" if modo_rapido else "deepface"),
            "message": "Rostro verificado exitosamente" if resultado["verified"] else "Rostro no reconocido",
            "timestamp": time.time()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error interno: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/analizar_frame_rapido")
async def analizar_frame_rapido(file: UploadFile = File(...)):
    return await analizar_frame(file, modo_rapido=True)

@app.get("/info")
async def sistema_info():
    if sistema is None:
        raise HTTPException(status_code=500, detail="Sistema no inicializado")
    return sistema.obtener_info_sistema()

@app.get("/test")
async def test():
    return {
        "status": "OK",
        "sistema_listo": sistema is not None,
        "timestamp": time.time()
    }

@app.get("/test_velocidad")
async def test_velocidad():
    if sistema is None:
        return {"error": "Sistema no inicializado"}
    try:
        img_test = np.zeros((100, 100, 3), dtype=np.uint8)
        img_test.fill(128)
        start_time = time.time()
        resultado = sistema.verificar_rostro_rapido(img_test)
        return {
            "test_completed": True,
            "elapsed_time": round(time.time() - start_time, 3),
            "resultado": resultado,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Servidor iniciando en puerto {port}")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True,
        timeout_keep_alive=60,
        timeout_graceful_shutdown=60,
        workers=1,
        limit_max_requests=100
    )
