# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import os
from reconocimiento import SistemaReconocimientoFacial

app = FastAPI(
    title="API de Reconocimiento Facial",
    version="1.0.0"
)

# Verificar que existe la imagen de referencia
imagen_referencia = 'my_face.jpg'
sistema = SistemaReconocimientoFacial(imagen_referencia)

@app.get("/")
async def root():
    return {"message": "API de Reconocimiento Facial activa", "status": "OK"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "imagen_referencia_existe": os.path.exists(imagen_referencia)
    }

@app.post("/analizar_frame")
async def analizar_frame(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        contenido = await file.read()
        
        if len(contenido) == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        
        npimg = np.frombuffer(contenido, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
        
        resultado = sistema.verificar_rostro(img)
        
        if resultado["error"]:
            raise HTTPException(status_code=500, detail=resultado["error"])
        
        return {
            "verified": resultado["verified"],
            "distance": resultado["distance"],
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)