import os
import uuid
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from predictor import DepthEstimationModel

app = FastAPI(title="Depth Estimation API", version="1.0.0")

try:
    depth_estimator = DepthEstimationModel()
    print("Depth estimation model loaded successfully")
except Exception as e:
    print(f"Failed to load depth estimation model: {e}")
    depth_estimator = None

TEMP_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def cleanup_file(file_path: str):
    """Dosyayı güvenli şekilde sil"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Could not delete {file_path}: {e}")

@app.get("/")
async def root():
    """API durumu"""
    return {
        "message": "Depth Estimation API",
        "status": "running",
        "model_loaded": depth_estimator is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    """
    Görüntüden derinlik haritası oluştur
    """
    if depth_estimator is None:
        raise HTTPException(
            status_code=503, 
            detail="Depth estimation model is not available"
        )
    
    input_path = None
    output_path = None
    
    try:
        # Dosya boyutu kontrolü
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Dosya uzantısı kontrolü
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Benzersiz dosya adları oluştur
        file_id = str(uuid.uuid4())
        input_filename = f"input_{file_id}{file_ext}"
        output_filename = f"output_{file_id}.png"
        
        input_path = os.path.join(TEMP_FOLDER, input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Dosyayı kaydet
        with open(input_path, "wb") as f:
            f.write(file_content)
        
        # Derinlik haritası oluştur
        result = depth_estimator.calculate_depthmap(input_path, output_path)
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to process the image. Please try again."
            )
        
        # Sonuç kontrolü
        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=500,
                detail="Depth map generation failed"
            )
        
        # NPY dosyası varsa onun da yolunu al
        npy_filename = f"output_{file_id}_raw.npy"
        npy_path = os.path.join(OUTPUT_FOLDER, npy_filename)
        
        response_data = {
            "success": True,
            "message": "Depth map generated successfully",
            "file_id": file_id,
            "original_filename": file.filename,
            "depth_map_url": f"/outputs/{output_filename}",
            "download_url": f"/download/{file_id}"
        }
        
        # NPY dosyası varsa ekle
        if os.path.exists(npy_path):
            response_data["raw_data_available"] = True
            response_data["raw_data_url"] = f"/outputs/{npy_filename}"
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        # Geçici dosyaları temizle
        cleanup_file(input_path)

@app.get("/download/{file_id}")
async def download_depth_map(file_id: str):
    """
    Oluşturulan derinlik haritasını indir
    """
    output_filename = f"output_{file_id}.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=output_path,
        filename=f"depth_map_{file_id}.png",
        media_type="image/png"
    )

@app.get("/download/{file_id}/raw")
async def download_raw_data(file_id: str):
    """
    Ham derinlik verilerini indir (.npy)
    """
    npy_filename = f"output_{file_id}_raw.npy"
    npy_path = os.path.join(OUTPUT_FOLDER, npy_filename)
    
    if not os.path.exists(npy_path):
        raise HTTPException(status_code=404, detail="Raw data file not found")
    
    return FileResponse(
        path=npy_path,
        filename=f"depth_data_{file_id}.npy",
        media_type="application/octet-stream"
    )

@app.delete("/cleanup/{file_id}")
async def cleanup_files(file_id: str):
    """
    Belirli bir file_id ile ilgili tüm dosyaları temizle
    """
    files_deleted = 0
    
    # PNG dosyasını sil
    png_path = os.path.join(OUTPUT_FOLDER, f"output_{file_id}.png")
    if os.path.exists(png_path):
        cleanup_file(png_path)
        files_deleted += 1
    
    # NPY dosyasını sil
    npy_path = os.path.join(OUTPUT_FOLDER, f"output_{file_id}_raw.npy")
    if os.path.exists(npy_path):
        cleanup_file(npy_path)
        files_deleted += 1
    
    return {
        "success": True,
        "message": f"Cleaned up {files_deleted} files for {file_id}"
    }

@app.get("/health")
async def health_check():
    """
    Sistem sağlık kontrolü
    """
    return {
        "status": "healthy",
        "model_loaded": depth_estimator is not None,
        "temp_folder_exists": os.path.exists(TEMP_FOLDER),
        "output_folder_exists": os.path.exists(OUTPUT_FOLDER)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)