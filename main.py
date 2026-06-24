import uvicorn
from fastapi import FastAPI
from gauge_metadata.router import router

app = FastAPI(
    title="Analog Gauge Metadata OCR Service",
    description=(
        "FastAPI service to extract analog gauge scale unit, min_value, and "
        "max_value using various OCR engines (EasyOCR, PaddleOCR, RapidOCR, Tesseract)."
    ),
    version="1.0.0",
)

# Register the router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

