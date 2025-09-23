"""Minimal test to debug the hanging issue."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return JSONResponse({"message": "Sparse service working"})

@app.get("/health")
async def health():
    logger.info("Health endpoint called")
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting minimal test server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
