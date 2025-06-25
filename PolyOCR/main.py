import uvicorn
import sys
import os

# Add the parent directory to Python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    uvicorn.run("PolyOCR.app:app", host="127.0.0.1", port=8000, reload=True)

