#!/usr/bin/env python3
"""
Simple server startup script for PolyOCR
"""
import uvicorn
import sys
import os

def main():
    print("🚀 Starting PolyOCR FastAPI Server...")
    # Get the absolute path of the project root directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    print(f"📁 Project directory: {project_root}")
    
    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        # Start the server
        uvicorn.run(
            "PolyOCR.app:app", 
            host="127.0.0.1", 
            port=8000, 
            reload=True,
            log_level="info",
            reload_dirs=[project_root]  # Watch for changes in the project directory
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory (`C:\\Users\\app27\\OneDrive\\Desktop\\New folder`)")
        print("2. Activate your virtual environment if you have one.")
        print("3. Install requirements: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
