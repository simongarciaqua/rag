import uvicorn
from api.index import app

if __name__ == "__main__":
    print("Iniciando Aquaservice AI Local (v2.2)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
