from fastapi import FastAPI
from api.engine_routes import router as engine_router
from api.library_routes import router as library_router

app = FastAPI(title="Headless Audio Engine")

app.include_router(engine_router, prefix="/v1/autodj")
app.include_router(library_router,  prefix="/v1/autodj")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)