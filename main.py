from fastapi import FastAPI
from api.engine_routes import router as engine_router
from api.library_routes import router as library_router
from api.queue_routes import router as queue_router
from api.smart_routes import router as smart_router
from api.queue_state_routes import router as queue_state_router
from api.queue_smart_render_routes import router as queue_smart_render_router
from api.setlist_routes import router as setlist_router
from api.playback_routes import router as playback_router
from api.smart_set_routes import router as smart_set_router

app = FastAPI(title="Headless Audio Engine")

app.include_router(engine_router, prefix="/v1/autodj")
app.include_router(library_router,  prefix="/v1/autodj")
app.include_router(queue_router,  prefix="/v1/autodj")
app.include_router(smart_router, prefix="/v1/autodj")
app.include_router(queue_state_router, prefix="/v1/autodj")
app.include_router(queue_smart_render_router, prefix="/v1/autodj")
app.include_router(setlist_router, prefix="/v1/autodj")
app.include_router(playback_router, prefix="/v1/autodj")
app.include_router(smart_set_router, prefix="/v1/autodj")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)