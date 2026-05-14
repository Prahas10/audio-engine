from fastapi import APIRouter

from models.schemas import ScanTrackRequest, ScanFolderRequest
from core.library import (
    scan_and_save_track,
    scan_folder_metadata,
    list_library_tracks
)


router = APIRouter(prefix="/library", tags=["Library"])


@router.post("/scan-track")
async def scan_track(req: ScanTrackRequest):
    metadata = scan_and_save_track(
        track_path=req.track_path,
        library_path=req.library_path
    )

    return {
        "status": "success",
        "track": metadata
    }


@router.post("/scan-folder")
async def scan_folder(req: ScanFolderRequest):
    return scan_folder_metadata(
        folder_path=req.folder_path,
        library_path=req.library_path,
        force_rescan=req.force_rescan,
        clear_existing=req.clear_existing
    )


@router.get("/tracks")
async def get_library_tracks(library_path: str = "storage/library_metadata.json"):
    return list_library_tracks(library_path)