"""Download the USGS 3DEP 1 m Cuprite DEM tiles and build a local VRT."""

from pathlib import Path
from urllib.request import Request, urlopen

from osgeo import gdal


SCRIPT_DIR = Path(__file__).resolve().parent
DEM_DIR = SCRIPT_DIR / "data" / "cuprite_dem"
DEM_VRT = DEM_DIR / "cuprite_usgs_1m.vrt"

# USGS 3DEP, NV_Southern_D23 (2025 acquisition; published 2026).
# Both tiles are needed for the current Cuprite flightline mosaic.  The target
# ROIs are in x48y416; the mosaic also reaches the western x47y416 tile.
DEM_TILES = {
    "USGS_1M_11_x47y416_NV_Southern_D23.tif": (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/"
        "NV_Southern_D23/TIFF/USGS_1M_11_x47y416_NV_Southern_D23.tif"
    ),
    "USGS_1M_11_x48y416_NV_Southern_D23.tif": (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/"
        "NV_Southern_D23/TIFF/USGS_1M_11_x48y416_NV_Southern_D23.tif"
    ),
}


def download(url: str, destination: Path) -> None:
    """Stream a tile to disk, keeping incomplete downloads out of the VRT."""
    partial = destination.with_suffix(destination.suffix + ".partial")
    resume_at = partial.stat().st_size if partial.exists() else 0
    request = Request(url, headers={"Range": f"bytes={resume_at}-"} if resume_at else {})
    print(f"[+] {'Resuming' if resume_at else 'Downloading'} {destination.name}")
    try:
        with urlopen(request, timeout=60) as response:
            append = resume_at > 0 and response.status == 206
            with partial.open("ab" if append else "wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> Path:
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    tile_paths = []
    for name, url in DEM_TILES.items():
        tile_path = DEM_DIR / name
        if not tile_path.exists():
            download(url, tile_path)
        else:
            print(f"[+] Using existing {tile_path.name}")
        tile_paths.append(tile_path)

    vrt = gdal.BuildVRT(str(DEM_VRT), [str(path.resolve()) for path in tile_paths])
    if vrt is None:
        raise RuntimeError("GDAL could not create the Cuprite DEM VRT.")
    vrt = None
    print(f"[+] Cuprite 1 m DEM VRT: {DEM_VRT}")
    return DEM_VRT


if __name__ == "__main__":
    main()
