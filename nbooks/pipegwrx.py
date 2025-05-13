from utilenames import tilenames_mkd,tilenames_tls
from os.path import join 
from os import makedirs
from glob import glob 
from uinterp import riofill
from ufuncs import get_raster_info,gdal_regrid
from uvars import gdsm_v_fn, gdtm_v_fn
import sys 
from uvars import topoxcale_dir 
sys.path.append(topoxcale_dir)
from topoxcale.mlxcale import mldownxcale
from topoxcale.sagaxcale import gwrdownxcale
import os 

tilenames =  tilenames_tls + tilenames_mkd

varname = "tdem_dem"
roitilenames = tilenames
outdir = "/home/ljp238/Downloads/mkg_sprint/ROI"
indir = "/media/ljp238/12TBWolf/BRCHIEVE/TILES12"
makedirs(outdir, exist_ok=True)
xres, yres = 0.01057350068885258912,-0.01057350068885258912 
t_epsg='EPSG:4326' # 4326 4979
mode="num"
varpath = f"{indir}/*/*{varname}.tif"
varfiles = glob(varpath)
roivarfiles = [fi for fi in varfiles for t in roitilenames if t in fi]

gdsmf_tiles = []
gdtmf_tiles = []
for fi in roivarfiles:
    tilename = fi.split('/')[-2]
    tile_dir = join(outdir,'TILES', tilename)
    #print(tile_dir)
    makedirs(tile_dir, exist_ok=True)
    _, _, _, xmin, xmax, ymin, ymax, _, _ = get_raster_info(fi)
    gdsmv_tile = f"{tile_dir}/{tilename}_gdsm_void.tif"
    gdtmv_tile = f"{tile_dir}/{tilename}_gdtm_void.tif"
    
    gdal_regrid(gdtm_v_fn, gdtmv_tile, xmin, ymin, xmax, ymax, xres, yres,mode, t_epsg, overwrite=False)
    gdal_regrid(gdsm_v_fn, gdsmv_tile, xmin, ymin, xmax, ymax, xres, yres,mode, t_epsg, overwrite=False)

    gdsmf_tile = gdsmv_tile.replace('void.tif', 'riofill.tif')
    gdtmf_tile = gdtmv_tile.replace('void.tif', 'riofill.tif')

    riofill(gdtmv_tile, gdtmf_tile, si=0)
    riofill(gdsmv_tile, gdsmf_tile, si=0)
    gdsmf_tiles.append(gdsmf_tile)
    gdtmf_tiles.append(gdtmf_tile)



sfix = "GWR"
for tilename in tilenames:
    
    xpath = f"/media/ljp238/12TBWolf/BRCHIEVE/TILES12/{tilename}/{tilename}_tdem_dem.tif"
    ypath = f"/home/ljp238/Downloads/mkg_sprint/ROI/TILES/{tilename}/{tilename}_gdtm_riofill_0.tif"
    out_path = ypath.replace('.tif', f'_{sfix}.tif')
    if not os.path.isfile(out_path):
        gwrdownxcale(xpath, ypath, out_path,oaux=False,epsg_code=4979, clean=False) 