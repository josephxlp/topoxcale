import os 
from osgeo import gdal, gdalconst
import rasterio 
import subprocess

gdal.UseExceptions()

def get_raster_info(tif_path):
    """
    Extracts raster metadata including projection, resolution, and bounding box.

    Parameters:
        tif_path (str): Path to the raster file.

    Returns:
        tuple: Raster projection, resolution, bounding box, and dimensions.
    """
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    xres = geotrans[1]
    yres = geotrans[5]
    w, h = ds.RasterXSize, ds.RasterYSize
    xmin, ymax = geotrans[0], geotrans[3]
    xmax = xmin + (xres * w)
    ymin = ymax + (yres * h)
    ds = None
    return proj, xres, yres, xmin, xmax, ymin, ymax, w, h

def get_nodata_value(raster_path):
    """
    Retrieves the NoData value of a raster.

    Parameters:
        raster_path (str): Path to the raster file.

    Returns:
        float: NoData value.
    """
    with rasterio.open(raster_path) as src:
        return src.nodata

def gdal_regrid(fi, fo, xmin, ymin, xmax, ymax, xres, yres,
                mode, t_epsg='EPSG:4979', overwrite=False):
    """
    Regrids a raster file using GDAL.

    Parameters:
        fi (str): Input raster file path.
        fo (str): Output raster file path.
        xmin, ymin, xmax, ymax (float): Bounding box.
        xres, yres (float): Target resolution.
        mode (str): Regridding mode ('num' or 'cat').
        t_epsg (str): Target EPSG code.
        overwrite (bool): Whether to overwrite existing output.

    Returns:
        None
    """
    if mode == 'num':
        ndv, algo, dtype = num_regrid_params()
    elif mode == 'cat':
        ndv, algo, dtype = cat_regrid_params()
    else:
        raise ValueError("Invalid mode. Use 'num' or 'cat'.")

    src_ndv = get_nodata_value(fi)
    dst_ndv = ndv

    print(f"Source NoData Value: {src_ndv}")
    print(f"Destination NoData Value: {dst_ndv}")

    overwrite_option = "-overwrite" if overwrite else ""
    output_width = round((xmax - xmin) / xres)
    output_height = round((ymax - ymin) / abs(yres))

    cmd = (f'gdalwarp -ot {dtype} -multi {overwrite_option} '
           f'-te {xmin} {ymin} {xmax} {ymax} '
          # f'-ts {output_width} {output_height} '
           f'-r {algo} -t_srs {t_epsg} -tr {xres} {yres} -tap '
           f'-co compress=lzw -co num_threads=all_cpus -co TILED=YES '
           f'-srcnodata {src_ndv} -dstnodata {dst_ndv} '
           f'{fi} {fo}')

    os.system(cmd)

def cat_regrid_params():
    """
    Returns parameters for categorical regridding.

    Returns:
        tuple: NoData value, resampling algorithm, and data type.
    """
    return 0, 'near', 'Byte'

def num_regrid_params():
    """
    Returns parameters for numerical regridding.

    Returns:
        tuple: NoData value, resampling algorithm, and data type.
    """
    return -9999.0, 'bilinear', 'Float32'

def build_vrt(epsg_code=4326, input_list="my_list.txt", output_vrt="doq_index.vrt"):
    """
    Builds a VRT file using GDAL with specified parameters.

    Parameters:
        epsg_code (int): The EPSG code for spatial reference.
        input_list (str): Path to the input file list.
        output_vrt (str): Name of the output VRT file.

    Returns:
        None
    """
    cmd = [
        "gdalbuildvrt",
        "-allow_projection_difference",
        "-q",
        #"-tap",
        "-a_srs", f"EPSG:{str(epsg_code)}",
        "-input_file_list", input_list,
        output_vrt
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ VRT file '{output_vrt}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while building VRT: {e}")