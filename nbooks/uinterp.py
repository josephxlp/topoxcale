import rasterio
import numpy as np
from rasterio.fill import fillnodata
import math
import os 

def load_raster(fpath):
    """
    Loads a raster dataset and reads the first band as a masked array.

    Parameters:
    fpath (str): Path to the raster file.

    Returns:
    numpy.ndarray: A NumPy array with nodata values set to np.nan.
    """
    with rasterio.open(fpath) as src:
        data = src.read(1, masked=True)
        data = data.filled(np.nan)  # Ensure nodata values are np.nan
    return data


def write_raster(output_path, data, reference_path):
    """
    Writes a NumPy array to a raster file, preserving metadata from a reference raster.

    Parameters:
    output_path (str): Path to save the output raster.
    data (numpy.ndarray): Data to write to the raster.
    reference_path (str): Path to the reference raster for metadata.
    """
    with rasterio.open(reference_path) as src:
        meta = src.meta.copy()

    # Update metadata
    meta.update(dtype=np.float32, nodata=np.nan)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data, 1)

def fill_nodata(data, si=0):
    """
    Fills nodata values in a raster using interpolation.

    Parameters:
    data (numpy.ndarray): Input raster data with nodata values.
    smoothing_iterations (int): Number of smoothing iterations. Defaults to 0.

    Returns:
    numpy.ndarray: Raster data with nodata values filled.
    """
    mask = np.isfinite(data)  # Valid data mask
    max_search_distance = int(math.sqrt(data.shape[0] ** 2 + data.shape[1] ** 2)) + 1

    return fillnodata(data, mask=mask, max_search_distance=max_search_distance, smoothing_iterations=si)

def riofill(fipath, fopath, si=0):
    fopath = fopath.replace('.tif', f'_{str(si)}.tif')
    if os.path.isfile(fopath):
        print(f"already created {fopath}")
        return fopath
    rdata = load_raster(fipath)
    fdata = fill_nodata(rdata, si)
    write_raster(fopath, fdata, fipath)
    return fopath