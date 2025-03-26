import os 
import time 


def gwr_grid_downscaling(xpath, ypath, opath, oaux=False,epsg_code=4979, clean=True):
    # if epsg_code=4979 is None, use from the data 
    """
    Perform Geographically Weighted Regression (GWR) for grid downscaling.
    
    Parameters:
    - xpath (str): Path to the high-resolution DEM (predictor variable).
    - ypath (str): Path to the coarse-resolution data (dependent variable).
    - opath (str): Path to save the output SAGA grid (.sdat file).
    - oaux (bool, optional): If True, generate additional outputs like regression correction, quality, and residuals.

    Returns:
    - None: Saves the output files to the specified paths.

    Documentation:
    https://saga-gis.sourceforge.io/saga_tool_doc/8.2.2/statistics_regression_14.html
    """
    otif = opath.replace('.sdat', '.tif')

    # Construct the base SAGA command
    cmd = (
        f"saga_cmd statistics_regression 14 "
        f"-PREDICTORS {xpath} "
        f"-DEPENDENT {ypath} "
        f"-REGRESSION {opath} "
        f"-SEARCH_RANGE 0 "  # Local search range
        f"-DW_WEIGHTING 3 "  # Gaussian weighting
        f"-DW_BANDWIDTH 4 "  # Default bandwidth for Gaussian
        f"-MODEL_OUT 1"       # Output model parameters
    )

    if oaux:
        # Add optional outputs for residual correction, quality, and residuals
        opath_rescorr = opath.replace('.sdat', '_RESCORR.sdat')
        opath_quality = opath.replace('.sdat', '_QUALITY.sdat')
        opath_residuals = opath.replace('.sdat', '_RESIDUALS.sdat')
        cmd += (
            f" -REG_RESCORR {opath_rescorr} "
            f"-QUALITY {opath_quality} "
            f"-RESIDUALS {opath_residuals}"
        )

    # Run the SAGA command
    os.system(cmd)

    # Convert the output SAGA grid to GeoTIFF
    sdat_to_geotif(opath, otif,epsg_code)

    print("GWR Grid Downscaling completed.")
    if oaux:
        print(f"Additional outputs saved: \n{opath_rescorr}, \n{opath_quality}, \n{opath_residuals}")

    if clean:
        time.sleep(1)
        
        dirpath = os.path.dirname(opath)
        print(f'Cleaning up intermediate files...\n{dirpath}')
        for f in os.listdir(dirpath):
            if not f.endswith('.tif'):
                fo = os.path.join(dirpath, f)
                if os.path.isfile(fo):  # Check if it's a file
                    print(f'Removing {fo}...')
                    os.remove(fo)
                else:
                    print(f'Skipping directory: {fo}')
                

def sdat_to_geotif(sdat_path, gtif_path, epsg_code=4979):
    """
    Converts a Saga .sdat file to a GeoTIFF file using GDAL.

    Parameters:
        sdat_path (str): Path to the input .sdat file.
        gtif_path (str): Path to the output GeoTIFF file.
        epsg_code (int): EPSG code for the spatial reference system. Default is 4979.
    """
    # Ensure the input file has the correct extension
    if not sdat_path.endswith('.sdat'):
        sdat_path = sdat_path.replace('.sgrd', '.sdat')

    # Check if the output file already exists
    if os.path.isfile(gtif_path):
        print(f'! The file "{gtif_path}" already exists.')
        return

    # Construct and execute the GDAL command
    cmd = f'gdal_translate -a_srs EPSG:{epsg_code} -of GTiff "{sdat_path}" "{gtif_path}"'
    result = os.system(cmd)

    if result == 0:
        print(f'# Successfully converted "{sdat_path}" to "{gtif_path}".')
    else:
        print(f'! Failed to convert "{sdat_path}" to "{gtif_path}". Check the input files and GDAL installation.')

