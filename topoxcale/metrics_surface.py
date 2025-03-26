import numpy as np
import rasterio
from time import time

def read_raster(raster_path):
    """Reads a raster file and returns the data array and profile."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        profile = src.profile
    return data, profile

def mask_nodata(data, nodata_value):
    """Masks out nodata values in the raster data."""
    return np.where(data == nodata_value, np.nan, data)

def calculate_r_squared(observed, predicted):
    """Calculates the Coefficient of Determination (R²)."""
    ss_total = np.sum((observed - np.mean(observed)) ** 2)
    ss_residual = np.sum((observed - predicted) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.00000000000000000000001#np.nan

def calculate_rmse(observed, predicted):
    """Calculates Root Mean Square Error (RMSE)."""
    return np.sqrt(np.nanmean((observed - predicted) ** 2))

def calculate_residuals(observed, predicted):
    """Calculates residuals (observed - predicted)."""
    return observed - predicted

def write_raster(output_path, data, profile):
    """Writes processed data to a new raster file."""
    profile.update(dtype=rasterio.float32)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)

def write_surface_metrics(pred_raster_path, obs_raster_path, output_r_squared_path, output_rmse_path, output_residuals_path):
    """Processes prediction and observation rasters to compute metrics."""
    # Read input rasters
    pred, pred_profile = read_raster(pred_raster_path)
    obs, obs_profile = read_raster(obs_raster_path)

    # Mask nodata values
    nodata = pred_profile.get('nodata', np.nan)
    pred = mask_nodata(pred, nodata)
    obs = mask_nodata(obs, nodata)

    # Initialize output arrays
    r_squared_grid = np.empty_like(pred)
    rmse_grid = np.empty_like(pred)
    residuals_grid = np.empty_like(pred)

    # Compute metrics
    start_time = time()
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            observed_value = obs[i, j]
            predicted_value = pred[i, j]
            if np.isnan(observed_value) or np.isnan(predicted_value):
                r_squared_grid[i, j] = np.nan
                rmse_grid[i, j] = np.nan
                residuals_grid[i, j] = np.nan
            else:
                r_squared_grid[i, j] = calculate_r_squared(np.array([observed_value]), np.array([predicted_value]))
                rmse_grid[i, j] = calculate_rmse(np.array([observed_value]), np.array([predicted_value]))
                residuals_grid[i, j] = calculate_residuals(observed_value, predicted_value)
    print(f"Metrics computed in {time() - start_time:.2f} seconds.")

    # Write outputs
    start_time = time()
    write_raster(output_r_squared_path, r_squared_grid, pred_profile)
    print(f"R² raster written in {time() - start_time:.2f} seconds.")

    start_time = time()
    write_raster(output_rmse_path, rmse_grid, pred_profile)
    print(f"RMSE raster written in {time() - start_time:.2f} seconds.")

    start_time = time()
    write_raster(output_residuals_path, residuals_grid, pred_profile)
    print(f"Residuals raster written in {time() - start_time:.2f} seconds.")

    print(f"All outputs saved: \nR²='{output_r_squared_path}', \nRMSE='{output_rmse_path}', \nResiduals='{output_residuals_path}'.")

# Example usage
# process_rasters("predictions.tif", "observations.tif", "r_squared.tif", "rmse.tif", "residuals.tif")