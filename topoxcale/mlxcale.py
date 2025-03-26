import os
import joblib
import tempfile
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from catboost import CatBoostRegressor


def read_raster(file_path, nodata_to_nan=True):
    """
    Reads a raster file using rasterio and optionally replaces nodata values with np.nan.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        if nodata_to_nan and src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        profile = src.profile  # Capture the profile before closing the dataset
    return data, profile

def remove_nulls(x, y): 
    """
    Removes rows with NaN values from x and y.
    """
    valid_mask = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
    return x[valid_mask], y[valid_mask]


def sample_data(x, y, n):
    """
    Samples the dataset by N. If N=0, returns all data.
    """
    if n == 0 or n >= len(x):
        return x, y
    indices = np.random.choice(len(x), n, replace=False)
    return x[indices], y[indices]


def check_and_resample(x_path, y_path):
    """
    Checks if two rasters have the same resolution and EPSG code. If not, resamples the finer resolution raster
    to match the coarser one and saves the result to a temporary file. Returns the resampled data and profile.
    """
    temp_file = os.path.join(tempfile.gettempdir(), "temp_resampled.tif")

    with rasterio.open(x_path) as src_x, rasterio.open(y_path) as src_y:
        # Check resolutions
        res_x = src_x.res
        res_y = src_y.res
        epsg_x = src_x.crs.to_epsg()
        epsg_y = src_y.crs.to_epsg()

        if res_x != res_y or epsg_x != epsg_y:
            print("Resolutions or CRS do not match. Resampling...")
            # Determine which is finer/coarser
            if res_x < res_y:  # x is finer
                src_fine, src_coarse = src_x, src_y
                fine_path, coarse_path = x_path, y_path
            else:  # y is finer
                src_fine, src_coarse = src_y, src_x
                fine_path, coarse_path = y_path, x_path

            # Resample finer raster to match coarser raster
            with rasterio.open(coarse_path) as dst_coarse:
                profile = dst_coarse.profile
                profile.update(dtype=rasterio.float32, nodata=np.nan)

                with rasterio.open(temp_file, 'w', **profile) as dst_temp:
                    reproject(
                        source=rasterio.band(src_fine, 1),
                        destination=rasterio.band(dst_temp, 1),
                        src_transform=src_fine.transform,
                        src_crs=src_fine.crs,
                        dst_transform=dst_coarse.transform,
                        dst_crs=dst_coarse.crs,
                        resampling=Resampling.bilinear
                    )
                print(f"Resampled {fine_path} to match {coarse_path}. Saved to {temp_file}")

            # Read the resampled data and profile
            resampled_data, resampled_profile = read_raster(temp_file)

            # Return paths/data after resampling
            if fine_path == x_path:
                return resampled_data, resampled_profile, y_path
            else:
                return x_path, resampled_data, resampled_profile

    # If no resampling is needed
    print("Resolutions and CRS match. No resampling needed.")
    return read_raster(x_path)[0], read_raster(x_path)[1], y_path


from catboost import CatBoostRegressor
import joblib

def train_model(x, y, model_name, model_params):
    """
    Trains a CatBoostRegressor model and saves it to disk.
    """
    # Map 'num_rounds' to 'iterations' if present in model_params
    if 'num_rounds' in model_params:
        model_params['iterations'] = model_params.pop('num_rounds')  # Replace num_rounds with iterations
    
    # Initialize the CatBoostRegressor with the provided parameters
    model = CatBoostRegressor(**model_params)
    
    # Train the model
    model.fit(x, y)
    
    # Save the trained model to disk
    joblib.dump(model, model_name)
    
    print(f"Model trained and saved as {model_name}")
            

def load_model(model_name):
    """
    Loads a pre-trained CatBoostRegressor model from disk.
    """
    if os.path.exists(model_name):
        return joblib.load(model_name)
    raise FileNotFoundError(f"Model file {model_name} does not exist.")


def predict_and_save(model, x_path, out_path):
    """
    Predicts y values using the model and writes them to the output raster file.
    """
    # Load x data
    x_data, profile = read_raster(x_path)
    original_shape = x_data.shape
    x_data_flat = x_data.reshape(-1, 1)

    # Predict
    predictions = model.predict(x_data_flat)

    # Reshape predictions back to original shape
    predictions_reshaped = predictions.reshape(original_shape)

    # Write to output raster
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(predictions_reshaped.astype(rasterio.float32), 1)

    print(f"Predictions saved to {out_path}")
    return predictions_reshaped, profile


def resample_y_to_match_predictions(y_path, out_path, temp_file):
    """
    Resamples y_path to match the resolution, extent, and CRS of the predictions raster (out_path).
    Saves the resampled raster to a temporary file.
    """
    with rasterio.open(y_path) as src_y, rasterio.open(out_path) as src_pred:
        profile = src_pred.profile
        profile.update(dtype=rasterio.float32, nodata=np.nan)

        with rasterio.open(temp_file, 'w', **profile) as dst_temp:
            reproject(
                source=rasterio.band(src_y, 1),
                destination=rasterio.band(dst_temp, 1),
                src_transform=src_y.transform,
                src_crs=src_y.crs,
                dst_transform=src_pred.transform,
                dst_crs=src_pred.crs,
                resampling=Resampling.bilinear
            )
        print(f"Resampled {y_path} to match {out_path}. Saved to {temp_file}")

    return temp_file


def save_residuals_and_correct_predictions(predictions, y_path, residuals_out_path, corrected_out_path):
    """
    Computes residuals (ground truth - predictions), adds them to predictions to correct the fit,
    and saves both residuals and corrected predictions to raster files.
    """
    # Load y data
    y_data, profile = read_raster(y_path)

    # Compute residuals (ground truth - predictions)
    residuals = y_data - predictions

    # Correct predictions by adding residuals
    corrected_predictions = predictions + residuals

    # Write residuals to output raster
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    os.makedirs(os.path.dirname(residuals_out_path), exist_ok=True)
    with rasterio.open(residuals_out_path, 'w', **profile) as dst:
        dst.write(residuals.astype(rasterio.float32), 1)

    print(f"Residuals saved to {residuals_out_path}")

    # Write corrected predictions to output raster
    os.makedirs(os.path.dirname(corrected_out_path), exist_ok=True)
    with rasterio.open(corrected_out_path, 'w', **profile) as dst:
        dst.write(corrected_predictions.astype(rasterio.float32), 1)

    print(f"Corrected predictions saved to {corrected_out_path}")






