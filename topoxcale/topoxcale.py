"""Main module."""
from os.path import exists,splitext,join
import tempfile

#from mlscale import * # split the functions like in pdal readers, writters and so on 
from topoxcale.mlxcale import (load_model,check_and_resample,read_raster,
                     remove_nulls,sample_data,train_model,predict_and_save,
                     resample_y_to_match_predictions,
                     save_residuals_and_correct_predictions)

model_params = {
    'num_rounds': 1000,  # This will be mapped to 'iterations'
    #'learning_rate': 0.1,
    #'depth': 6,
    'verbose': 200
}

def mldownxcale(x_path, y_path, model_name, model_params, out_path, n=0):
    """
    Main function to handle training, prediction, saving, and residual correction.
    """
    if exists(model_name):
        print(f"Model {model_name} already exists. Skipping training.")
        model = load_model(model_name)
    else:
        # Check resolution and CRS, resample if necessary
        x_data, x_profile, y_path = check_and_resample(x_path, y_path)

        # Read data
        y_data, _ = read_raster(y_path)

        x_data = x_data.reshape(-1, 1)
        y_data = y_data.ravel()

        # Remove nulls
        x_data, y_data = remove_nulls(x_data, y_data)

        # Sample data
        x_sampled, y_sampled = sample_data(x_data, y_data, n)

        # Train model
        train_model(x_sampled, y_sampled, model_name,model_params)
        model = load_model(model_name)

    # Predict always, regardless of whether out_path exists
    predictions, pred_profile = predict_and_save(model, x_path, out_path)

    # Resample y_path to match the predictions raster
    temp_y_resampled = join(tempfile.gettempdir(), "temp_y_resampled.tif")
    resampled_y_path = resample_y_to_match_predictions(y_path, out_path, temp_y_resampled)

    # Save residuals and corrected predictions
    residuals_out_path = splitext(out_path)[0] + "_residuals.tif"
    corrected_out_path = splitext(out_path)[0] + "_corrected.tif"
    save_residuals_and_correct_predictions(predictions, resampled_y_path, residuals_out_path, corrected_out_path)

