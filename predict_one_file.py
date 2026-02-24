# Predict segmentation from one (multi-modal) nifti image
# Supports both old .h5 (Keras 2) and new SavedModel (.tf_inference) formats
# Tested with Python 3.12, TensorFlow 2.20
# Backward compatible with TF >= 2.17 for SavedModel, TF >= 2.7 for .h5
# @author : Philippe Boutinaud - Fealinx
import gc
import os
import json
import time
import numpy as np
from pathlib import Path
import argparse
import nibabel
import tensorflow as tf


def _load_image(filename):
    dataNii = nibabel.load(filename)
    # load file and add dimension for the modality
    image = dataNii.get_fdata(dtype=np.float32)[..., np.newaxis]
    return image, dataNii.affine


def _is_saved_model(model_path):
    """Check if a model path is a SavedModel directory (vs .h5 file)."""
    model_path = Path(model_path)
    if model_path.is_dir():
        # SavedModel directory: contains saved_model.pb
        return (model_path / "saved_model.pb").exists()
    return False


def _is_h5_model(model_path):
    """Check if a model path is an HDF5 (.h5) Keras model file."""
    model_path = Path(model_path)
    return model_path.is_file() and model_path.suffix.lower() in (".h5", ".hdf5")


def _load_h5_model(model_path, verbose=False):
    """Load a legacy .h5 Keras 2 model.

    Requires the tf-keras package and TF_USE_LEGACY_KERAS=1 environment variable
    to be set BEFORE importing tensorflow (handled at script start).

    If no GPU is available, the model is rebuilt with float32 dtype policy
    to avoid mixed_float16 incompatibility on CPU.
    """
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"tf": tf}
    )

    # Check if model uses mixed_float16 and no GPU is available
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        # Check if any layer uses mixed_float16
        has_mixed = False
        for layer in model.layers:
            dtype_str = str(getattr(layer, "dtype_policy", ""))
            if "float16" in dtype_str or "mixed" in dtype_str:
                has_mixed = True
                break

        if has_mixed:
            if verbose:
                print("INFO : Model uses mixed_float16 but no GPU detected. "
                      "Rebuilding model in float32 for CPU inference.")
            config = model.get_config()
            config_str = json.dumps(config)
            config_str = config_str.replace('"mixed_float16"', '"float32"')
            config_modified = json.loads(config_str)

            with tf.keras.utils.custom_object_scope({"tf": tf}):
                model_f32 = model.__class__.from_config(config_modified)

            for new_layer, old_layer in zip(model_f32.layers, model.layers):
                weights = old_layer.get_weights()
                if weights:
                    weights_f32 = [w.astype(np.float32) for w in weights]
                    new_layer.set_weights(weights_f32)
            model = model_f32

    return model


def _load_saved_model(model_path, verbose=False):
    """Load a TF SavedModel (.tf_inference directory)."""
    model = tf.saved_model.load(model_path)
    return model


def _predict_h5(model, images, batch_size=1):
    """Run prediction with a Keras .h5 model."""
    return model.predict(images, batch_size=batch_size)


def _predict_saved_model(model, images, batch_size=1):
    """Run prediction with a SavedModel using the serve endpoint.

    SavedModel's serve() endpoint requires manual batching.
    """
    n_images = len(images)
    batched_preds = []
    for ibatch in np.arange(0, n_images, batch_size):
        begin, end = int(ibatch), int(min(ibatch + batch_size, n_images))
        batch = images[begin:end]
        batched_preds.append(model.serve(batch))
    return np.concatenate(batched_preds, axis=0)


# --- Script parameters ---
parser = argparse.ArgumentParser(
    description="Run inference with tensorflow models(s) on an image that "
                "may be built from several modalities. "
                "Supports both .h5 (Keras 2) and SavedModel (.tf_inference) formats."
)
parser.add_argument(
    "-i", "--input",
    type=Path,
    action='append',
    help="(multiple) input modality")

parser.add_argument(
    "-m", "--model",
    type=Path,
    action='append',
    help="(multiple) prediction models (.h5 files or SavedModel directories)")

parser.add_argument(
    "-b", "--braimask",
    type=Path,
    help="brain mask image")

parser.add_argument(
    "-o", "--output",
    type=Path,
    help="path for the output file (output of the inference from tensorflow model)")

parser.add_argument(
    "-g", "--gpu",
    type=int,
    default=0,
    help="GPU card ID, default 0; for CPU use -1")

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size for SavedModel inference (default: 1). "
         "Increase if GPU memory allows.")

parser.add_argument(
    "--verbose",
    help="increase output verbosity",
    action="store_true")

args = parser.parse_args()

_VERBOSE = args.verbose
_BATCH_SIZE = args.batch_size

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if _VERBOSE:
    if args.gpu >= 0:
        print(f"Trying to run inference on GPU {args.gpu}")
    else:
        print("Trying to run inference on CPU")

# The tf model files for the predictors, the prediction will be averaged
predictor_files = args.model
if len(predictor_files) == 0:
    raise ValueError("ERROR : No model given on command line")
modalities = args.input
if len(modalities) == 0:
    raise ValueError("ERROR : No image/modality given on command line")
brainmask = args.braimask
output_path = args.output

affine = None
image_shape = None
# Load brainmask if given (and get the affine & shape from it)
if brainmask is not None:
    brainmask, aff = _load_image(brainmask)
    image_shape = brainmask.shape
    if affine is None:
        affine = aff

# Load and/or build image from modalities
images = []
for modality in modalities:
    image, aff = _load_image(modality)
    if affine is None:
        affine = aff
    if image_shape is None:
        image_shape = image.shape
    else:
        if image.shape != image_shape:
            raise ValueError(
                f'Images have different shape {image_shape} vs {image.shape} in {modality}'  # noqa: E501
            )
    if brainmask is not None:
        image *= brainmask
    images.append(image)
# Concat all modalities
images = np.concatenate(images, axis=-1)
# Add a dimension for a batch of one image
images = np.reshape(images, (1,) + images.shape)

# Detect model format from first model
first_model = predictor_files[0]
use_saved_model = _is_saved_model(first_model)
use_h5 = _is_h5_model(first_model)

if not use_saved_model and not use_h5:
    raise ValueError(
        f"ERROR : Cannot determine model format for: {first_model}\n"
        "Expected either a .h5 file or a SavedModel directory (containing saved_model.pb)"
    )

if _VERBOSE:
    fmt = "SavedModel" if use_saved_model else ".h5 (Keras)"
    print(f"INFO : Detected model format: {fmt}")
    print(f"INFO : Number of folds: {len(predictor_files)}")

# For .h5 models with legacy Keras, check that tf-keras is available
if use_h5:
    if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        if _VERBOSE:
            print("WARNING : Setting TF_USE_LEGACY_KERAS=1 for .h5 model loading. "
                  "For best results, set this environment variable before running the script.")
    try:
        import tf_keras  # noqa: F401 — just check it's importable
    except ImportError:
        print("WARNING : tf-keras package not found. "
              "Old .h5 models may fail to load. Install with: pip install tf-keras")

chrono0 = time.time()
# Load models & predict
predictions = []
for predictor_file in predictor_files:
    if use_saved_model:
        try:
            import keras
            keras.utils.clear_session(free_memory=True)
        except (ImportError, AttributeError):
            tf.keras.backend.clear_session()
    else:
        tf.keras.backend.clear_session()
    gc.collect()

    try:
        if use_saved_model:
            model = _load_saved_model(predictor_file, verbose=_VERBOSE)
        else:
            model = _load_h5_model(predictor_file, verbose=_VERBOSE)
    except Exception as err:
        print(f'\n\tWARNING : Exception loading model : {predictor_file}\n{err}')
        continue

    if _VERBOSE:
        name = Path(predictor_file).stem if use_h5 else Path(predictor_file).name
        print(f'INFO : Predicting fold : {name}')

    if use_saved_model:
        prediction = _predict_saved_model(model, images, batch_size=_BATCH_SIZE)
    else:
        prediction = _predict_h5(model, images, batch_size=_BATCH_SIZE)

    if brainmask is not None:
        prediction *= brainmask
    predictions.append(prediction)

if len(predictions) == 0:
    raise RuntimeError("ERROR : No successful predictions from any model fold")

# Average all predictions
predictions = np.mean(predictions, axis=0)

chrono1 = (time.time() - chrono0) / 60.
if _VERBOSE:
    print(f'Inference time : {chrono1:.2f} min.')

# Save prediction
nifti = nibabel.Nifti1Image(predictions[0], affine=affine)
nibabel.save(nifti, output_path)

if _VERBOSE:
    print(f'\nINFO : Done with predictions -> {output_path}\n')
