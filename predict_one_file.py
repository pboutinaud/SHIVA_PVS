# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx
import gc
import os
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


# Script parameters
parser = argparse.ArgumentParser(
    description="Run inference with tensorflow models(s) on an image that may be built from several modalities"
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
    help="(multiple) prediction models")

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
    "--verbose",
    help="increase output verbosity",
    action="store_true")

args = parser.parse_args()

_VERBOSE = args.verbose

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

chrono0 = time.time()
# Load models & predict
predictions = []
for predictor_file in predictor_files:
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        model = tf.keras.models.load_model(
            predictor_file,
            compile=False,
            custom_objects={"tf": tf})
    except Exception as err:
        print(f'\n\tWARNING : Exception loading model : {predictor_file}\n{err}')
        continue
    print('INFO : Predicting fold :', predictor_file.stem)
    prediction = model.predict(
        images,
        batch_size=1
        )
    if brainmask is not None:
        prediction *= brainmask
    predictions.append(prediction)

# Average all predictions
predictions = np.mean(predictions, axis=0)

chrono1 = (time.time() - chrono0) / 60.
if _VERBOSE:
    print(f'Inference time : {chrono1} sec.')

# Save prediction
nifti = nibabel.Nifti1Image(predictions[0], affine=affine)
nibabel.save(nifti, output_path)

if _VERBOSE:
    print(f'\nINFO : Done with predictions -> {output_path}\n')

# %%
