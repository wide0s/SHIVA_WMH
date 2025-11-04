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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Tensoflow's INFO messages are not printed
import tensorflow as tf
from skimage.transform import resize
from scipy.ndimage import zoom


target_dimensions = (160, 214, 176)


def _resample_to_isotropic(img: nibabel.Nifti1Image, target=1.0):
    data = img.get_fdata()
    affine = img.affine.copy()
    zooms = img.header.get_zooms()[:3]
    factors = [zooms[i] / target for i in range(3)]
    data_iso = zoom(data, zoom=factors, order=1)
    new_affine = affine.copy()
    new_affine[:3, :3] = np.diag([target]*3)
    return nibabel.Nifti1Image(data_iso, new_affine)


def _normalize_image(img: nibabel.Nifti1Image):
    data = img.get_fdata().astype(np.float32)
    p99 = np.percentile(data[data>0], 99)
    data = np.clip(data/p99, 0, 1) # why ?
    return nibabel.Nifti1Image(data, img.affine)


def _load_image(filename, normalize: bool, adjust_dimensions: bool):
    print(f'INFO : Loading image {filename}')
    dataNii = nibabel.load(filename)
    if normalize:
        print(f'INFO : Normalizing image...')
        dataNii = _normalize_image(_resample_to_isotropic(dataNii))
    # load file and add dimension for the modality
    image_data = dataNii.get_fdata(dtype=np.float32)[..., np.newaxis]
    # adjust image dimensions
    image_dimensions = (image_data.shape[0], image_data.shape[1],
                        image_data.shape[2])
    if adjust_dimensions and image_dimensions != target_dimensions:
        image_data = resize(image_data, target_dimensions,
                            order=1,  # 1 for linear interpolation
                            preserve_range=True)
        print(f'INFO : Adjusted image dimensions from {image_dimensions} to {image_data.shape}')
    return image_data, dataNii.affine


def _intermediate_image_path(filename: str, output_dir: str = None,
                             normalize:bool = False,
                             adjust_dimensions: bool = False):
    if output_dir is None:
        output_dir = os.path.dirname(filename)
    filename = os.path.basename(filename)
    if adjust_dimensions:
        filename = 'resized_' + filename
    if normalize:
        filename = 'norm_' + filename
    filename = 'i_' + filename
    return os.path.join(output_dir, filename)


def _save_image(image_data, affine, output_path):
    nifti = nibabel.Nifti1Image(image_data, affine=affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nibabel.save(nifti, output_path)

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

parser.add_argument(
    "-a", "--adjust-dimensions",
    help="adjust the image dimensions to match those the model was trained on",
    action="store_true")

parser.add_argument(
    "-n", "--normalize",
    help="normalize voxel sizes to 1 mm x 1 mm x 1 mm",
    action="store_true")

parser.add_argument(
    "-t", "--temp-dir",
    type=Path,
    help="path to the temporary directory")

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
    brainmask, aff = _load_image(brainmask,
                                 args.normalize,
                                 args.adjust_dimensions)
    image_shape = brainmask.shape
    if affine is None:
        affine = aff
    if args.temp_dir:
        intermediate_path = _intermediate_image_path(args.braimask,
                                                     args.temp_dir,
                                                     args.normalize,
                                                     args.adjust_dimensions)
        _save_image(brainmask, affine, intermediate_path)

# Load and/or build image from modalities
images = []
for modality in modalities:
    image, aff = _load_image(modality,
                             args.normalize,
                             args.adjust_dimensions)
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
    if args.temp_dir:
        intermediate_path = _intermediate_image_path(modality,
                                                     args.temp_dir,
                                                     args.normalize,
                                                     args.adjust_dimensions)
        _save_image(image, affine, intermediate_path)

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
_save_image(predictions[0], affine, output_path)

if _VERBOSE:
    print(f'\nINFO : Done with predictions -> {output_path}\n')

# %%
