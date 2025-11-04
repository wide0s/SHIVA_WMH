import sys, numpy as np, nibabel as nib
from scipy.ndimage import zoom

def resample_to_isotropic(img: nib.Nifti1Image, target=1.0):
    data = img.get_fdata()
    affine = img.affine.copy()
    zooms = img.header.get_zooms()[:3]
    factors = [zooms[i] / target for i in range(3)]
    data_iso = zoom(data, zoom=factors, order=1)
    new_affine = affine.copy()
    new_affine[:3, :3] = np.diag([target]*3)
    return nib.Nifti1Image(data_iso, new_affine)

def normalize(img: nib.Nifti1Image):
    data = img.get_fdata().astype(np.float32)
    p99 = np.percentile(data[data>0], 99) # data[data>0] ??
    data = np.clip(data/p99, 0, 1) # why ?
    return nib.Nifti1Image(data, img.affine)

flair = nib.load(sys.argv[1])
t1    = nib.load(sys.argv[2])
flair_prep = normalize(resample_to_isotropic(flair))
t1_prep    = normalize(resample_to_isotropic(t1))
nib.save(flair_prep, sys.argv[3])
nib.save(t1_prep, sys.argv[4])

# python prep_normalize.py Amsterdam_GE1T5_150_Orig/FLAIR.nii.gz Amsterdam_GE1T5_150_Orig/T1.nii.gz FLAIR_1mm.nii.gz T1_1mm.nii.gz