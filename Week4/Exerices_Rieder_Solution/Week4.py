import numpy as np
from scipy import ndimage
from PIL import Image
import pydicom
import sys
from pydicom.data import get_testdata_files
import matplotlib.pyplot as plt
import matplotlib


# Exercise 1: Explore Koala!

# 1.(a) Read the koala image
im = Image.open('koala.tiff')
imarray = np.array(im)
print(imarray)

# b. Find the 75 percentile (   ) of all signal intensities (SI);
# c. Find the pixels whose SI is above   and halve their SI;
result = np.percentile(imarray, 75, axis=2)
print(result)
plt.imshow(result, aspect='auto', interpolation='none', origin='lower')

# d. Save the image using the file name
plt.savefig('koala_processed.tiff')

# e. Get the difference image
result_process = np.reshape(result, (768, 1024, -1))
process_koala_diff = result_process - imarray
plt.imshow(process_koala_diff, cmap=plt.cm.bone)
plt.show()
plt.savefig('koala_dff.tiff')
koala_diff = Image.open("./koala_dff.tiff")
rotated = koala_diff.rotate(270)

# f. Rotate Image and rescale
rescale = rotated.resize((640, 960//2), Image.NEAREST)
rescale.save("koala_diff_rot.tiff")


# Excise 2: Explore the MRI image MR000008

filename = "./SE000001/MR000008"
dcm = pydicom.dcmread(filename)
dcm.PatientName  # => LIONHEART^WILLIAM
dcm.PatientID  # => RJN7270540X
data_bytes = dcm.PixelData
data = dcm.pixel_array
print(data)
plt.imshow(data, cmap=plt.cm.bone)
# plt.show()
plt.savefig('MR000008.tiff')
MR000008_processed = np.percentile(data, 75, axis=1, keepdims=True)
MR000008_processed.reshape(256, -1)
MR000008_diff = MR000008_processed - data
plt.imshow(MR000008_diff, cmap=plt.cm.bone)
plt.show()
MR000008_diff = Image.open('./MR000008_diff.tiff')
MR000008_diff_rot = MR000008_diff.rotate(270)
MR000008_diff_rot = MR000008_diff_rot.resize((640, 960//2), Image.NEAREST)
MR000008_diff_rot.save("MR000008_diff_rot.tiff")

# Inspect three more slices picture
MR000009_name = "./SE000001/MR000009"
MR0000010_name = "./SE000001/MR000010"
MR0000011_name = "./SE000001/MR000011"

dcm9 = pydicom.dcmread(MR000009_name)
dcm10 = pydicom.dcmread(MR0000010_name)
dcm11 = pydicom.dcmread(MR0000011_name)

Pic9_data = dcm9.pixel_array
Pic10_data = dcm10.pixel_array
Pic11_data = dcm11.pixel_array

plt.imshow(Pic9_data, cmap=plt.cm.bone)
plt.show()
plt.savefig('MR000009.tiff')

plt.imshow(Pic10_data, cmap=plt.cm.bone)
plt.show()
plt.savefig('MR0000010.tiff')

plt.imshow(Pic11_data, cmap=plt.cm.bone)
plt.show()
plt.savefig('MR0000011.tiff')
