import numpy as np
import cv2

from scipy import fftpack as fftp
from matplotlib import pyplot as plt

def detect(image):
    firstq = 30
    secondq = 40
    thres = 0.5

    dct_rows = 0
    dct_cols = 0

   # `dct_rows` and `dct_cols`: Variables to store the dimensions of the DCT image.
   # are initialized to zero and will be used later to ensure the image dimensions are divisible by 8 
   # for DCT compression.
   # `dct_image`: A numpy array to store the DCT image.

    image = cv2.imread(image)
    shape = image.shape
    
    #Load the image using OpenCV's "imread" function and store its shape in the "shape" variable.
    #Pad the image so that its dimensions are divisible by 8 (which is a requirement for DCT):
    if shape[0] % 8 != 0:
        dct_rows = shape[0]+8-shape[0] % 8
    else:
        dct_rows = shape[0]

    if shape[1] % 8 != 0:
        dct_cols = shape[1]+8-shape[1] % 8
    else:
        dct_cols = shape[1]
    
    """
    `shape[0]` represents the height of the image.
    The condition `shape[0] % 8 != 0` checks if the height is not divisible by 8 
    - If the condition is true, it means the height needs to be padded.The equation adds the difference between 8 
    and the remainder of `shape[0]` divided by 8 to the original height. 
    The result is a new height that is divisible by 8.
    - If the condition is false, it means the height is already divisible by 8, so no padding is needed. 
    In that case, `dct_rows` is simply assigned the original height.
    `shape[1]` represents the width of the image.
    """
    dct_image = np.zeros((dct_rows, dct_cols, 3), np.uint8)
    
    #np.zeros()` creates an array of zeros with the specified dimensions `(dct_rows, dct_cols, 3)`.
    #  The third dimension of size 3 corresponds to the RGB channels of the image.

    dct_image[0:shape[0], 0:shape[1]] = image

    #dct_image` is a new image with padded dimensions.
    #cv2.cvtColor()` converts the image from BGR to YCrCb color space.
    #The Y component is the luminance (brightness) of the image, CR:Red Chrominance ,CB:Blue Chrominance
    # This conversion separates the image into Y (luminance) and Cr/Cb (chrominance) channels.

    y = cv2.cvtColor(dct_image, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    w = y.shape[1]
    h = y.shape[0]
    n = w*h/64

    #Divide the image into non-overlapping 8x8 blocks:
    
    Y = y.reshape(h//8, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8)

    """
    `h` and `w` are the height and width of the image.
    - The `reshape()` function creates an array of non-overlapping 8x8 blocks.
    - `swapaxes()` swaps the axes so that each block is in the first dimension.
    - `reshape()` is called again to flatten the array into a 2D array of 8x8 blocks.
    """
    # Compute the DCT of each 8x8 block:
    
    qDCT = []
    
    """
    The resulting DCT coefficients are converted to a numpy array and the mean of all blocks is subtracted from 
    each block.
    - The coefficients are rounded to integers using `np.rint`.
       - The DCT coefficients are stored in the `qDCT` array.
       Quantization: After DCT, the resulting frequency coefficients are quantized.
         Quantization involves dividing the coefficients by a set of values, called quantization values. 
         This process reduces the number of unique values in the coefficients, thus reducing the amount of 
         information that needs to be stored.
    """
    for i in range(0, Y.shape[0]):
        qDCT.append(cv2.dct(np.float32(Y[i])))
    
    #The mean value of each block is subtracted to normalize the acquired DCT coefficients.

    qDCT = np.asarray(qDCT, dtype=np.float32)
    qDCT = np.rint(qDCT - np.mean(qDCT, axis=0)).astype(np.int32)
    f, a1 = plt.subplots(8, 8)
    a1 = a1.ravel()
    k = 0

    # flag = True
    """
        Peak Counting and Detection:
       - A subplot of 8x8 grid is created using `plt.subplots()`, and the axes are flattened into `a1`.
       - A loop iterates over each axis, representing each 8x8 block.
       - For each block, the quantized DCT coefficients are extracted and a histogram of the coefficients is calculated
         using `np.histogram()`.
       - The histogram data is then transformed using FFT (`fftp.fft()`), and the result is reshaped and rolled
         using numpy operations.
       - The slope of the transformed data is calculated, and the indices of peaks are identified.
       - For each index, if the corresponding FFT value is above the threshold

        Compression detection:
       - The code checks for a specific subplot index (`k==3`), corresponding to a specific frequency component.
       - If the number of peaks above the threshold (`peak_count`) is greater than or equal to 20, it returns `True` 
       indicating that the image has been compressed using DCT. Otherwise, Otherwise, return False.
    """
    for idx, ax in enumerate(a1):
        k += 1
        data = qDCT[:, int(idx/8), int(idx % 8)]
        val, key = np.histogram(data, bins=np.arange(data.min(), data.max()+1))
        z = np.absolute(fftp.fft(val))
        z = np.reshape(z, (len(z), 1))
        rotz = np.roll(z, int(len(z)/2))

        slope = rotz[1:] - rotz[:-1]
        indices = [i+1 for i in range(len(slope)-1)
                   if slope[i] > 0 and slope[i+1] < 0]

        peak_count = 0

        for j in indices:
            if rotz[j][0] > thres:
                peak_count += 1

        if(k==3):
            if peak_count>=20: return True
            else: return False
            # flag = False 