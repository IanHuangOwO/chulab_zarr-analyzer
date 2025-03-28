import numpy as np
import tifffile

# 創建一個 456x320x528 的 3D 陣列，初始值為 0
data = np.zeros((660, 400,570 ), dtype=np.uint8)

# 中間分割點 x=228
mid_point = 285

# 將左邊 (x < 228) 設為灰階值 1
data[:, :, :mid_point] = 1

# 將右邊 (x >= 228) 設為灰階值 2
data[:, :, mid_point:] = 2

# 儲存為 TIFF 檔案
tifffile.imwrite(r'F:\Lab\others\YA_HAN\bird_hema.tif', data)

print(f"檔案尺寸: {data.shape}")

##transformix.exe -in D:\blood_vessel2\2024_2\bird_hema.tif -out D:\blood_vessel2\2024_2 -tp Z:\Ya_Hui\Hu_TBI_2024_WC\20240711_11_51_51_Hu_TBI_WC_60d_2-2_Lectin-561_Ibal-642_4X_z4_tiffs_destriped\birds\registration\coarse\out\TransformParameters.2.txt