import os

import pydicom
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd


def dcm_to_png(dcm_file, png_file):
    # 读取DICOM文件
    ds = pydicom.dcmread(dcm_file)

    # 获取图像数据
    img_array = ds.pixel_array

    # 将图像数据转换为8位（0-255）
    img_array = img_array - np.min(img_array)
    img_array = img_array / np.max(img_array)
    img_array = (img_array * 255).astype(np.uint8)

    # 创建Pillow图像
    img = Image.fromarray(img_array)

    # 保存为PNG文件
    img.save(png_file)
    print("DONE!")
    # print(f"Converted {dcm_file} to {png_file}")


if __name__ == '__main__':
    CBIS_path = r'/mnt/gemlab_data/wrs_mammogram_database/CBIS-DDSM/main'
    save_path = r'/home/data/hjc/cbis_test'
    save_image = os.path.join(save_path, 'images')
    save_mask = os.path.join(save_path, 'masks')
    csv_path = r'metadata.csv'
    csv_file = pd.read_csv(csv_path)
    if not os.path.exists(save_image):
        os.makedirs(save_image)
    if not os.path.exists(save_mask):
        os.makedirs(save_mask)
    # 遍历每一行数据
    for index, row in csv_file.iterrows():
        # print(f"Index: {index}")
        data_type = row["Subject ID"]
        number = row["Number of Images"]
        file_path = row['File Location'].lstrip(".\\").replace("\\", "/")
        if data_type.startswith("Mass-Test"):
            image_type = row["Series Description"]
            if image_type == "ROI mask images":
                image_list = os.listdir(os.path.join(CBIS_path, file_path))
                i = 0
                for file in image_list:
                    image_file = os.path.join(CBIS_path, file_path, file)
                    if file.endswith('(1).dcm'):
                        print(image_file)
                    else:
                        print(image_file)
                        file_save_path = os.path.join(save_mask, data_type + '_' + str(i) + '.png')
                        i=i+1
                        dcm_to_png(image_file, file_save_path)




            if image_type == "full mammogram images":
                image_path = os.path.join(os.path.join(CBIS_path, file_path), '1-1.dcm')
                file_save_path = os.path.join(save_image, data_type + '.png')
                dcm_to_png(image_path, file_save_path)
            # print(file_save_path)
            # elif image_type == "ROI mask images":
            #     if number == 1:
            #         image_path = os.path.join(os.path.join(CBIS_path, file_path), '1-1.dcm')
            #     else:
            #         image_path = os.path.join(os.path.join(CBIS_path, file_path), '1-2.dcm')
            #     file_save_path = os.path.join(save_mask, data_type.rstrip("_1") + '.png')
            #     dcm_to_png(image_path, file_save_path)

        # mask_path = row["ROI mask file path"]
        # image_file = os.path.join(CBIS_path, image_path)
        # mask_file = os.path.join(CBIS_path, mask_path)
        # image_name = image_path.split('/')[-1]
        # print(image_name)
