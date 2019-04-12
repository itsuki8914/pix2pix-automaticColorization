import cv2
import os
import numpy as np
import sys
RAW_SIZE = 512
CLOP_SIZE = 128


def crop(input_folder, output_folder):
    FOLDER = input_folder

    files=os.listdir(FOLDER)

    out_hr = output_folder
    if not os.path.exists(out_hr):
        os.mkdir(out_hr)


    for i, path in enumerate(files):

        if i % 100 ==0:
            print("{} pics were processed".format(i))
        img = cv2.imread("{}/{}".format(FOLDER,path))
        #print(img.shape)
        height ,width= img.shape[:2]

        iters = np.minimum((height//RAW_SIZE+1)* (width//RAW_SIZE+1), 4)


        if height < RAW_SIZE or width < RAW_SIZE:
            continue
        try:
            for j in range(int(iters)):
                y_start = np.random.randint(0,height-RAW_SIZE)
                x_start = np.random.randint(0,width-RAW_SIZE)

                hr_img = img[y_start:y_start+RAW_SIZE,x_start:x_start+RAW_SIZE]
                #lr_img = cv2.resize(hr_img,(CLOP_SIZE,CLOP_SIZE))

                image_name = os.path.splitext(os.path.basename(path))[0]


                cv2.imwrite("{}/{}_{}_hr.png".format(out_hr,image_name,j),hr_img)


                #cv2.imwrite("{}/{}_lr.png".format(out_lr,image_name),lr_img)

                hr_size = os.path.getsize("{}/{}_{}_hr.png".format(out_hr,image_name,j))
                #print("ok")
                if hr_size < 300000:
                    #print("{}/{}_hr.png".format(out_hr,image_name))
                    os.remove("{}/{}_{}_hr.png".format(out_hr,image_name,j))
                    #os.remove("{}/{}_lr.png".format(out_lr,image_name))
                    print("{} was removed because its hrSize has {} byte less than 300,000bytes.".format(path,hr_size))
        except:
            print("errors occured in {}, which has shape({},{})".format(path,height,width))

        if i >30000:
            break


if __name__ == '__main__':
    input_folder = "4val"
    output_folder = "val"

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]


    print("input_folder: {}, output_folder: {}".format(input_folder, output_folder))
    crop(input_folder, output_folder)
