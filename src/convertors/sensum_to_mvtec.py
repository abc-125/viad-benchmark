import os
import shutil
import random
import argparse


"""Convertor for SensumSODF dataset to MVTecAD format.
Uses only one split into train and test with 150 good 
images left for testing.

Args:
    sensum_path (str): Path to the SensumSODF dataset.
    converted_path (str): Path to the newly converted dataset.
"""


# parser
parser = argparse.ArgumentParser()
parser.add_argument("--sensum_path", type=str, help="Path to the SensumSODF dataset")
parser.add_argument("--converted_path", type=str, help="Path to the newly converted dataset")


if __name__=="__main__":
    args = parser.parse_args()
    sensum_path = args.sensum_path
    converted_path = args.converted_path
    print("Converting SensumSODF into MVTecAD format using a single split.")

    random.seed(143)

    for cls in ["capsule", "softgel"]:
        print("Processing {}".format(cls))
        cls_path = os.path.join(converted_path, cls)
        train_path = os.path.join(cls_path, "train/good")
        test_good_path = os.path.join(cls_path, "test/good")
        test_bad_path = os.path.join(cls_path, "test/bad")
        gt_path = os.path.join(cls_path, "ground_truth/bad")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_bad_path, exist_ok=True)
        os.makedirs(test_good_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

        path_good = os.path.join(sensum_path, cls, "negative", "data")
        path_bad = os.path.join(sensum_path, cls, "positive", "data")
        path_bad_gt = os.path.join(sensum_path, cls, "positive", "annotation")
        files_test_bad = os.listdir(path_bad)
        files_good = os.listdir(path_good)
        random.shuffle(files_good)
        files_train_good = files_good[150:]
        files_test_good = files_good[:150]  # use 150 good images for test

        for file in files_test_bad:
            image_full_path = os.path.join(path_bad, file)
            image_new_full_path = os.path.join(test_bad_path, file)
            shutil.copy(image_full_path, image_new_full_path)
            gt_full_path = os.path.join(path_bad_gt, file)
            gt_new_full_path = os.path.join(gt_path, file)
            shutil.copy(gt_full_path, gt_new_full_path)

        for file in files_test_good:
            image_full_path = os.path.join(path_good, file)
            image_new_full_path = os.path.join(test_good_path, file)
            shutil.copy(image_full_path, image_new_full_path)

        for file in files_train_good:
            image_full_path = os.path.join(path_good, file)
            image_new_full_path = os.path.join(train_path, file)
            shutil.copy(image_full_path, image_new_full_path)


