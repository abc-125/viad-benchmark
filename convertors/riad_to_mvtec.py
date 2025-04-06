import os
import json
import shutil
import argparse


"""Convertor for Real-IAD dataset to MVTecAD format.
Uses only view from above (single view) 
and merges all defects classes into a single class.

Args:
    riad_path (str): Path to the Real-IAD dataset.
    converted_path (str): Path to the newly converted dataset.
"""

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--riad_path", type=str, help="Path to the Real-IAD dataset")
parser.add_argument("--converted_path", type=str, help="Path to the newly converted dataset")


if __name__=="__main__":
    args = parser.parse_args()
    riad_path = args.riad_path
    converted_path = args.converted_path
    print("Converting Real-IAD into MVTecAD format using only view from above (single view).")
    json_path = os.path.join(riad_path, "realiad_jsons_sv")  # sv - single view
    json_names = os.listdir(json_path)

    for json_name in json_names:
        # prepare folders
        cls_name = json_name[:-5]
        print("Processing {}".format(cls_name))
        cls_path = os.path.join(converted_path, cls_name)
        train_path = os.path.join(cls_path, "train/good")
        test_good_path = os.path.join(cls_path, "test/good")
        test_bad_path = os.path.join(cls_path, "test/bad")
        gt_path = os.path.join(cls_path, "ground_truth/bad")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_bad_path, exist_ok=True)
        os.makedirs(test_good_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

        # copy images and masks
        with open(os.path.join(json_path, json_name)) as f:
            json_descr = json.load(f)

            # train data
            train_lst = json_descr["train"]
            for train_file in train_lst:
                image_full_path = os.path.join(riad_path, cls_name, train_file["image_path"])
                image_new_full_path = os.path.join(train_path, os.path.basename(train_file["image_path"]))
                shutil.copy(image_full_path, image_new_full_path)

            # test data
            test_lst = json_descr["test"]
            for test_file in test_lst:
                image_full_path = os.path.join(riad_path, cls_name, test_file["image_path"])
                if test_file["anomaly_class"] == "OK":
                    # good files
                    image_new_full_path = os.path.join(test_good_path, os.path.basename(test_file["image_path"]))
                    shutil.copy(image_full_path, image_new_full_path)
                else:
                    # bad files
                    image_new_full_path = os.path.join(test_bad_path, os.path.basename(test_file["image_path"]))
                    shutil.copy(image_full_path, image_new_full_path)
                    # ground truth masks
                    gt_full_path = os.path.join(riad_path, cls_name, test_file["mask_path"])
                    gt_new_full_path = os.path.join(gt_path, os.path.basename(test_file["mask_path"]))
                    shutil.copy(gt_full_path, gt_new_full_path)



