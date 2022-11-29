import os
import logging
from tqdm import tqdm
import argparse
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def intersection(paths_list):
    
    for idx, paths in enumerate(paths_list):
        # Get File Names without extension
        # directory for WaSR results -> remove 'm'
        if set([os.path.splitext(x)[0][-1] for x in paths]) == set('m'):
            paths = [os.path.splitext(x)[0][:-1] for x in paths]
        else: paths = [os.path.splitext(x)[0] for x in paths]
        
        if idx==0:
            inters_list = paths
        else: inters_list = [x for x in inters_list if x in paths]
    
    return inters_list

def set_logger(logging_level=logging.INFO, log_name="root", logging_filepath=None):
    # Define Logger
    if log_name == "root":
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name=log_name)

    # Set Logger Display Level
    logger.setLevel(level=logging_level)

    # Set Formatter
    formatter = logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    # formatter = "[%(levelname)s] | %(asctime)s : %(message)s"

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # stream_handler.setFormatter(CustomFormatter(formatter))
    logger.addHandler(stream_handler)

    # Set File Handler
    if logging_filepath is not None:
        file_handler = logging.FileHandler(logging_filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_arguments():
    # Declare Argument Parser
    parser = argparse.ArgumentParser(
        description="WaSR Segmentation Result Comparison Tool",
    )

    # Get Result Image Paths
    parser.add_argument(
        "-P", "--img_paths", nargs="+", required=True, help="Set Image Paths"
    )
    parser.add_argument(
        "-S", "--save_base_path", type=str, default=os.path.dirname(__file__),
        help="Set Result Save Base Path"
    )

    # Parse Arguments
    args = parser.parse_args()
    return args


def check_validity_load_results(args, logger):
    # Check for Assertion
    assert len(args.img_paths) > 1, "At least two paths are required...!"
    paths_dict = {}
    for img_path in args.img_paths:
        assert os.path.isdir(img_path), "Path {} does not exist...!".format(img_path)
        paths_dict_files = sorted(os.listdir(img_path))
        assert len(paths_dict_files) > 0, "There are no files...!"
        paths_dict[img_path] = paths_dict_files
    file_names = intersection(list(paths_dict.values()))
    assert len(file_names) > 0, "The directories do not contain the same files...!"

    # Transpose Dictionary
    fn_wise_dict = {k: [] for k in file_names}
    for fn_idx, fn in enumerate(fn_wise_dict.keys()):
        load_logging_msg = \
            "Loading Image Path for [{}]...! ({}/{})".format(fn, fn_idx+1, len(fn_wise_dict))
        logger.info(load_logging_msg)

        for base_path, _fn_list in paths_dict.items():
            cnt = 0
            for _fn in _fn_list:
                cnt += 1
                if (os.path.splitext(_fn)[0] == fn) or ((os.path.splitext(_fn)[0][-1] == 'm') and (os.path.splitext(_fn)[0][:-1] == fn)):
                    fn_wise_dict[fn].append(os.path.join(base_path, _fn))
                    break

    # Mask Results Dictionary
    mask_results_dict = {k: {} for k in fn_wise_dict.keys()}
    for idx, (k, v) in enumerate(fn_wise_dict.items()):
        load_logging_msg = \
            "Loading Images for [{}]...! ({}/{})".format(k, idx+1, len(fn_wise_dict))
        logger.info(load_logging_msg)

        for _v in v:
            mask_results_dict[k][_v] = cv2.imread(_v, cv2.IMREAD_UNCHANGED)

    return mask_results_dict


if __name__ == "__main__":
    # Get Argparse Arguments
    args = get_arguments()

    # Set Logger
    _logger = set_logger()

    # Get File Info
    mask_results_dict = check_validity_load_results(args=args, logger=_logger)

    # Make Result Directories
    result_dir = os.path.join(args.save_base_path, "results")
    if os.path.isdir(result_dir) is False:
        os.makedirs(result_dir)

    # Draw Figures for every mask dictionary
    for idx, (fname, masks_dict) in enumerate(mask_results_dict.items()):
        # Logger
        draw_logging_msg = \
            "Drawing Images for [{}]...! ({}/{})".format(fname, idx+1, len(mask_results_dict))
        _logger.info(draw_logging_msg)

        # Get Ablation Number
        ablation_number = len(masks_dict)

        # Get Data
        mask_src_paths, masks = list(masks_dict.keys()), list(masks_dict.values())

        # Declare Plt Figure
        row = int(np.sqrt(ablation_number))
        col = ablation_number//row if ablation_number % row == 0 else ablation_number//row + 1
        fig_mask = plt.figure(figsize=(4*col, 3*row), dpi=150)
        fig_mask.suptitle(fname, fontsize=15)

        # Loop for Ablation Number
        for abl_idx in range(ablation_number):
            mask_src_path, mask = mask_src_paths[abl_idx], masks[abl_idx]
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask_src_name = mask_src_path.replace('/images','').split('/')[-2]

            # Set Axes
            curr_axes = fig_mask.add_subplot(row, col, abl_idx+1)
            curr_axes.grid(False)
            curr_axes.set_xticks([])
            curr_axes.set_yticks([])
            curr_axes.set_title(mask_src_name, fontsize=10)

            # Draw Mask
            curr_mask = curr_axes.imshow(mask_bgr)

        # Save Figure
        fig_mask.savefig(
            os.path.join(result_dir, "{}.png".format(fname)),
        )
        save_fig_logging_msg = \
            "Saving Figure for [{}]...! ({}/{})".format(
                fname, idx + 1, len(mask_results_dict)
            )
        _logger.info(save_fig_logging_msg)

        # Close Figure
        plt.close(fig=fig_mask)