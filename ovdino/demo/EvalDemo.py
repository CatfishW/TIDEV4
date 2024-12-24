# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import sys
import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

sys.path.insert(0, "./")  # noqa
from demo.predictors import OVDINODemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image,annotations_to_instances
from detectron2.utils.logger import setup_logger
from detrex.data.datasets import clean_words_or_phrase
from tools.crop_image import annotate_image_with_detectron2
from tools.find_support_images import find_filename_by_support_ids,retrieve_annotation_by_filename
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_AVAILABLE = True
except ImportError:
    print("SAM2 is not installed.")
    SAM2_AVAILABLE = False
#import boxmode
from detectron2.structures import BoxMode

# constants
WINDOW_NAME = "COCO detections"


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    #cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def renormalize_bbox(class_label, x_center, y_center, width, height, image_width, image_height):
    x_center = x_center * image_width
    y_center = y_center * image_height
    width = width * image_width
    height = height * image_height
    
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)
    
    return class_label, x_min, y_min, x_max, y_max
def get_parser():
    parser = argparse.ArgumentParser(
        description="detrex demo for visualizing customized inputs"
    )
    parser.add_argument(
        "--config-file",
        default="projects/ovdino/configs/tidev4_swin_tiny224_bert_base_infer_demo.py",
        #default="projects/ovdino/configs/ovdino_swin_tiny224_bert_base_infer_demo.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--sam-config-file",
        default=None,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--sam-init-checkpoint",
        default=None,
        metavar="FILE",
        help="path to sam checkpoint file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--cross_prompt_image",
        nargs="+",
        help="A list of space separated prompt images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--category_names", nargs="+", help="A list of sapce separete category names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="RGB",
        help="The format of the loading images.",
    )
    parser.add_argument(
        "--metadata_dataset",
        type=str,
        default="coco_2017_val",
        help="The metadata infomation to be used. Default to COCO val metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    import json
    import torch
    thresholds = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65]
    for threshold in thresholds:
        args = get_parser().parse_args()
        args.confidence_threshold = threshold
        args.output = os.path.join("demo/imgs/output", f"threshold_{threshold}")
        os.makedirs(args.output, exist_ok=True)
        mp.set_start_method("spawn", force=True)
        support_json = "datasets/DatasetToBeEval/Support/support.json"
        support_json = json.load(open(support_json))
        images_names = ["datasets/DatasetToBeEval/Query/images/"+item["image_name"] for item in support_json]
        support_ids = [item["support_id"] for item in support_json]
        cross_prompt_images = [find_filename_by_support_ids(support_id) for support_id in support_ids]
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(args))
        cfg = setup(args)
        #args.category_names = "car tree building"
        args.input = images_names
        args.cross_prompt_image = cross_prompt_images
        category_names = None#['car','bus','person','dog',"phone","human head"]
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.train.init_checkpoint)
        
        

        model.eval()

        if args.sam_config_file is not None and SAM2_AVAILABLE:
            logger.info(f"Building SAM2 model: {args.sam_config_file}")
            sam_model = build_sam2(
                args.sam_config_file, args.sam_init_checkpoint, device="cuda"
            )
            sam_predictor = SAM2ImagePredictor(sam_model)
        else:
            sam_predictor = None

        if args.category_names is not None:
            category_names = args.category_names
        if category_names is not None:
            category_names = [clean_words_or_phrase(cat_name) for cat_name in category_names]
            print(category_names)

        demo = OVDINODemo(
            model=model,
            sam_predictor=sam_predictor,
            min_size_test=args.min_size_test,
            max_size_test=args.max_size_test,
            img_format=args.img_format,
            metadata_dataset=args.metadata_dataset,
        )
        if args.input:
            if len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
            if os.path.isdir(args.input[0]):
                args.input = [
                    os.path.join(args.input[0], file) for file in os.listdir(args.input[0])
                ]
                os.makedirs(args.output, exist_ok=True)
            index = 0
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                # h, w
                if args.cross_prompt_image is not None:
                    start_time = time.time()
                    visual_embeds_list = []
                    for cross_prompt_image in args.cross_prompt_image[index]:
                        list_of_annotations = retrieve_annotation_by_filename(os.path.basename(cross_prompt_image))
                        cross_prompt_image = read_image(cross_prompt_image, format="BGR")
                        cross_prompt_image = cv2.resize(cross_prompt_image, (640, 640))
                        image_shape = cross_prompt_image.shape[:2]
                        instances = annotations_to_instances(list_of_annotations, image_shape)
                        #visualize cross prompt image and its annotations
                        x_min, y_min, x_max, y_max = instances.gt_boxes.tensor[0].tolist()
                        # cv2.rectangle(cross_prompt_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                        # cv2.imshow("query_image", img)
                        # cv2.imshow("cross_prompt_image", cross_prompt_image)
                        # cv2.waitKey(300)
                        # print("cross_prompt_image's category_id:",instances.gt_classes[0].tolist())
                        visual_embeds = demo.run_on_image(
                            cross_prompt_image, category_names, args.confidence_threshold,instances=instances,extract_visual_prompt_mode=True
                        )
                        visual_embeds_list.append(visual_embeds)
                    #merge visual embeds into (1,150,768)
                    visual_embeds = torch.stack(visual_embeds_list)
                    visual_embeds = visual_embeds.mean(dim=0)     
                    predictions, visualized_output = demo.run_on_image(
                        img, category_names, args.confidence_threshold,visual_embeds=visual_embeds
                    )
                else:
                    annotations = annotate_image_with_detectron2(args.input[0])
                    instances = annotations_to_instances(annotations, image_shape)
                    start_time = time.time()
                    predictions, visualized_output = demo.run_on_image(
                        img, category_names, args.confidence_threshold,instances=instances
                    )
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        path,
                        (
                            "detected {} instances".format(len(predictions["instances"]))
                            if "instances" in predictions
                            else "finished"
                        ),
                        time.time() - start_time,
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert (
                            len(args.input) == 1
                        ), "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                else:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    #cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
                index+=1
        elif args.webcam:
            assert args.input is None, "Cannot have both --input and --webcam!"
            assert args.output is None, "output not yet supported with --webcam!"
            cam = cv2.VideoCapture(0)
            for vis in tqdm.tqdm(demo.run_on_video(cam)):
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, vis)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
            cam.release()
            cv2.destroyAllWindows()
        elif args.video_input:
            video = cv2.VideoCapture(args.video_input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(args.video_input)
            codec, file_ext = (
                ("x264", ".mkv")
                if test_opencv_video_format("x264", ".mkv")
                else ("mp4v", ".mp4")
            )
            if codec == ".mp4v":
                warnings.warn("x264 codec not available, switching to mp4v")
            if args.output:
                if os.path.isdir(args.output):
                    output_fname = os.path.join(args.output, basename)
                    output_fname = os.path.splitext(output_fname)[0] + file_ext
                else:
                    output_fname = args.output
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(args.video_input)
            for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                if args.output:
                    output_file.write(vis_frame)
                else:
                    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                    cv2.imshow(basename, vis_frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
            video.release()
            if args.output:
                output_file.release()
            else:
                cv2.destroyAllWindows()
                cv2.destroyAllWindows()
                cv2.destroyAllWindows()