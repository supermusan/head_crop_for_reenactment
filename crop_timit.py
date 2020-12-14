import numpy as np
import pandas as pd
import imageio
import os
import subprocess
import warnings
import glob
import time
from util import bb_intersection_over_union, join, scheduler, crop_bbox_from_frames, save, scheduler_count, bb_space
from argparse import ArgumentParser
from skimage.transform import resize
warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')
REF_FRAME_SIZE = 360
REF_FPS = 25


def extract_bbox(frame, fa):
    # frame[..., ::-1] swith channel 1 and 3
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) != 0:
        bbox = max([(bb_space(bbox), tuple(bbox)) for bbox in bboxes])[1]
    else:
        bbox = np.array([0, 0, 0, 0, 0])
    return np.maximum(np.array(bbox), 0)


def save_bbox_list(video_path, bbox_list):
    f = open(os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt'), 'w')
    print("LEFT,TOP,RIGHT,BOT", file=f)
    for bbox in bbox_list:
        print("%s,%s,%s,%s" % tuple(bbox[:4]), file=f)
    f.close()


def estimate_bbox(video_path, fa, check_exists=False):
    if check_exists:
        if os.path.exists(os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt')):
            print('Existed:{}'.format(
                os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt')))
            return
    reader = imageio.get_reader(video_path)
    bbox_list = []

    try:
        for i, frame in enumerate(reader):
            mult = frame.shape[0] / REF_FRAME_SIZE
            frame = resize(frame, (REF_FRAME_SIZE, int(frame.shape[1] / mult)), preserve_range=True)

            bbox = extract_bbox(frame, fa)
            bbox_list.append(bbox * mult)
    except IndexError:
        None

    save_bbox_list(video_path, bbox_list)


def store(frame_list, tube_bbox, video_id, person_id, start, end, video_count, args):
    out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox, min_frames=args.min_frames,
                                            image_shape=args.image_shape, min_size=args.min_size, 
                                            increase_area=args.increase)
    if out is None:
        return []

    name = (person_id + "#" + video_id + '#' + str(video_count).zfill(3) + ".mp4")
    partition = 'test' if person_id in TEST_PERSONS else 'train'
    save(os.path.join(args.out_folder, partition, name), out, args.format, fps=REF_FPS)
    return [{'bbox': '-'.join(map(str, final_bbox)), 'start': start, 'end': end, 'fps': REF_FPS,
             'video_id': '#'.join([video_id, person_id]), 'height': frame_list[0].shape[0], 
             'width': frame_list[0].shape[1], 'partition': partition}]


def crop_video_neighbor(person_id, video_id, video_path, args):
    bbox_path = os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt')
    reader = imageio.get_reader(video_path)

    d = pd.read_csv(bbox_path)
    video_count = 0
    prev_bbox = None
    start = 0
    tube_bbox = None
    frame_list = []
    chunks_data = []

    try:
        for i, frame in enumerate(reader):
            bbox = np.array(d.iloc[i])

            if prev_bbox is None:
                prev_bbox = bbox
                start = i
                tube_bbox = bbox

            if bb_intersection_over_union(prev_bbox, bbox) < args.iou_with_initial or len(
                    frame_list) >= args.max_frames:
                chunks_data += store(frame_list, tube_bbox, video_id, person_id, start, i, video_count, args)
                video_count += 1
                start = i
                tube_bbox = bbox
                frame_list = []
            prev_bbox = bbox
            tube_bbox = join(tube_bbox, bbox)
            frame_list.append(frame)
    except IndexError as e:
        None

    chunks_data += store(frame_list, tube_bbox, video_id, person_id, start, i + 1, video_count, args)

    return chunks_data


def crop_video(person_id, video_id, video_path, args):
    bbox_path = os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt')
    reader = imageio.get_reader(video_path)

    d = pd.read_csv(bbox_path)
    video_count = 0
    initial_bbox = None
    start = 0
    tube_bbox = None
    frame_list = []
    chunks_data = []

    try:
        for i, frame in enumerate(reader):
            bbox = np.array(d.iloc[i])

            if initial_bbox is None:
                initial_bbox = bbox
                start = i
                tube_bbox = bbox

            if bb_intersection_over_union(initial_bbox, bbox) < args.iou_with_initial or len(
                    frame_list) >= args.max_frames:
                chunks_data += store(frame_list, tube_bbox, video_id, person_id, start, i, video_count, args)
                video_count += 1
                initial_bbox = bbox
                start = i
                tube_bbox = bbox
                frame_list = []
            tube_bbox = join(tube_bbox, bbox)
            frame_list.append(frame)
    except IndexError as e:
        None
    
    chunks_data += store(frame_list, tube_bbox, video_id, person_id, start, i + 1, video_count, args)

    return chunks_data

def count_dataset(params):
    person_id, device_id, args = params
    data_folder_id = os.path.join(args.dataset_folder, person_id)
    blendshape_videos = [item for item in glob.glob(os.path.join(data_folder_id, "*/*/*.mp4"))]
    other_videos = [item for item in glob.glob(os.path.join(data_folder_id, "*/*/*/*.mp4"))]
    print("data_folder_id:{}\t sum:{}".format(data_folder_id, len(blendshape_videos) + len(other_videos)))
    return len(blendshape_videos) + len(other_videos)

def run(params):
    person_id, device_id, args = params
    if args.neighbor_or_initial == 'initial':
        crop_video_fn = crop_video
    else:
        crop_video_fn = crop_video_neighbor
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    # update the config options with the config file
    if args.estimate_bbox:
        # print("loading face_landmarker.")
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    #data_folder_id = os.path.join(args.dataset_folder, person_id)
    #print("data_folder_id:", data_folder_id)
    #blendshape_videos = [(item, os.path.split(item)[0].replace(data_folder_id+'/', "").replace('/', '-'))
    #                     for item in glob.glob(os.path.join(data_folder_id, "*/*/*.mp4"))]
    #other_videos = [(item, os.path.split(item)[0].replace(data_folder_id+'/', "").replace('/', '-'))
    #                for item in glob.glob(os.path.join(data_folder_id, "*/*/*/*.mp4"))]
    #videos_path_list = sorted(blendshape_videos + other_videos, key=lambda x:x[1])
    # videos_path_list = sorted(os.path.join(args.dataset_folder, person_id))
    videos_path_list = [os.path.join(args.dataset_folder, person_id)]
    print(videos_path_list[0])


    chunks_data = []
    video_id = ""

    try:
        if args.estimate_bbox:
            for chunk in videos_path_list:
                while True:
                    try:
                        estimate_bbox(chunk, fa, args.bbox_check_exists)
                        break
                    except RuntimeError as e:
                        if str(e).startswith('CUDA'):
                            print("Warning: out of memory, sleep for 1s")
                            time.sleep(1)
                        else:
                            print(e)
                            break
        if args.crop:
            for chunk in videos_path_list:
                if not os.path.exists(os.path.join(args.bbox_folder, os.path.basename(chunk)[:-4] + '.txt')):
                    print(os.path.join(args.bbox_folder, os.path.basename(chunk)[:-4] + '.txt'))
                    print("BBox not found %s" % chunk)
                    continue

                chunks_data += crop_video_fn(person_id, video_id, chunk, args)

    except Exception as e:
        print(e)

    return chunks_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder",
                        default='/data_activate/wangweisen/reenactment/dataset/TIMIT/real_videos',
                        help='Root path to src videos')

    parser.add_argument("--iou_with_initial", type=float, default=0.25,
                        help="The minimal allowed iou with inital/neighbor bbox")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--min_frames", default=64, type=int, help='Mimimal number of frames')
    parser.add_argument("--max_frames", default=1024, type=int, help='Maximal number of frames')
    parser.add_argument("--min_size", default=256, type=int, help='Minimal allowed size')
    parser.add_argument("--format", default='.png', help='Store format (.png, .mp4)')
    parser.add_argument("--neighbor_or_initial", default='initial',
                        help='calculate IOU between current box and initial/neighbor box')

    parser.add_argument("--bbox_folder", default='bbox-timit', help="Path to folder with bboxes")
    parser.add_argument("--out_folder", default='timit-png', help='Folder for processed dataset')
    parser.add_argument("--chunks_metadata", default='timit-metadata.csv', help='File with metadata')

    parser.add_argument("--workers", default=1, type=int, help='Number of parallel workers')
    parser.add_argument("--device_ids", default="0", help="Names of the devices comma separated.")

    parser.add_argument("--no-estimate-bbox", dest="estimate_bbox", action="store_false",
                        help="Do not estimate the bboxes")
    parser.add_argument("--no-crop", dest="crop", action="store_false", help="Do not crop the videos")
    parser.add_argument("--only_count", dest="only_count", action="store_true", help="Count the samples in dataset")
    parser.add_argument("--bbox_check_exists", dest="bbox_check_exists", action="store_true",
                        help="skip the existed bbox *.txt")

    parser.set_defaults(crop=True)
    parser.set_defaults(estimate_bbox=True)
    parser.set_defaults(only_count=False)
    parser.set_defaults(bbox_check_exists=False)

    args = parser.parse_args()

    TEST_PERSONS = []
    if not os.path.exists(args.bbox_folder):
        os.makedirs(args.bbox_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    ids = set(os.listdir(args.dataset_folder))
    ids = sorted(list(ids))
    print(args)
    if args.neighbor_or_initial == 'initial':
        print("crop strategy is 'initial")
    else:
        print("crop strategy is 'neighbor")
    if args.only_count:
        scheduler_count(ids, count_dataset, args)
        exit(0)
    # print(ids)

    scheduler(ids, run, args)
