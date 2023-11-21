import sys, os, argparse, shutil
from tqdm import tqdm
import json


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', '-j', help='path to the json file containing BBs', required=True)
    parser.add_argument('--output', '-o', required=False, default="./output",
                        help='Output directory to save bounding boxes in .txt format')
    return parser


def start_process(args):
    if os.path.exists(args.json_dir) is False:
        print(f"The input path {args.json_dir} does not exists")
        return
    if os.path.exists(args.output) is False:
        os.makedirs(args.output)

    jsons = sorted(os.listdir(args.json_dir))
    pbar = tqdm(jsons)
    for idx, item in enumerate(pbar):
        #print(item)
        pbar.set_postfix_str(item)
        with open(os.path.join(args.json_dir, item), 'r') as f:
            data = json.load(f)
        with open(os.path.join(args.output, f"gt_{item.split('.')[0]}.txt"), 'a') as of:
            for key in data.keys():
                if 'coordinates' not in data[key]:
                    continue
                coor = [str(itm) for sl in data[key]['coordinates'] for itm in sl]
                line = ",".join(coor)
                line += ",__\n"
                of.writelines(line)
        #if idx == 1:
        #    break


def start_main():
    args = process_args().parse_args()
    start_process(args)


if __name__ == '__main__':
    start_main()
