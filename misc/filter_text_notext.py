import os, sys, shutil
import argparse


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbfiles', '-b', help='path to the bounding boxes to be filter', required=True)
    #parser.add_argument('--images', '-i', help='gt images coorsponding to bbfiles', required=True)
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory to save filtered data')
    return parser


def move_data(il_out, il_bbs, item):
    basename = item.split('.')[0]
    basename = basename[4:]
    coor_src = os.path.join(il_bbs, 'coordinates', basename)
    coor_dst = os.path.join(il_out, 'coordinates')
    shutil.move(coor_src, coor_dst, copy_function=shutil.copytree)


def handle_one_il(il_bbs, il_out):
    if os.path.exists(il_out) is False:
        os.makedirs(il_out)
        os.makedirs(os.path.join(il_out, 'coordinates'))

    for item in os.listdir(il_bbs):
        if item.endswith(".txt") is False:
            continue
        f = os.path.join(il_bbs, item)
        with open(f, 'r') as fp:
            l_cnt = len(fp.readlines())
        if l_cnt:
            move_data(il_out, il_bbs, item)


def filter_input(args):
    bbfs = args.bbfiles
    out = args.output
    #imgs = args.images

    if os.path.exists(bbfs) is False:
        print(f'Input BB path {bbfs} does not exists')
        exit(-1)
    #if os.path.exists(imgs) is False:
    #    print(f'Input images path {imgs} does not exits')
    #    exit(-1)

    for il in os.listdir(bbfs):
        il_bbs = os.path.join(bbfs, il)
        il_out = os.path.join(out, il)

        if os.path.isdir(il_bbs) is False:
            print(f'skipped file {il}')
            continue
        handle_one_il(il_bbs, il_out)


def start_main():
    args = process_args().parse_args()
    filter_input(args)


if __name__ == '__main__':
    start_main()
