import argparse
import os, sys
import shutil
import json


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', '-i', help='Images to rename', required=True)
    parser.add_argument('--output', '-o', required=False, default='./results',
                        help='Output directory to save renamed images and name-map json')
    parser.add_argument('--name', '-n', help='Basename of the dataset',
                        required=False, default='IITJ-ILST')
    return parser


def sanity_checks(args):
    if os.path.exists(args.images) is False:
        raise Exception(f'{args.images}: Not found')

    if os.path.exists(args.output) is False:
        os.makedirs(args.output)
    path = os.path.join(args.output, args.name)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return 0


def handle_format_and_copy(im, lipi, newname, odir):
    ext = im.split('.')[-1]
    '''
    if ext.lower() == 'png':
        ext = 'jpg'
        #handle png
        print(f'{im}')
    '''
    if ext.lower() == 'jpeg':
        ext = 'jpg'
    newname = newname + f'.{ext.lower()}'
    dst = os.path.join(odir, newname)
    src = os.path.join(lipi, im)
    shutil.copy2(src, dst)
    #print(f'{src} --> {dst}')
    return newname


def process_one_script(il, basepath, opath, dsname):
    name_map = {}

    basename = il.lower()
    lipi = os.path.join(basepath, il)

    opath = os.path.join(opath, dsname)
    json_of = os.path.join(opath, il, basename + '.json')
    imlist_of = os.path.join(opath, il, basename + '.txt')
    odir = os.path.join(opath, il, 'images')

    if os.path.exists(odir) is False:
        os.makedirs(odir)
    elif os.path.isdir(odir) is False:
        print(f'File {odir} already exists. skipped!')
        return name_map
    if os.path.exists(imlist_of) is True:
        os.remove(imlist_of)

    with open(imlist_of, 'a') as iml_of:
        for idx, im in enumerate(sorted(os.listdir(lipi))):
            num = idx + 1
            newname = basename + f'_{num:05d}'
            newname = handle_format_and_copy(im, lipi, newname, odir)
            name_map[newname] = im
            iml_of.write(newname+'\n')

    jobj = json.dumps(name_map, indent=1)
    with open(json_of, 'w') as of:
        of.write(jobj)

    return name_map

def process_images(args):
    impath = args.images
    for il in sorted(os.listdir(impath)):
        imdir = os.path.join(impath, il)
        if os.path.isdir(imdir) is False:
            continue
        _ = process_one_script(il, impath, args.output, args.name)


def start_main():
    print(f'Simple image file renaming utility')
    args = process_args().parse_args()
    sanity_checks(args)
    process_images(args)


if __name__ == '__main__':
    start_main()
