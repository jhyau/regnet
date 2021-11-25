import pprint

import subprocess
import os
import sys

data_root = '/juno/u/jyau/regnet/data/ASMR'

pref2dir = {
    'ASMR_Addictive_Tapping_1_Hr_No_Talking': 'ASMR_Addictive_Tapping_1_Hr',
    'The_Ultiimate_Tapping_ASMR_3_Hours_No_Talking': 'Ultimate_Tapping_ASMR_3_Hours'}

def get_file_paths(filelist):
    output = []
    with open(filelist, 'r') as f:
        for line in f:
            line = line.strip()
            prefix, _ = line.split('-', 1)
            dirname = pref2dir[prefix]
            dirpath = os.path.join(data_root, dirname)
            assert os.path.exists(dirpath), 'path %r not found' % dirpath
            path = os.path.join(dirpath, '%s.mp4' % line)
            assert os.path.exists(path), 'path %r not found' % path
            output.append((dirpath, os.path.basename(path)))
    return output

def main(args):
    binout = 'temp.h264'
    filelist = args[0]
    speed = float(args[1])
    assert speed in [0.5, 2.0, 0.75, 1.0]
    fps = 30 // speed
    print(fps)
    atempo = speed
    bin_cmd_fidx = 2
    binout = 'temp.h264'
    bin_cmd = ['ffmpeg', '-i', None, '-map', '0:v', '-c:v', 'copy', '-bsf:v',
               'h264_mp4toannexb', binout]
    speed_cmd_fidx = 8
    outpath_idx = -1
    speed_cmd = ['ffmpeg', '-fflags', '+genpts', '-r', str(fps), '-i', binout,
                 '-i', None, '-y',
                 '-map', '0:v', '-c:v', 'copy', '-map', '1:a', '-af',
                 'atempo=%s' % atempo, '-movflags', 'faststart', None]

    pathinfo = get_file_paths(filelist)
    # pprint.pprint(path_info)
    for dirname, fname in pathinfo:
        fpath = os.path.join(dirname, fname)
        outdir = '%sx%s' % (dirname, speed)
        outf = fname.replace('.mp4', '-%s.mp4' % speed)
        outpath = os.path.join(outdir, outf)
        os.makedirs(outdir, exist_ok=True)
        # adjust the commands for this round.
        bin_cmd[bin_cmd_fidx] = fpath
        speed_cmd[speed_cmd_fidx] = fpath
        speed_cmd[outpath_idx] = outpath
        subprocess.run(bin_cmd, check=True)
        subprocess.run(speed_cmd, check=True)
        os.remove(binout)

if __name__ == '__main__':
    main(sys.argv[1:])
