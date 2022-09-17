import os
import numpy as np
from multiprocessing import Pool
import pickle
import argparse

def process_audios(feat_fn):
    feat_fp = os.path.join(feat_dir, f'{feat_fn}.npy')

    if os.path.exists(feat_fp):
        try :
            ret = np.load(feat_fp).shape[-1]
        except: 
            print('error')
            return feat_fn, 0
        return feat_fn, ret
    else:
        return feat_fn, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute inception score")
    parser.add_argument("--dir", type=str, help="path to the input dir")
    args = parser.parse_args()

    feat_type = 'mel_64_200'
    exp_dir = args.dir  # base_out_dir from step2

    out_fp = os.path.join(exp_dir, 'dataset.pkl')

    # ### Process ###
    feat_dir = os.path.join(exp_dir, feat_type)

    feat_fns = [fn.replace('.npy', '') for fn in os.listdir(feat_dir)]

    pool = Pool(processes=20)
    dataset = []

    for i, (feat_fn, length) in enumerate(pool.imap_unordered(process_audios, feat_fns), 1):
        print(feat_fn)
        if length == 0:
            continue
        dataset += [(feat_fn, length)]

    with open(out_fp, 'wb') as f:
        pickle.dump(dataset, f)

    print(len(dataset))