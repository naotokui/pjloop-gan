import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute inception score")
    parser.add_argument("--dir", type=str, help="path to the input dir")
    args = parser.parse_args()

    feat_type = 'mel'
    exp_dir = args.dir

    out_dir = os.path.join(exp_dir, "mel_64_200")

    # ### Process ###

    dataset_fp = os.path.join(exp_dir, f'dataset.pkl')
    #feat_dir = os.path.join(exp_dir, feat_type)
    feat_dir = out_dir
    out_fp_mean = os.path.join(out_dir, f'mean.{feat_type}.npy')
    out_fp_std = os.path.join(out_dir, f'std.{feat_type}.npy')

    with open(dataset_fp, 'rb') as f:
        dataset = pickle.load(f)

    in_fns = [fn for fn, _ in dataset]

    scaler = StandardScaler()
    for fn in in_fns:
        #print(fn)
        in_fp = os.path.join(feat_dir, f'{fn}.npy')
        #print(fn)
        data = np.load(in_fp).T
        if np.isnan(data).any():
            print(fn)
            os.remove(os.path.join(feat_dir, f'{fn}.npy'))
            continue
        
        #print(data.shape)
        #print('data: ', data)
        scaler.partial_fit(data)
        #print(scaler.mean_, scaler.scale_)
        if True in np.isnan(scaler.scale_):
            break
    
    mean = scaler.mean_
    std = scaler.scale_
    np.save(out_fp_mean, mean)
    np.save(out_fp_std, std)
    print(mean, std)