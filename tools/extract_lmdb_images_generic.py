import os
import argparse
import lmdb
import cv2
import numpy as np

def extract_images(lmdb_dir, out_dir, prefix="img", only_hr=False):
    os.makedirs(out_dir, exist_ok=True)
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
    saved = 0
    tried = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            tried += 1
            key = k.decode("utf-8", errors="ignore").lower()

            # 可选：只保留 key 中带 hr 的图
            if only_hr and ("hr" not in key):
                continue

            arr = np.frombuffer(v, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            fn = f"{prefix}_{saved:07d}.png"
            cv2.imwrite(os.path.join(out_dir, fn), img)
            saved += 1

    env.close()
    print(f"[OK] {lmdb_dir} -> {out_dir}, saved={saved}, tried={tried}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--prefix", default="img", type=str)
    parser.add_argument("--only_hr", action="store_true")
    args = parser.parse_args()

    extract_images(args.lmdb_dir, args.out_dir, args.prefix, args.only_hr)