import argparse
import os
from multiprocessing import Pool

from objathor.utils.download_utils import download_with_progress_bar

ALL_CKPT_IDS = [
    "DINOv2-ViTs-3-CHORES-S",
    "SigLIP-ViTb-3-CHORES-L",
    "SigLIP-ViTb-3-CHORES-S",
    "SigLIP-ViTb-3-CHORESNav-L",
    "SigLIP-ViTb-3-CHORESNav-S",
    "SigLIP-ViTb-3-double-det-CHORES-L",
    "SigLIP-ViTb-3-double-det-CHORES-S",
]


def download_ckpt(info):
    url = info["url"]
    save_dir = info["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, "checkpoint_final.ckpt")
    ckpt_url = f"{url}/checkpoint_final.ckpt"
    download_with_progress_bar(
        url=ckpt_url,
        save_path=ckpt_path,
        desc=f"Downloading: checkpoint_final.ckpt.",
    )

    config_path = os.path.join(save_dir, "config.yaml")
    config_url = f"{url}/config.yaml"
    download_with_progress_bar(
        url=config_url,
        save_path=config_path,
        desc=f"Downloading: config.yaml.",
    )


def main():
    parser = argparse.ArgumentParser(description="Trained ckpt downloader.")
    parser.add_argument("--save_dir", required=True, help="Directory to save the downloaded files.")
    parser.add_argument(
        "--ckpt_ids",
        default=None,
        nargs="+",
        help=f"List of ckpt names to download, by default this will include all ids. Should be a subset of: {ALL_CKPT_IDS}",
    )
    parser.add_argument("--num", "-n", default=1, type=int, help="Number of parallel downloads.")
    args = parser.parse_args()

    if args.ckpt_ids is None:
        args.ckpt_ids = ALL_CKPT_IDS

    os.makedirs(args.save_dir, exist_ok=True)

    download_args = []
    for ckpt_id in args.ckpt_ids:
        save_dir = os.path.join(args.save_dir, ckpt_id)
        download_args.append(
            dict(
                url=f"https://pub-bebbada739114fa1aa96aaf25c873a66.r2.dev/checkpoints/{ckpt_id}",
                save_dir=save_dir,
            )
        )

    with Pool(args.num) as pool:
        pool.map(download_ckpt, download_args)


if __name__ == "__main__":
    main()
