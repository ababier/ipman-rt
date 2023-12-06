import os
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZipFile

import wget

from open_kbp.clean_data import clean_data
from utils import DATA_DIR


def download_data() -> str:
    try:
        print("Downloading open-kbp-data.zip (600 Mb). This may take a few minutes ")
        filename = wget.download(
            url="https://cyc7ww.sn.files.1drv.com/y4m5Sz5ryDCH-5Em0nWUGc0P37uluQxTtWqn-BJfiiaBEgCOjpcLYC_fWQEs4x7CdfqTYhP1ysW0pDARMmuV6V-2K0n"
            "ZbuFfJlB4QXjWG3mNSrKp0WN6iegdyamdQcUhMb4h_ilfkslrndgxYmn1mFVQx3zHfkru0rDQIOWs4xWEzS_rB2z45Vvy1pk1tP1VsJojJ2N49DCtyWAOjwBN5T0RQ"
        )
        print(f"{filename} was downloaded successfully.")
        return filename
    except HTTPError:
        raise ValueError(
            f"Download link is broken."
            f"Please download data from https://1drv.ms/u/s!AuMp5xOixVAhhI4MWV8d-6-hH7CigA?e=sbhLLc and move the zip file to {Path().resolve()}."
            f"If the link does not work please raise an issue in the GitHub repository at https://github.com/ababier/ipman-rt/issues"
        )


if __name__ == "__main__":
    # Download data if necessary
    if not DATA_DIR.exists() and not Path("open-kbp-data.zip").exists():
        download_data()

    # Unzip data if data not already unzipped
    if not DATA_DIR.exists() and not Path("open-kbp-data").exists():
        with ZipFile("open-kbp-data.zip") as zipped_object:
            zipped_object.extractall(path=DATA_DIR)
        os.remove("open-kbp-data.zip")
        os.rename("open-kbp-data", DATA_DIR)

    # Clean data to proceed with IPMAN
    clean_data()
