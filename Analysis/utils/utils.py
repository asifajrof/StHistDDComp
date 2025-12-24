import os
from utils.download_data import DataDownloader
from utils.logger import get_logger
logger = get_logger(__name__)

def download_data():
    folder_id = "1ABz2YNg0slMKUxKOw_EJgZenDjyMIjd8"
    download_dir = "./data"

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        logger.info(f"Starting download for folder ID: {folder_id} into directory: {download_dir}")
        DataDownloader.download_folder(folder_id, download_dir)
        logger.info("Download completed.")
    else:
        logger.info(f"Directory {download_dir} already exists. Skipping download.")