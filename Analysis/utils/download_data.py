import gdown
import os
from utils.logger import get_logger

logger = get_logger(__name__)

class DataDownloader:
    @staticmethod
    def download_folder(folder_id: str, download_dir: str):
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        
        logger.info(f"Downloading folder from {url} to {download_dir}...")
        gdown.download_folder(url, output=download_dir, quiet=False)
        logger.info(f"Downloaded folder to {download_dir}.")

# Example usage:
if __name__ == "__main__":
    folder_id = "1ABz2YNg0slMKUxKOw_EJgZenDjyMIjd8"
    download_dir = "./data"
    DataDownloader.download_folder(folder_id, download_dir)