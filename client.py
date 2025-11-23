
import argparse
import io
import os
import threading
from typing import List
import requests
from PIL import Image
from stag import SKTagger, raw_extensions
from xmphandler import XMPHandler

class RemoteTagger(SKTagger):
    def __init__(self, server_url: str, **kwargs):
        super().__init__(**kwargs)
        self.server_url = server_url

    def get_tags_for_image(self, pil_image: Image.Image) -> str:
        # Resize the image to 384x384 for optimized transfer
        pil_image.thumbnail((384, 384))

        # Convert PIL image to byte stream
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format='JPEG')
        byte_arr.seek(0)

        try:
            files = {'file': ('image.jpg', byte_arr, 'image/jpeg')}
            response = requests.post(f"{self.server_url}/tag_image", files=files)
            response.raise_for_status()

            data = response.json()
            if 'tags' in data:
                return "|".join(data['tags'])
            elif 'error' in data:
                print(f"Error from server: {data['error']}")
                return ""
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to the server: {e}")
            return ""
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return ""

def main():
    parser = argparse.ArgumentParser(description='STAG client for remote image tagging.')
    parser.add_argument('imagedir', metavar='DIR', help='Path to the directory with images.')
    parser.add_argument('--server_url', metavar='URL', default='http://127.0.0.1:5000', help='URL of the STAG server.')
    parser.add_argument('--prefix', metavar='STR', default='st', help='Top category for tags.')
    parser.add_argument('--force', action='store_true', help='Force tagging even if images are already tagged.')
    parser.add_argument('--test', action='store_true', help="Don't write or modify XMP files.")
    parser.add_argument('--prefer-exact-filenames', action='store_true', help="Create XMP files with exact filenames.")
    args = parser.parse_args()

    tagger = RemoteTagger(
        server_url=args.server_url,
        model_path="",  # Not needed for client
        image_size=384, # Not needed for client
        force_tagging=args.force,
        test_mode=args.test,
        prefer_exact_filenames=args.prefer_exact_filenames,
        tag_prefix=args.prefix,
    )

    stop_event = threading.Event()
    stop_event.clear()
    tagger.enter_dir(args.imagedir, stop_event)

if __name__ == "__main__":
    main()
