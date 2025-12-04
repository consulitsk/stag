from huggingface_hub import hf_hub_download

def download_model():
    """Downloads the recognize-anything-plus-model."""
    hf_hub_download(
        repo_id="xinyu1205/recognize-anything-plus-model",
        filename="ram_plus_swin_large_14m.pth",
    )

if __name__ == "__main__":
    download_model()
