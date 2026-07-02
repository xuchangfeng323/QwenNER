from huggingface_hub import snapshot_download
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

snapshot_download(repo_id="Qwen/Qwen2.5-7B",local_dir="./model/Qwen2.5-7B",resume_download=True,
    ignore_patterns=[
        ".gitattributes",
        "README.md",
    ])