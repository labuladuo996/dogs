import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


# Token 从服务器环境变量读取，不写进代码文件
token = os.environ["KAGGLE_API_TOKEN"]
url = "https://www.kaggle.com/api/v1/competitions/data/download-all/dog-breed-identification"

data_dir = Path("dog-breed-identification")
zip_path = Path("dog-breed-identification.zip")
data_dir.mkdir(exist_ok=True)

# Bearer Token 用于访问 Kaggle 官方下载接口
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
with urllib.request.urlopen(req) as res, open(zip_path, "wb") as f:
    shutil.copyfileobj(res, f)

# 解压后会得到 train、test、labels.csv 和 sample_submission.csv
with zipfile.ZipFile(zip_path) as z:
    z.extractall(data_dir)

print("downloaded:", zip_path)
print("extracted to:", data_dir)
