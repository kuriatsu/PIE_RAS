# PIE_RAS
Recognition Assistance Interface test tools for PIE dataset

## Requirement
### Python
opencv-python, pickle, xml, multiprocessing
```bash
git clone git@github.com:kuriatsu/PIE_RAS.git pie_ras
```
## Prepare dataset
Download dataset
```bash
# download set03 clips
cd <path_to_the_root_folder>
bash pie_ras/download_clips.sh
# download annotation
git clone git@github.com:aras62/PIE.git
unzip PIE/annotations.zip </path/to/data_dir>/annotations
unzip PIE/annotations_attributes.zip <path_to_the_root_folder>/annotations_attributes
unzip PIE/annotations_vehicle.zip <path_to_the_root_folder>/annotations_vehicle
cd PIE
```

Split clip to image (only for prediction)
```python
from pie_data import PIE
pie_path = <path_to_the_root_folder>
imdb = PIE(data_path=pie_path)
imdb.extract_and_save_images(extract_frame_type='annotated')
```

Get intention prediction data (test data is set03)
Every frames are predicted using previous 15 frames and extracted to `<path_to_the_root_folder>/extracted_data`
```
mkdir <path_to_the_root_folder>/extracted_data/test
git clone git@github.com:kuriatsu/PIEPredict.git
python3 train_test.py 2
```

Rearange dataset to reduce processing load, result will be extracted to `<path_to_the_root_folder>/extracted_data/database.pkl`
```
python3 pie_extract_clip.py
```

