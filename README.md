# PIE_RAS
Recognition Assistance Interface test tools for PIE dataset

<table>
    <tr>
        <td>Intention</td>
        <td>Trajectory</td>
        <td>Traffic Light</td>
    <tr>
        <td><img src="https://user-images.githubusercontent.com/38074802/167537319-648eb6c3-fd0c-45bb-8d3c-4e4889887395.png"></td>
        <td><img src="https://user-images.githubusercontent.com/38074802/167537274-77fcb4bc-1d93-471d-b5fa-737496622a98.png"></td>
        <td><img src="https://user-images.githubusercontent.com/38074802/167537123-ca9877d0-7c19-4fed-9a9d-b954ae2472a5.png"></td>
    </tr>
</table>



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

Get traffic light detection result (at the last frame of each traffic light annotation)  
The result will be extracted to `<path_to_the_root_folder>/tlr_result`
```bash
python3 train_tlr.py
```

Rearange dataset to reduce processing load, result will be extracted to `<path_to_the_root_folder>/extracted_data/database.pkl`
```
python3 pie_extract_clip.py
```

![pie_experiment_design_comp](https://user-images.githubusercontent.com/38074802/167537879-658ba6ec-2204-4e65-b0b9-4573a22870a0.png)



## RUN
1. change subject name and experiment type from [tl, int, traj] in *pie_experiment.py*
2. run command
    ```bash
    python3 pie_experiment.py
    ```
3. Experiment

![pie_experiment_setup_cmp](https://user-images.githubusercontent.com/38074802/167537869-ca5aa073-dee4-4380-b530-2f40cd36d36e.png)

## Analyze
* Intervention time and accuracy
```bash
python3 result/analyze.py
```

* draw  PR curve of the predictions
```bash
python3 result/predict_analyze.py
```

* draw SDT curve
```bash
python3 result/sdt.py
```
