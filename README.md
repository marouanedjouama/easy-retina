create a new conda env (RECOMMENDED)
``` shell
conda create -n easy-retina python=3.10
conda activate easy-retina
```

download dependencies 
``` shell
pip install -r requirements.txt
```


Download face detection & recognition models

``` shell
bash scripts/download_models.sh
```

You should rearrange your dataset directory structure as illustrated below.

```bash
user
├── face_dataset
│   ├── Alice
│   │   ├── Alice1.jpg
│   │   ├── Alice2.jpg
│   ├── Bob
│   │   ├── Bob.jpg
```

OR you can download an already structured celeb faces dataset for testing.\
you will get an additional recognition_test folder with testing images.

``` shell
gdown 1Mgx5L8l2FO1laaWl0_vaE41D4Zohq1qs
unzip celeb_face_dataset.zip
rm celeb_face_dataset.zip
```


Process the dataset

``` python
cd recognition
python setup_vec_dataset.py
```

run the recognition model on a test image 

``` python
python recognize.py --image alice.jpg
```

run the detection model on a test image 

``` python
cd ../detection
python detect.py --image bob.jpg
```