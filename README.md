Load face recognition model

``` shell
bash scripts/get_model.sh
```

You should rearrange your dataset directory structure as illustrated below.

```bash
user
├── database
│   ├── Alice
│   │   ├── Alice1.jpg
│   │   ├── Alice2.jpg
│   ├── Bob
│   │   ├── Bob.jpg
```


Process the dataset

``` python
python setup_vec_dataset.py
```


run the model on a test image 

``` python
python recognize.py
```