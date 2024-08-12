```
$ git clone ${repo}
$ cd ${repo}
$ python -m venv .env
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ pip install .
```

Download fine tuned embedding model and document index
```
# from the repo root directory

$ mkdir data
$ cd data
$ curl "${get the presign url from my email}" -o "inference_data.zip"
$ unzip inference_data.zip 
$ cd ..
$ 

```


Get the latest dump version from:
https://dumps.wikimedia.org/enwiki/

Update the snapshot version in wikisearch/data/get_wikipedia_data.py

```
cd wikisearch/data
% python get_wikipedia_data.py
```