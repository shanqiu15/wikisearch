`python -m venv .env`

` %  source .venv/bin/activate`


Get the latest dump version from:
https://dumps.wikimedia.org/enwiki/

Update the snapshot version in wikisearch/data/get_wikipedia_data.py

```
cd wikisearch/data
% python get_wikipedia_data.py
```