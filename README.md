
## How to run the code
Step 1: Clone the repo and install dependencies

```
$ git clone ${repo}
$ cd ${repo}
$ python -m venv .env
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ pip install .
```

Step 2: Download fine tuned embedding model and document index
```
# from the repo root directory

$ mkdir data
$ cd data
$ curl "${get the presign url from my email}" -o "inference_data.zip"
$ unzip inference_data.zip 
$ cd ..
```

Step 3: set OpenAI API key. To run the code in question answer mode you need to set OpenAI key in env variable
```
% export OPENAI_API_KEY= Your OpenAI Key
```
we use gpt-4o for answer generation. Runing the code without setting OpenAI key will return only the retrieved document chunks


Step 4: Run the search
```
% wikisearch "Were Scott Derrickson and Ed Wood of the same nationality?"
```



## How the search engine was created?

### Download the latest wikipedia dump
Get the latest dump version from:
https://dumps.wikimedia.org/enwiki/

Clone and setup dependecies for this repo: `https://github.com/huggingface/olm-datasets`
Then use scripts/get_wikipedia_data.py to pull the latest wikipedia dump, you can change the SNAPSHOT_DATE to download a different snapshot

```
% cd scripts
% python get_wikipedia_data.py
```

### Download the Hopot QA dataset from huggingface

Here are more details about the dataset https://huggingface.co/datasets/hotpotqa/hotpot_qa

```
import datasets

retrieve_dataset = datasets.load_dataset("hotpotqa/hotpot_qa", 'distractor', trust_remote_code=True)
```

After download the dataset remove all questions whose support documents doesn't exist in the new wikipedia snapshot, and get original wikipedia pages for each query.

The processed dataset and wikipedia documents can be found here:
```
https://hao-public-sharing-pdx.s3.us-west-2.amazonaws.com/demo-data/fine_tune.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASIN2BOQZ7HQSIDH6%2F20240812%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240812T044058Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=bc886621bb8e47c70bd23a3cee43e9a52eddad1d3f4c1ae7d56ee61eb29639d2
```


### Fine tune embedding model using Hopot QA training set

Create a triplet dataset by sampling 100000 Hopot QA questions. The support document chunks will be used as postive chunks and other context chunks will be used as negtive examples. You can find more details in the notebook: scripts/model_ft_and_index.ipynb

I choose to use 'Alibaba-NLP/gte-large-en-v1.5' as the base model by looking at the MTEB leadboard https://huggingface.co/spaces/mteb/leaderboard
Among all the top performance models gte-large-en-v1.5 only has 434M parameters which makes it easy to run and fine tune using small GPU host


Here is the eval result before and after fine tuning on 10000 testing triplets sampled from hotpot qa validation set

Before fine tuning:
```
{'hotpot-val-set_cosine_accuracy': 0.867,
 'hotpot-val-set_dot_accuracy': 0.1331,
 'hotpot-val-set_manhattan_accuracy': 0.8678,
 'hotpot-val-set_euclidean_accuracy': 0.867,
 'hotpot-val-set_max_accuracy': 0.8678}
```

After fine tuning:
```
{'hotpot-val-set_cosine_accuracy': 0.9619,
 'hotpot-val-set_dot_accuracy': 0.0386,
 'hotpot-val-set_manhattan_accuracy': 0.9614,
 'hotpot-val-set_euclidean_accuracy': 0.9615,
 'hotpot-val-set_max_accuracy': 0.9619}
```
As you can see the consine accuracy increase from 0.867 to 0.9619


### Create index
After fine tuning the embedding model we create a llama-index faiss index using 1000 queries form hotpot qa valiation dataset
For each query we keep the support documents plus 2 negtive documents from the context list.

We split documents in to chunks with 512 tokens and embed these chunks using the fine tuned model. The final corpus contain 22942 document chunks.

### QA
Because Hotpot QA questions require finding and reasoning over multiple supporting documents to answer, we use llama index's SubQuestionQueryEngine to split the original query into subquestions for document retrieval

For example:
For question: "Were Scott Derrickson and Ed Wood of the same nationality?"
It will query the index twice with the following sub queries. 
```
[wiki_search_engine] Q: What is the nationality of Scott Derrickson?
[wiki_search_engine] Q: What is the nationality of Ed Wood?
```


### Eval:
Run a quick eval on the 1000 validation queries and got 540 correct answers, which bring the accuracy to 54%

The result qualit is similar to https://contextual.ai/introducing-rag2/

Note that these is not apple to apple's comparison because I didn't index the whole wikipedia corpus due to resource limits.


You can run the search code follow this readme. Note that because we don't have 
