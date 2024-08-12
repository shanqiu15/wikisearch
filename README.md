# Wikipedia Search Engine
This README guides you through setting up and running a Wikipedia search engine that uses a fine-tuned embedding model and document index.


## How to run the code
### Step 1: Clone the repo and install dependencies

```
$ git clone ${repo}
$ cd ${repo}
$ python -m venv .env
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ pip install .
```

Step 2: Download the Fine-tuned Embedding Model and Document Index
From the root directory of the repository:

```
$ mkdir data
$ cd data
$ curl "https://hao-public-sharing-pdx.s3.us-west-2.amazonaws.com/demo-data/inference_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASIN2BOQZ7HQSIDH6%2F20240812%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240812T043220Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=293744ec7a77a5c11c0810ef3deaaeac0bdde0a107db1b703cd7fc285bf54e90" -o "inference_data.zip"
$ unzip inference_data.zip 
$ cd ..
```

Step 3: Set OpenAI API Key
To run the code in question-answer mode, set your OpenAI API key as an environment variable:
```
% export OPENAI_API_KEY= Your OpenAI Key
```
We use GPT-4o for answer generation. Running the code without setting the OpenAI key will return only the retrieved document chunks.

Step 4: Run the Search
```
% wikisearch "Were Scott Derrickson and Ed Wood of the same nationality?"
```



## How the Search Engine Was Created
### Download the Latest Wikipedia Dump
Get the latest Wikipedia dump from: https://dumps.wikimedia.org/enwiki/

Clone and set up dependencies for this repository: https://github.com/huggingface/olm-datasets

Then, use scripts/get_wikipedia_data.py to pull the latest Wikipedia dump. You can change the SNAPSHOT_DATE to download a different snapshot.
```
% cd scripts
% python get_wikipedia_data.py
```

### Download the Hotpot QA Dataset from Hugging Face

Here are more details about the dataset https://huggingface.co/datasets/hotpotqa/hotpot_qa

```
import datasets

retrieve_dataset = datasets.load_dataset("hotpotqa/hotpot_qa", 'distractor', trust_remote_code=True)
```

After downloading the dataset, remove all questions whose support documents don't exist in the new Wikipedia snapshot, and retrieve the original Wikipedia pages for each query.

The processed dataset and Wikipedia documents can be found here:
```
https://hao-public-sharing-pdx.s3.us-west-2.amazonaws.com/demo-data/fine_tune.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASIN2BOQZ7HQSIDH6%2F20240812%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240812T044058Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=bc886621bb8e47c70bd23a3cee43e9a52eddad1d3f4c1ae7d56ee61eb29639d2
```


### Fine-Tune the Embedding Model Using the Hotpot QA Training Set
Create a triplet dataset by sampling 100,000 Hotpot QA questions. The support document chunks will be used as positive chunks, and other context chunks will be used as negative examples. For more details, see the notebook: scripts/model_ft_and_index.ipynb.

We chose 'Alibaba-NLP/gte-large-en-v1.5' as the base model based on the MTEB leaderboard: MTEB Leaderboard (https://huggingface.co/spaces/mteb/leaderboard)

This model has 434M parameters, making it feasible to fine-tune on a small GPU host.

#### Evaluation Results
Here are the evaluation results before and after fine-tuning on 10,000 testing triplets sampled from the Hotpot QA validation set:

Before Fine-Tuning:
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

As you can see, the cosine accuracy increased from 0.867 to 0.9619.

### Create the Index
After fine-tuning the embedding model, we created a Llama-Index FAISS index using 1,000 queries from the Hotpot QA validation dataset. For each query, we kept the support documents plus 2 negative documents from the context list.

We split documents into chunks with 512 tokens and embedded these chunks using the fine-tuned model. The final corpus contains 22,942 document chunks.

### Question Answering (QA)
Hotpot QA questions often require finding and reasoning over multiple supporting documents. Therefore, we use Llama Index's SubQuestionQueryEngine to split the original query into sub-questions for document retrieval.

#### Example:
For question: "Were Scott Derrickson and Ed Wood of the same nationality?"
It will query the index twice with the following sub queries. 
```
[wiki_search_engine] Q: What is the nationality of Scott Derrickson?
[wiki_search_engine] Q: What is the nationality of Ed Wood?
```


### Evaluation
We ran a quick evaluation on 1,000 validation queries and achieved 540 correct answers, resulting in an accuracy of 54%.

The quality of the results is similar to what is presented in Contextual AI's RAG2 (https://contextual.ai/introducing-rag2/). Note that this comparison isn't entirely fair because we didn't index the entire Wikipedia corpus due to resource limitations.

Here are some example queries you can try:
```
 'Were Scott Derrickson and Ed Wood of the same nationality?',
 'Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?',
 'The director of the romantic comedy "Big Stone Gap" is based in what New York city?',
 '2014 S/S is the debut album of a South Korean boy group that was formed by who?',
 'Who is older, Annie Morton or Terry Richardson?',
 'Are Local H and For Against both from the United States?',
 'The football manager who recruited David Beckham managed Manchester United during what timeframe?',
 'Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?',
 'Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what years?',
 'Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?',
 'Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? ',
 'What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?',
 "The 2011–12 VCU Rams men's basketball team, led by third year head coach Shaka Smart, represented Virginia Commonwealth University which was founded in what year?",
 'Are both Dictyosperma, and Huernia described as a genus?',
 'Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?',
 'Are Freakonomics and In the Realm of the Hackers both American documentaries?',
 'Which band, Letters to Cleo or Screaming Trees, had more members?',
 'Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended when ?',
 'Are both Elko Regional Airport and Gerald R. Ford International Airport located in Michigan?',
 'Ralph Hefferline was a psychology professor at a university that is located in what city?',
 'Alfred Balk served as the secretary of the Committee on the Employment of Minority Groups in the News Media under which United States Vice President?',
 'Who is the writer of this song that was inspired by words on a tombstone and was the first track on the box set Back to Mono?',
 'Are Ferocactus and Silene both types of plant?',
 'Which British first-generation jet-powered medium bomber was used in the South West Pacific theatre of World War II?',
 'What race track in the midwest hosts a 500 mile race eavery May?',
 'D1NZ is a series based on what oversteering technique?',
 'who is younger Keith Bostic or Jerry Glanville ?',
 'Are both Cypress and Ajuga genera?',
 'What was the Roud Folk Song Index of the nursery rhyme inspiring What Are Little Girls Made Of?',
 'What WB supernatrual drama series was Jawbreaker star Rose Mcgowan best known for being in?',
 'Vince Phillips held a junior welterweight title by an organization recognized by what larger Hall of Fame?',
 'The 2017–18 Wigan Athletic F.C. season will be a year in which the team competes in the league cup known as what for sponsorship reasons?',
 'What color clothing do people of the Netherlands wear during Oranjegekte or to celebrate the national holiday Koningsdag? ',
 'What was the name of the 1996 loose adaptation of William Shakespeare\'s "Romeo & Juliet" written by James Gunn?',
 'Robert Suettinger was the national intelligence officer under which former Governor of Arkansas?',
 'What nationality were social anthropologists Alfred Gell and Edmund Leach?',
 'In which year was the King who made the 1925 Birthday Honours born?',
 'What is the county seat of the county where East Lempster, New Hampshire is located?',
 'The Album Against the Wind was the 11th Album of a Rock singer Robert C Seger born may 6 1945. What was the Rock singers stage name ?',
 'What was the name of a woman from the book titled "Their Lives: The Women Targeted by the Clinton Machine " and was also a former white house intern?',
 'In what year was the novel that Lourenço Mutarelli based "Nina" on based first published?',
 'Where are Teide National Park and Garajonay National Park located?',
 "How many copies of Roald Dahl's variation on a popular anecdote sold?",
 'What occupation do Chris Menges and Aram Avakian share?',
 'Andrew Jaspan was the co-founder of what not-for-profit media outlet?',
```
