__author__ = "Christopher Potts and Omar Khattab"
__version__ = "CS224u, Stanford, Spring 2022"

import torch
 
if torch.cuda.is_available():
    !pip uninstall cupy-cuda11x -y
    !pip install cupy-cuda113

import torch

if torch.cuda.is_available():
    !nvcc --version

import collections
from contextlib import nullcontext
from collections import namedtuple
from datasets import load_dataset
import json
import numpy as np
import random
import re 
import string
import torch
from typing import List
from scipy.special import softmax
from tqdm.notebook import tqdm
import pandas as pd

seed = 1

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch

if torch.cuda.is_available():
    !pip install faiss-gpu==1.7.0
else:
    !pip install faiss-cpu==1.7.0

import openai
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

transformers.logging.set_verbosity_error()

import os
import sys
sys.path.insert(0, 'ColBERT/')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
from colbert.searcher import Searcher

def _find_generated_answer(tokens, newline="\n" ): 
    """Our LMs tend to insert initial newline characters before
    they begin generating text. This function ensures that we 
    properly capture the true first line as the answer while
    also ensuring that token probabilities are aligned."""        
    answer_token_indices = []
    char_seen = False            
    for i, tok in enumerate(tokens):
        # This is the main condition: a newline that isn't an initial
        # string of newlines:
        if tok == newline and char_seen:
            break
        # Keep the initial newlines for consistency:
        elif tok == newline and not char_seen:
            answer_token_indices.append(i)
        # Proper tokens:
        elif tok != newline:
            char_seen = True
            answer_token_indices.append(i)
    return answer_token_indices 

# "gpt-neo-125m" "gpt-neo-1.3B" "gpt-neo-2.7B" "gpt-j-6B"
eleuther_model_name = "gpt-neo-125m"
# EleutherAI/gpt-neo-125m

eleuther_tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/{eleuther_model_name}", 
    padding_side="left", 
    padding='longest', 
    truncation='longest_first', max_length=2000)
eleuther_tokenizer.pad_token = eleuther_tokenizer.eos_token

eleuther_model = AutoModelForCausalLM.from_pretrained(
    f"EleutherAI/{eleuther_model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
eleuther_model = eleuther_model.to(device)


def run_eleuther(prompts, temperature=0.1, top_p=0.95, **generate_kwargs): 
    """
    Parameters
    ----------
    prompts : iterable of str
    temperature : float
        It seems best to set it low for this task!
    top_p : float
       
    For options for `generate_kwargs`, see:
    
    https://huggingface.co/docs/transformers/master/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
    
    Options that are likely to be especially relevant include 
    `temperature`, `length_penalty`, and the parameters that
    determine the decoding strategy. With `num_return_sequences > 1`,
    the default parameters in this function do multinomial sampling.
    
    Returns
    -------
    list of dicts
    
    {"prompt": str, 
     "generated_text": str, "generated_tokens": list of str, "generated_probs": list of float,
     "answer": str, "answer_tokens": list of str, "answer_probs": list of float
    }
         
    """
    prompt_ids = eleuther_tokenizer(
        prompts, return_tensors="pt", padding=True).input_ids.to(device)
        
    with torch.inference_mode():
        # Automatic mixed precision if possible.
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            model_output = eleuther_model.generate(
                prompt_ids,
                temperature=temperature, # https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/generation/logits_process.py#L166
                do_sample=True,
                top_p=top_p,           # https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/generation/utils.py#L3912
                max_new_tokens=16,
                num_return_sequences=1, 
                pad_token_id=eleuther_tokenizer.eos_token_id, 
                return_dict_in_generate=True,
                output_scores=True,
                **generate_kwargs)
        
    # Converting output scores using the helpful recipe here:
    # https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
    gen_ids = model_output.sequences[:, prompt_ids.shape[-1] :]
    gen_probs = torch.stack(model_output.scores, dim=1).softmax(-1)
    gen_probs = torch.gather(gen_probs, 2, gen_ids[:, :, None]).squeeze(-1)
    
    # Generated texts, including the prompts:
    gen_texts = eleuther_tokenizer.batch_decode(
        model_output.sequences, skip_special_tokens=True)
    
    data = []     
    iterator = zip(prompts, gen_ids, gen_texts, gen_probs)    
    for prompt, gen_id, gen_text, gen_prob in iterator:       
        gen_tokens = eleuther_tokenizer.convert_ids_to_tokens(gen_id)
        generated_text = gen_text[len(prompt): ]
        gen_prob = [float(x) for x in gen_prob.cpu().numpy()] # float for JSON storage
        ans_indices = _find_generated_answer(gen_tokens, newline="Ċ")
        answer_tokens = [gen_tokens[i] for i in ans_indices]
        answer_probs = [gen_prob[i] for i in ans_indices]
        answer = "".join(answer_tokens).replace("Ġ", " ").replace("Ċ", "\n")                                       
        data.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "generated_tokens": gen_tokens,
            "generated_probs": gen_prob,
            "generated_answer": answer,
            "generated_answer_probs": answer_probs,
            "generated_answer_tokens": answer_tokens})                        

    return data

eleuther_ex = run_eleuther([    
    "What year was Stanford University founded?", 
    "In which year did Stanford first enroll students?"])

eleuther_ex


def run_gpt3(prompts, engine="text-curie-001", temperature=0.1, top_p=0.95, **gpt3_kwargs):
    """To use this function, sign up for an OpenAI account at
        
    https://beta.openai.com/signup
    
    That should give you $18 in free credits, which is more than enough
    for this assignment assuming you are careful with testing.
    
    Once your account is set up, you can get your API key from your 
    account dashboard and paste it in below as the value of 
    `openai.api_key`.
    
    Parameters
    ----------
    prompts : iterable of str
    engine : str
        This has to be one of the models whose name begins with "text".
        The "instruct" class of models can't be used, since they seem
        to depend on some kinds of QA-relevant supervision.        
        For options, costs, and other details: 
        https://beta.openai.com/docs/engines/gpt-3                
    temperature : float
        It seems best to set it low for this task!
    top_p : float
        
    For information about values for `gpt3_kwargs`, see
    
    https://beta.openai.com/docs/api-reference/completions
    
    Returns
    -------
    list of dicts   
    
    """
    # Fill this in with the value from your OpenAI account. First
    # verify that your account is set up with a spending limit that
    # you are comfortable with. If you just opened your account,
    # you should have $18 in credit and so won't need to supply any
    # payment information.
    openai.api_key = None
    
    
    assert engine.startswith("text"), \
        "Please use an engine whose name begins with 'text'."
        
    response = openai.Completion.create(
        engine=engine,       
        prompt=prompts,
        temperature=temperature,
        top_p=top_p,
        echo=False,   # This function will not work
        logprobs=1,   # properly if any of these
        n=1,          # are changed!
        **gpt3_kwargs)
    
    # From here, we parse each example to get the values
    # we need:
    data = []
    for ex, prompt in zip(response["choices"], prompts):
        tokens = ex["logprobs"]["tokens"]
        logprobs = ex["logprobs"]["token_logprobs"]        
        probs = list(np.exp(logprobs))
        if "<|endoftext|>" in tokens:
            end_i = tokens.index("<|endoftext|>")
            tokens = tokens[ : end_i]  # This leaves off the "<|endoftext|>"
            probs = probs[ : end_i]    # token -- perhaps dubious.
        ans_indices = _find_generated_answer(tokens)
        answer_tokens = [tokens[i] for i in ans_indices]
        answer_probs = [probs[i] for i in ans_indices]
        answer = "".join(answer_tokens)        
        data.append({
            "prompt": prompt,
            "generated_text": ex["text"],
            "generated_tokens": tokens,
            "generated_probs": probs,
            "generated_answer": answer,
            "generated_answer_tokens": answer_tokens,
            "generated_answer_probs": answer_probs})
        
    return data             

squad = load_dataset("squad")

SquadExample = namedtuple("SquadExample",  "id title context question answers")

def get_squad_split(squad, split="validation"):
    """
    Use `split='train'` for the train split.
    
    Returns
    -------
    list of SquadExample named tuples with attributes
    id, title, context, question, answers
    
    """    
    fields = squad[split].features
    data = zip(*[squad[split][field] for field in fields])
    return [SquadExample(eid, title, context, question, answers["text"]) 
            for eid, title, context, question, answers in data]

squad_dev = get_squad_split(squad)

squad_dev[0]

dev_exs = sorted(squad_dev, key=lambda x: hash(x.id))[: 200]

squad_train = get_squad_split(squad, split="train")

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Normalize string and split string into tokens."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Compute the Exact Match score."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1_from_tokens(gold_toks: List[str], pred_toks: List[str]) -> float:
    """Compute the F1 score from tokenized gold answer and prediction."""
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute the F1 score."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    return compute_f1_from_tokens(gold_toks, pred_toks)

def evaluate(examples, prompts, gens):
    """Generic evalution function.
    
    Parameters
    ----------
    examples: iterable of `SquadExample` instances
    prompts: list of str
    preds: list of LM-generated texts to evaluate as answers
    
    Returns
    -------
    dict with keys "em_per", "macro_f1", "examples", where
    each "examples" value is a dict
    
    """        
    results = []
    for ex, prompt, gen in zip(examples, prompts, gens):
        answers = ex.answers
        pred = gen['generated_answer']
        # The result is the highest EM from the available answer strings:
        em = max([compute_exact(ans, pred) for ans in answers])
        f1 = max([compute_f1(ans, pred) for ans in answers])
        gen.update({
            "id": ex.id, 
            "question": ex.question, 
            "prediction": pred, 
            "answers": answers, 
            "em": em,
            "f1": f1
        })
        results.append(gen)
    data = {}        
    data["macro_f1"] = np.mean([d['f1'] for d in results])
    data["em_per"] = sum([d['em'] for d in results]) / len(results)
    data["examples"] = results
    return data

ex = namedtuple("SquadExample",  "id title context question answers")

examples = [
    ex("0", "CS224u", 
       "The course to take is NLU!", 
       "What is the course to take?", 
       ["NLU", "CS224u"])]

prompts = ["Dear model, Please answer this question!\n\nQ: What is the course to take?\n\nA:"]

gens = [{"generated_answer": "NLU", "generated_text": "NLU\nWho am I?"}]

evaluate(examples, prompts, gens)

def evaluate_no_context(examples, gen_func=run_eleuther, batch_size=20):
    prompts = [] 
    gens = []
    for i in range(0, len(examples), batch_size):
        ps = [ex.question for ex in examples[i: i+batch_size]]
        gs = gen_func(ps)        
        prompts += ps
        gens += gs    
    return evaluate(examples, prompts, gens)    

%%time
nocontext_results = evaluate_no_context(dev_exs)

print(nocontext_results['macro_f1'])

def build_few_shot_qa_prompt(ex, squad_train, n_context=2, joiner="\n\n"):
    segs = []
    train_exs = random.sample(squad_train, k=n_context)    
    for t in train_exs:
        segs += [
            f"Title: {t.title}",
            f"Background: {t.context}",
            f"Q: {t.question}",
            f"A: {t.answers[0]}"
        ]
    segs += [
        f"Title: {ex.title}",
        f"Background: {ex.context}",
        f"Q: {ex.question}",
        f"A:"
    ]
    return joiner.join(segs)                

print(build_few_shot_qa_prompt(dev_exs[0], squad_train, n_context=1))

def evaluate_few_shot_qa(examples, squad_train, gen_func=run_eleuther, batch_size=20, n_context=2):
    prompts = []
    gens = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i: i+batch_size]
        ps = [build_few_shot_qa_prompt(ex, squad_train, n_context=n_context) for ex in batch]        
        gs = gen_func(ps)       
        prompts += ps
        gens += gs
    return evaluate(examples, prompts, gens)

%%time
few_shot_qa_results = evaluate_few_shot_qa(dev_exs, squad_train, n_context=1)

print(few_shot_qa_results['macro_f1'])

index_home = os.path.join("experiments", "notebook", "indexes")

if not os.path.exists(os.path.join("data", "openqa", "colbertv2.0.tar.gz")):
    !mkdir -p data/openqa
    # ColBERTv2 checkpoint trained on MS MARCO Passage Ranking (388MB compressed)
    !wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -P data/openqa/
    !tar -xvzf data/openqa/colbertv2.0.tar.gz -C data/openqa/

if not os.path.exists(os.path.join(index_home, "cs224u.collection.2bits.tgz")):
    !wget https://web.stanford.edu/class/cs224u/data/cs224u.collection.2bits.tgz -P experiments/notebook/indexes
    !tar -xvzf experiments/notebook/indexes/cs224u.collection.2bits.tgz -C experiments/notebook/indexes

collection = os.path.join(index_home, "cs224u.collection.2bits", "cs224u.collection.tsv")

collection = Collection(path=collection)

f'Loaded {len(collection):,} passages'

index_name = "cs224u.collection.2bits"

with Run().context(RunConfig(experiment='notebook')):
    searcher = Searcher(index=index_name)

query = "linguistics"

print(f"#> {query}")

# Find the top-3 passages for this query
results = searcher.search(query, k=3) 

# Print out the top-k retrieved passages
for passage_id, passage_rank, passage_score in zip(*results):
    print(f"\t[{passage_rank}]\t{passage_score:.1f}\t {searcher.collection[passage_id]}")

from utility.utils.dpr import has_answer, DPR_normalize

def success_at_k(examples, k=20):
    scores = []
    for ex in examples: 
        scores.append(evaluate_retrieval_example(ex, k=k))
    return sum(scores) / len(scores)
        
    
def evaluate_retrieval_example(ex, k=20):    
    results = searcher.search(ex.question, k=k)
    for passage_id, passage_rank, passage_score in zip(*results):
        passage = searcher.collection[passage_id]
        score = has_answer([DPR_normalize(ans) for ans in ex.answers], passage)
        if score:
            return 1
    return 0

def build_zero_shot_openqa_prompt(question, passage, joiner="\n\n"):
    title, context = passage.split(" | ", 1)
    segs = [
        f"Title: {title}",
        f"Background: {context}",
        f"Q: {question}",
        "A:"
    ]
    return joiner.join(segs)    

r = searcher.search("Walking in the street", k=3)
[searcher.collection[idx] for idx in r[0]] 

def evaluate_zero_shot_openqa(examples, joiner="\n\n", gen_func=run_eleuther, batch_size=20):
    prompts = []
    gens = []
    for i in range(0, len(examples), batch_size):
        exs = examples[i: i+batch_size]
        results = [searcher.search(ex.question, k=1) for ex in exs]
        passages = [searcher.collection[r[0][0]] for r in results]
        ps = [build_zero_shot_openqa_prompt(ex.question, psg, joiner=joiner) 
              for ex, psg in zip(exs, passages)]
        gs = gen_func(ps)       
        prompts += ps
        gens += gs
    return evaluate(examples, prompts, gens)

%%time
zero_shot_openqa_results = evaluate_zero_shot_openqa(dev_exs)

print(zero_shot_openqa_results['macro_f1'])

def build_few_shot_no_context_prompt(question, train_exs, joiner="\n\n"):
    """No context few-shot OpenQA prompts.

    Parameters
    ----------
    question : str   
    train_exs : iterable of SQuAD train examples. These can be 
        obtained via a random sample 
        from `squad_train` as defined above.
    joiner : str
        The character to use to join pieces of the prompt into 
        a single str.

    Returns
    -------
    str, the prompt

    """
   
    segs = []
    for t in train_exs:
        segs += [
            f"Q: {t.question}",
            f"A: {t.answers[0]}"
        ]
    
    segs += [
        f"Q: {question}",
        f"A:"
    ]
    
    return joiner.join(segs) 

def test_build_few_shot_no_context_prompt(func):
    train_exs = [
        SquadExample(0, "T1", "Q1", "C1", ["A1"]),
        SquadExample(1, "T2", "Q2", "C2", ["A2"]),
        SquadExample(2, "T3", "Q3", "C3", ["A3"])]
    question = "My Q"
    result = func(question, train_exs, joiner="\n")
    expected = ""
    tests = [
        (1, "\n", 'Q: C1\nA: A1\nQ: My Q\nA:'),                
        (1, "\n\n", 'Q: C1\n\nA: A1\n\nQ: My Q\n\nA:'),
        (2, "\n", 'Q: C1\nA: A1\nQ: C2\nA: A2\nQ: My Q\nA:')]
    err_count = 0       
    for n_context, joiner, expected in tests:
        result = func(question, train_exs[: n_context], joiner=joiner)
        if result != expected:
            err_count +=1 
            print(f"Error:\n\nExpected:\n\n{expected}\n\nGot:\n\n{result}")    
    if err_count == 0:
        print("No errors detected in `build_few_shot_no_context_prompt`")     

test_build_few_shot_no_context_prompt(build_few_shot_no_context_prompt)

def evaluate_few_shot_no_context(
        examples,
        squad_train,
        batch_size=20,
        n_context=2,
        joiner="\n\n",
        gen_func=run_eleuther):
    """Evaluate a few-shot OpenQA with no context approach 
    defined by `build_few_shot_no_context_prompt` and `gen_func`.

    Parameters
    ----------
    examples : iterable of SQuAD train examples
        Presumably a subset of `squad_dev` as defined above.
    squad_train : iterable of SQuAD train examples
    batch_size : int
        Number of examples to send to `gen_func` at once.
    n_context : n
        Number of examples to use from `squad_train`.
    joiner : str
        Used by `build_few_shot_open_qa_prompt` to join segments
        of the prompt into a single str.
    gen_func : either `run_eleuther` or `run_gpt3`

    Returns
    -------
    dict as determined by `evaluate` above.

    """
    # A list of strings that you build and feed into `gen_func`.
    prompts = []

    # A list of dicts that you get from `gen_func`.
    gens = []

    # Iterate through the examples in batches:
    for i in range(0, len(examples), batch_size):
        # Sample some SQuAD training examples to use with
        # `build_few_shot_no_context_prompt` and `ex.question`,
        # run the resulting prompt through `gen_func`, and
        # add your prompts and results to `prompts` and `gens`.

        batch = examples[i: i+batch_size]
        ps = [build_few_shot_no_context_prompt(ex.question, 
                                               random.sample(squad_train, k=n_context),
                                               joiner=joiner
                                              ) for ex in batch]
        gs = gen_func(ps)
        prompts += ps
        gens += gs
    # Return value from a call to `evalaute`, with `examples`
    # as provided by the user and the `prompts` and `gens`
    # you built:
    return evaluate(examples, prompts, gens)

def test_evaluator(func):
    examples = [SquadExample(0, "T1", "Q1", "C1", ["A1"])]    
    squad_train = [SquadExample(0, "sT1", "sQ1", "sC1", ["sA1"])] 
    
    def gen_func(*prompts):
        return [{
            "generated_answer": "Constant output", 
            "generated_answer_tokens": ["Constant", "output"], 
            "generated_answer_probs": [0.1, 0.2]}]
    
    batch_size = 1    
    n_context = 1    
    joiner = "\n"
    result = func(
        examples, 
        squad_train, 
        batch_size=1, 
        n_context=1, 
        joiner=joiner, 
        gen_func=gen_func)
    expected_keys = {'em_per', 'examples', 'macro_f1'}
    result_keys = set(result.keys())     
    if expected_keys != result_keys:
        print(f"Unexpected keys in result. "
              f"Expected: {expected_keys}; Got: {result_keys}")
        return
    expected_ex_keys = {
        'f1', 'id', 'em', 'generated_answer_tokens', 'generated_answer_probs',
        'prediction', 'generated_answer', 'question', 'answers'}
    result_ex_keys = set(result["examples"][0].keys())
    if expected_ex_keys != result_ex_keys:
        print(f"Unexpected keys in result['examples']. "
              f"Expected: {expected_ex_keys}; Got: {result_ex_keys}")
        return
    print("No errors detected in `evaluate_few_shot_open_qa`")  

def build_few_shot_open_qa_prompt(question, passage, train_exs, joiner="\n\n"):
    """Few-shot OpenQA prompts.

    Parameters
    ----------
    question : str
    passage : str
        Presumably something retrieved via search.
    train_exs : iterable of SQuAD train examples
        These can be obtained via a random sample from 
        `squad_train` as defined above.
    joiner : str
        The character to use to join pieces of the prompt 
        into a single str.

    Returns
    -------
    str, the prompt

    """

    segs = []
    for t in train_exs:
        segs += [
            f"Title: {t.title}",
            f"Background: {t.context}",
            f"Q: {t.question}",
            f"A: {t.answers[0]}"
        ]

    my_title, my_background = passage.split(" | ")
    
    segs += [
            f"Title: {my_title}",
            f"Background: {my_background}",
            f"Q: {question}",
            f"A:"
        ]
    return joiner.join(segs)  

def test_build_few_shot_open_qa_prompt(func):
    train_exs = [
        SquadExample(0, "T1", "Q1", "C1", ["A1"]),
        SquadExample(1, "T2", "Q2", "C2", ["A2"]),
        SquadExample(2, "T3", "Q3", "C3", ["A3"])]            
    question = "My Q"    
    passage = "Title | target passage"    
    tests = [
        (1, "\n", ('Title: T1\nBackground: Q1\nQ: C1\nA: A1\n'
                   'Title: Title\nBackground: target passage\nQ: My Q\nA:')),
        (1, "\n\n", ('Title: T1\n\nBackground: Q1\n\nQ: C1\n\nA: A1\n\n'
                     'Title: Title\n\nBackground: target passage\n\nQ: My Q\n\nA:')),
        (2, "\n", ('Title: T1\nBackground: Q1\nQ: C1\nA: A1\nTitle: T2\n'
                   'Background: Q2\nQ: C2\nA: A2\nTitle: Title\n'
                   'Background: target passage\nQ: My Q\nA:'))]
    err_count = 0       
    for n_context, joiner, expected in tests:
        result = func(question, passage, train_exs[: n_context], joiner=joiner)
        if result != expected:
            err_count +=1 
            print(f"Error:\n\nExpected:\n\n{expected}\n\nGot:\n\n{result}")    
    if err_count == 0:
        print("No errors detected in `build_few_shot_open_qa_prompt`")

test_build_few_shot_open_qa_prompt(build_few_shot_open_qa_prompt)

def evaluate_few_shot_open_qa(
        examples,
        squad_train,
        batch_size=20,
        n_context=2,
        joiner="\n\n",
        gen_func=run_eleuther):
    """Evaluate a few-shot OpenQA approach defined by 
    `build_few_shot_open_qa_prompt` and `gen_func`.

    Parameters
    ----------
    examples : iterable of SQuAD train examples
        Presumably a subset of `squad_dev` as defined above.
    squad_train : iterable of SQuAD train examples
    batch_size : int
        Number of examples to send to `gen_func` at once.
    joiner : str
        Used by `build_few_shot_open_qa_prompt` to join segments
        of the prompt into a single str.
    gen_func : either `run_eleuther` or `run_gpt3`

    Returns
    -------
    dict as determined by `evaluate` above.

    """
    # A list of strings that you build and feed into `gen_func`.
    prompts = []

    # A list of dicts that you get from `gen_func`.
    gens = []

    # Iterate through the examples in batches:
    for i in range(0, len(examples), batch_size):
        # Use the `searcher` defined above to get passages
        # using `ex.question` as the query, and use your
        # `build_few_shot_open_qa_prompt` to build prompts.


        exs = examples[i: i+batch_size]
        results = [searcher.search(ex.question, k=1) for ex in exs]
        passages = [searcher.collection[r[0][0]] for r in results]
        ps = [build_few_shot_open_qa_prompt(ex.question, 
                                            psg, 
                                            random.sample(squad_train, k=n_context),
                                            joiner=joiner) 
              for ex, psg in zip(exs, passages)]
        gs = gen_func(ps)  
        prompts += ps
        gens += gs
    # Return value from a call to `evalaute`, with `examples`
    # as provided by the user and the `prompts` and `gens`
    # you built:
    return evaluate(examples, prompts, gens)

test_evaluator(evaluate_few_shot_open_qa)

def get_passages_with_scores(question, k=5):
    """Pseudo-probabilities from the retriever.

    Parameters
    ----------
    question : str
    k : int
        Number of passages to retrieve.

    Returns
    -------
    passages (list of str), passage_probs (np.array)

    """
    # Use the `searcher` to get `k` passages for `questions`:

    results = searcher.search(question, k=k)
    


    # Softmax normalize the scores and convert the list to
    # a NumPy array:

    normalized_scores = softmax(results[2])


    # Get the passages as a list of texts:

    passages = [searcher.collection[r] for r in results[0]]
    
    return passages, normalized_scores

def test_get_passages_with_scores(func):
    question = "What is linguistics?"        
    passages, passage_probs = get_passages_with_scores(question, k=2)    
    if len(passages) != len(passage_probs):
        print("`get_passages_with_scores` should return equal length "
              "lists of passages and passage probabilities.")
        return
    if len(passages) != 2:
        print(f"`get_passages_with_scores` should return `k` passages. Yours returns {len(passages)}")
        return
    if not all(isinstance(psg, str) for psg in passages):
        print("The first return argument should be a list of passage strings.")
        return
    if not all(isinstance(p, (float, np.float32, np.float64)) for p in passage_probs): 
        print("The second return argument should be a list of floats.")
        return 
    print("No errors detected in `get_passages_with_scores`")

test_get_passages_with_scores(get_passages_with_scores)

def answer_scoring(passages, passage_probs, prompts, gen_func=run_eleuther):
    """Implements our basic scoring strategy.

    Parameters
    ----------
    passages : list of str
    passage_probs : list of float
    prompts : list of str
    gen_func : either `run_eleuther` or `run_gpt3`

    Returns
    -------
    list of pairs (score, dict), sorted with the largest score first.
    `dict` should be the return value of `gen_func` for an example.

    """
    data = []
    for passage, passage_prob, prompt in zip(passages, passage_probs, prompts):
        # Run `gen_func` on [prompt] (crucially, the singleton list here),
        # and get the dictionary `gen` from the singleton list `gen_func`
        # returns, and then use the values to score `gen` according to our
        # scoring method.
        #
        # Be sure to use "generated_answer_probs" for the scores.

        gs = gen_func([prompt])[0]
        data.append([np.product(gs["generated_answer_probs"]) * passage_prob, gs])

    # Return `data`, sorted with the highest scoring `(score, gen)`
    # pair given first.

    data = sorted(data, key=lambda x: -x[0])
    return data

def test_answer_scoring(func):
    passages = [
        "Pragmatics is the study of language use.", 
        "Phonology is the study of linguistic sound systems."]
    passage_probs = [0.75, 0.25]
    prompts = passages
    
    def gen_func(*prompts):
        return [{
            "generated_answer": "Constant output", 
            "generated_answer_tokens": ["Constant", "output"], 
            "generated_answer_probs": [0.1, 0.2]}]
    
    data = func(passages, passage_probs, prompts, gen_func=gen_func)

    if not all(len(x) == 2 for x in data):
        print("`answer_scoring` should return a list of pairs (score, gen)")
        return 
    if not isinstance(data[0][0], (float, np.float32, np.float64)):
        print("The first member of each pair in `data` should be a score (type `float`).")
        return    
    if not isinstance(data[0][1], dict):
        print("The second member of each pair in `data` should be a dict " 
              "created by running `gen_func` on a single example.")
        return    
    if data[0][0] != max([x for x, y in data]):
        print("`answer_scoring` should sort its data with the highest score first.")
        return 
    
    print("No errors detected in `answer_scoring`")

test_answer_scoring(answer_scoring)

def answer_scoring_demo(question):
    """Example usage for answer_scoring. Here we extract the top-scoring
    results, which can then be used in an evaluation."""    
    passages, passage_probs = get_passages_with_scores(question)
    prompts = [build_zero_shot_openqa_prompt(question, psg) for psg in passages]
    data = answer_scoring(passages, passage_probs, prompts)
    # Top-scoring answer string:
    return data[0][1]

answer_scoring_demo("How long is Moby Dick?")

# PLEASE MAKE SURE TO INCLUDE THE FOLLOWING BETWEEN THE START AND STOP COMMENTS:
#   1) Textual description of your system.
#   2) The code for your original system.
#   3) The score achieved by your system in place of MY_NUMBER.
#        With no other changes to that line.
#        You should report your score as a decimal value <=1.0
# PLEASE MAKE SURE NOT TO DELETE OR EDIT THE START AND STOP COMMENTS

# NOTE: MODULES, CODE AND DATASETS REQUIRED FOR YOUR ORIGINAL SYSTEM
# SHOULD BE ADDED BELOW THE 'IS_GRADESCOPE_ENV' CHECK CONDITION. DOING
# SO ABOVE THE CHECK MAY CAUSE THE AUTOGRADER TO FAIL.

# START COMMENT: Following are the key considerations when designing the original system:
"""
1. Different values of the temperature and num_beams hyperparameters are evaluated on the SQuAD dev dataset to determine the value 
    to use based on macro_f1 score.
2. For every question, two most relevant contexts are determined based on ColBERT index by using the `get_passages_with_scores` functions.
    Normalized scores are calculated for each retrived passage and are used as the passage probability given the question. This step could
    be improvised further based on the discussion above by using a languge model to determine the passage probability.
3. For every question/context pair find the two most relevant SQuAD example are found from the SQuAD train dataset based on BM25 for 
    doing few-shot learning. The index of the BM25 model is setup by concating the context and question strings of the SQuAD dataset.
4. When scoring the answers for each passage retrived in step 3, a normalization factor is used so that long answers are not excessively panelized.
5. A product of passage probability (step 2) and answer given context is used to score different answers.

Note - Due to compute limitations `gpt-neo-125m` is used in this assignment and grid search is is performed on temperature and num_beams parameters
    for a very limited choices. Ideally one could use a bigger model, perform hyperpameter tuning on other model parameters, sample more contexts,
    do few-shot learning learning with more q/a pairs with the SQuAD training data with access to more compute resources.
"""
# My peak score was: 0.143387
if 'IS_GRADESCOPE_ENV' not in os.environ:
    from rank_bm25 import BM25Okapi
    
    class squad_bm25():
        """Class to find relevant quetion from SQuAD data set for few-shot learning using BM25.
        
        First a BM25 index is developed based on based on context + question strings for each example SQuAD train dataset.
        Then for each query, closest k SQuAD training examples are determined and returned.
        """
        def __init__(self, squad_data, tokenizer):
            self.squad_data = squad_data
            self.tokenizer = tokenizer
            self.build_index()

        def build_index(self):
            corpus = [ex.context+ " " + ex.question for ex in self.squad_data]
            tokenized_corpus = [self.tokenizer(item) for item in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

        def select_relevant_docs(self, query, k):
            tokenized_query = self.tokenizer(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            indices = np.argpartition(doc_scores, -k)[-k:]

            return [self.squad_data[ind] for ind in indices]
    
    # Develop the BM25 index based on Squad train dataset. 
    SQUAD_INDEX = squad_bm25(squad_train, get_tokens)
    
    
    def build_few_shot_open_qa_prompt(question, passage, squad_index=SQUAD_INDEX, k=2, joiner="\n\n"):
        """Few-shot OpenQA prompts.

        Parameters
        ----------
        question : str
        passage : str
            Presumably something retrieved via search.
        squad_index : Index for SQuAD training examples.
        k: Number of question and answer pairs to identify for doing few-shot learning.
        joiner : str
            The character to use to join pieces of the prompt into a single str.

        Returns
        -------
        str, the prompt

        """

        train_exs = squad_index.select_relevant_docs(passage + " " + query, k)
        segs = []
        for t in train_exs:
            segs += [
                f"Title: {t.title}",
                f"Background: {t.context}",
                f"Q: {t.question}",
                f"A: {t.answers[0]}"
            ]

        my_title, my_background = passage.split(" | ")

        segs += [
                f"Title: {my_title}",
                f"Background: {my_background}",
                f"Q: {question}",
                f"A:"
            ]
        return joiner.join(segs) 
    

    
    def run_eleuther_v2(prompts, **generate_kwargs): 
        """ Very similar to run_eleuther provided in the beginning of the notebook. 
        
        The inputs related to temperature and top_p are passed as kwargs so that the same function can be used to experiment with num_beams.
            
        Parameters
        ----------
        prompts : iterable of str

        For options for `generate_kwargs`, see:

        https://huggingface.co/docs/transformers/master/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate

        Options that are likely to be especially relevant include 
        `temperature`, `length_penalty`, and the parameters that
        determine the decoding strategy. With `num_return_sequences > 1`,
        the default parameters in this function do multinomial sampling.

        Returns
        -------
        list of dicts

        {"prompt": str, 
         "generated_text": str, "generated_tokens": list of str, "generated_probs": list of float,
         "answer": str, "answer_tokens": list of str, "answer_probs": list of float
        }

        """
        prompt_ids = eleuther_tokenizer(
            prompts, return_tensors="pt", padding=True).input_ids.to(device)

        with torch.inference_mode():
            # Automatic mixed precision if possible.
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
                model_output = eleuther_model.generate(
                    prompt_ids,
                    max_new_tokens=16,
                    num_return_sequences=1, 
                    pad_token_id=eleuther_tokenizer.eos_token_id, 
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generate_kwargs)

        # Converting output scores using the helpful recipe here:
        # https://discuss.huggingface.co/t/generation-probabilities-how-to-compute-probabilities-of-output-scores-for-gpt2/3175
        gen_ids = model_output.sequences[:, prompt_ids.shape[-1] :]
        gen_probs = torch.stack(model_output.scores, dim=1).softmax(-1)
        gen_probs = torch.gather(gen_probs, 2, gen_ids[:, :, None]).squeeze(-1)

        # Generated texts, including the prompts:
        gen_texts = eleuther_tokenizer.batch_decode(
            model_output.sequences, skip_special_tokens=True)

        data = []     
        iterator = zip(prompts, gen_ids, gen_texts, gen_probs)    
        for prompt, gen_id, gen_text, gen_prob in iterator:       
            gen_tokens = eleuther_tokenizer.convert_ids_to_tokens(gen_id)
            generated_text = gen_text[len(prompt): ]
            gen_prob = [float(x) for x in gen_prob.cpu().numpy()] # float for JSON storage
            ans_indices = _find_generated_answer(gen_tokens, newline="Ċ")
            answer_tokens = [gen_tokens[i] for i in ans_indices]
            answer_probs = [gen_prob[i] for i in ans_indices]
            answer = "".join(answer_tokens).replace("Ġ", " ").replace("Ċ", "\n")                                       
            data.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "generated_tokens": gen_tokens,
                "generated_probs": gen_prob,
                "generated_answer": answer,
                "generated_answer_probs": answer_probs,
                "generated_answer_tokens": answer_tokens})                        

        return data
    
    
    def answer_scoring(passages, passage_probs, prompts, gen_func, num_beams=None, temperature=None):
        """Implements our basic scoring strategy with normalization for long answers.

        Parameters
        ----------
        passages : list of str
        passage_probs : list of float
        prompts : list of str
        gen_func : supports only `run_eleuther_v2`.
        num_beams: Number of beams to identify high probability answers when doing beam search on generated answers.
        temperature: temperature to scale logit scores before calculating softmax probabilities on the vocab of words when not doing beam search.
        
        Returns
        -------
        list of pairs (score, dict), sorted with the largest score first.
        `dict` should be the return value of `gen_func` for an example.

        """
        data = []
        for passage, passage_prob, prompt in zip(passages, passage_probs, prompts):
            # Run `gen_func` on [prompt] (crucially, the singleton list here),
            # and get the dictionary `gen` from the singleton list `gen_func`
            # returns, and then use the values to score `gen` according to our
            # scoring method.
            #
            # Be sure to use "generated_answer_probs" for the scores.

            if temperature is not None:
                gs = gen_func([prompt], 
                              temperature=temperature,
                             do_sample=True,
                             top_p=0.95)[0]
            elif num_beams is not None:
                gs = gen_func([prompt], 
                              num_beams=num_beams,
                             do_sample=False)[0]
                
            # Normalize so that longer answers do not have a very small probability even if the prob of tokens in the answers are high.
            norm_factor = 1/len(gs["generated_answer_probs"])
            data.append([(np.product(gs["generated_answer_probs"]) ** norm_factor) * passage_prob, gs])

        # Return `data`, sorted with the highest scoring `(score, gen)`
        # pair given first.

        data = sorted(data, key=lambda x: -x[0])
        return data


    def original_system(question, gen_func=run_eleuther_v2, temperature=None, num_beams=None, k_passages=2, k_train_qa=2, joiner="\n\n"):
        """ Original system as a solution of homework that given a quesion returns the answer as a dict.
        
        Refer to the discussion at the top of this cell for key details on the developed system. The values of the temperature
        and num_beams arguments passed in this function are determined separately based on grid search.
        
        Parameters
        ----------
        question: quesiton string
        gen_func: model to generate answer sequence
        temperature: temperature to scale logit scores before calculating softmax probabilities on the vocab of words when not doing beam search.
        num_beams: Number of beams to identify high probability answers when doing beam search on generated answers.
        k_passages: Number of relevant passages to the questions for finding answers. A new prompt is created per passage.
        k_train_qa: Number of Q/A pairs for doing few-shot learning.
        joiner : str
            The character to use to join pieces of the prompt into a single str.
            
        """
        
        if not np.logical_xor(temperature is None, num_beams is None):
            raise ValueError("Ensure either 'num_beams' or 'temperature' are not none.")
        passages, passage_probs = get_passages_with_scores(question=question, k=k_passages)
        prompts = [build_few_shot_open_qa_prompt(question, passage, squad_index=SQUAD_INDEX, k=k_train_qa, joiner=joiner) for passage in passages]
        answer = answer_scoring(passages=passages, passage_probs=passage_probs, prompts=prompts, 
                                temperature=temperature, num_beams=num_beams, gen_func=gen_func)[0]
        return answer[1]
    
    
    def tune_temperature(examples, system, temperature_list, joiner="\n\n"):
        """Function to search over different values of the temperature parameter.
        
        For each temperature, macro_f1 score is calculated over the SQuAD dev dataset.
        """
        scores = {}
        for temperature in temperature_list:
            prompts = []
            gens = []
            for example in tqdm(examples, f"temperature={temperature}"):
                answer = system(question=example.question, temperature=temperature, gen_func=run_eleuther_v2)
                prompts += [" "] # The evaluate function does not need this.
                gens += [answer]
            scores[temperature] = evaluate(examples, prompts, gens)
            print(f"Temperature = {temperature}, macro_f1={scores[temperature]['macro_f1']}")
            
        best_temperature = -float("inf")
        best_score = -float("inf")

        for key, value in scores.items():
            if value["macro_f1"] > best_score:
                best_score = value["macro_f1"]
                best_temperature = key
                
        scores_disp = {key:value["macro_f1"] for key, value in scores.items()}
        scores_disp = pd.DataFrame({
            "temperature": scores_disp.keys(),
            "macro_f1": scores_disp.values()
        }
            )
        display(scores_disp)
        return best_temperature, scores
    
    # best_temperature, scores = tune_temperature(examples=dev_exs, system=original_system, temperature_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    def tune_n_beams(examples, system, num_beams_list, joiner="\n\n"):
        """Function to search over different values of the num_beams parameter.
        
        For each num_beams, macro_f1 score is calculated over the SQuAD dev dataset. As number of beams increase, macro_f1 is typically expcted
            to improve, however, at the expense of compute cost. A very limited number of values for n_beams could be experimented due to
            compute limits.
        """
        scores = {}
        for num_beams in num_beams_list:
            prompts = []
            gens = []
            for example in tqdm(examples, f"num_beams={num_beams}"):
                answer = system(question=example.question, num_beams=num_beams, gen_func=run_eleuther_v2)
                prompts += [" "] # The evaluate function does not need this.
                gens += [answer]
            scores[num_beams] = evaluate(examples, prompts, gens)
            print(f"num_beams = {num_beams}, macro_f1={scores[num_beams]['macro_f1']}")
            
        best_num_beams = -float("inf")
        best_score = -float("inf")

        for key, value in scores.items():
            if value["macro_f1"] > best_score:
                best_score = value["macro_f1"]
                best_num_beams = key
                
        scores_disp = {key:value["macro_f1"] for key, value in scores.items()}
        scores_disp = pd.DataFrame({
            "num_beams": scores_disp.keys(),
            "macro_f1": scores_disp.values()
        }
            )
        display(scores_disp)
        return best_num_beams, scores
    
    # best_num_beams, scores_beams = tune_n_beams(examples=dev_exs, system=original_system, num_beams_list=[2, 3, 4])

    
    def final_system(question):
        """Final function after all the parameters are determined.
        """
        return original_system(question=question, k_passages=2, k_train_qa=2, gen_func=run_eleuther_v2, temperature=None, num_beams=4, joiner="\n\n")
# STOP COMMENT: Please do not remove this comment.

if not os.path.exists(os.path.join("data", "openqa", "cs224u-openqa-test-unlabeled.txt")):
    !mkdir -p data/openqa
    !wget https://web.stanford.edu/class/cs224u/data/cs224u-openqa-test-unlabeled.txt -P data/openqa/

def create_bakeoff_submission():
    filename = os.path.join("data", "openqa", "cs224u-openqa-test-unlabeled.txt")
    
    # This should become a mapping from questions (str) to response
    # dicts from your system.
    gens = {} 
        
    with open(filename) as f:
        questions = f.read().splitlines()
    
    # `questions` is the list of questions you need to evaluate your system on.
    # Put whatever code you need to in here to evaluate your system.
    # All you need to be sure to do is create a list of dicts with at least
    # the keys of the dicts returned by `run_gpt` and `run_eleuther`.
    # Add those dicts to `gens`.
    #
    # Here is an example where we just do "Open QA with no context",
    # for an "original system" that would not earn any credit (since
    # it is not original!):
    for question in tqdm(questions):
        gens[question] = final_system(question)
        
    # Quick tests we advise you to run: 
    # 1. Make sure `gens` is a dict with the questions as the keys:
    assert all(q in gens for q in questions)
    # 2. Make sure the values are dicts and have the key we will use:
    assert all(isinstance(d, dict) and "generated_answer" in d for d in gens.values())
            
    # And finally the output file:
    with open("cs224u-openqa-bakeoff-entry.json", "wt") as f:
        json.dump(gens, f, indent=4)    

create_bakeoff_submission()
