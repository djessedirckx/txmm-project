import argparse
from collections import Counter, defaultdict
from itertools import chain
import math
import os
import re
from tqdm import tqdm
from typing import List

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from rouge import Rouge

PERMITTED_TITLES_SOURCE = "scientific-paper-summarisation/Data/Utility_Data/permitted_titles.txt"
non_content_keys = ['MAIN-TITLE', 'HIGHLIGHTS', 'KEYPHRASES', 'ABSTRACT', 'ACKNOWLEDGEMENTS', 'REFERENCES']
stop_words = set(stopwords.words('english'))

def preprocess_sentence(sentence, filter_sentence=True):
    """
    Preprocesses a sentence, turning it all to lowercase and tokenizing it into words.
    :param sentence: the sentence to pre-process.
    :return: the sentence, as a list of words, all in lowercase
    """
    
    if filter_sentence:
        sentence = sentence.lower()
        word_tokens = word_tokenize(sentence)

        # Apply POS tagging on word
        pos_tags = pos_tag(word_tokens)

        # Remove stopwords from sentence
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words and w.isalnum()]
        filtered_sentence_pos = [(w, tag) for w, tag in pos_tags if not w.lower() in stop_words and w.isalnum()]
        return filtered_sentence, filtered_sentence_pos
    
    # Remove all line endings, multiple whitespace etc. from sentence
    cleaned_sentence = ' '.join(sentence.split())  
    return cleaned_sentence

def paper_tokenize(text, sentences_as_lists=False, preserve_order=False):
    """
    Takes a paper with the sections delineated by '@&#' and splits them into a dictionary where the key is the section
    and the value is the text under that section. This could probably be a bit more efficient but it works well enough.
    :param text: the text of the paper to split
    :param sentences_as_lists: if true, returns the text of each section as a list of sentences rather than a single
                               string.
    :param preserve_order: if true, tracks the order in which the paper sections occured.
    :returns: a dictionary of the form (section: section_text)
    """
    with open(PERMITTED_TITLES_SOURCE, "r") as pt:
        permitted_titles = pt.read().split("\n")

    # Split the text into sections
    if preserve_order:
        split_text_1 = re.split("@&#", text)
        split_text = zip(split_text_1, range(len(split_text_1)))
    else:
        split_text = re.split("@&#", text)

    # The key value. This value is changed if a permitted section title is encountered in the list.
    state = ""

    # After the for loop, this dictionary will have keys relating to each permitted section, and values corresponding
    # to the text of that section
    sentences_with_states = defaultdict(str)
    sentences_with_states_pos = defaultdict(str)
    sentences = defaultdict(str)

    section_counts = defaultdict(int)
    
    paper_abstract = ""
    sentence_index = 0
    
    if preserve_order:
        for text, pos in split_text:

            # Hack for proper sentence tokenization because NLTK tokeniser doesn't work properly for tokenising papers
            text = text.replace("etal.", "etal")
            text = text.replace("et al.", "etal")
            text = text.replace("Fig.", "Fig")
            text = text.replace("fig.", "fig")
            text = text.replace("Eq.", "Eq")
            text = text.replace("eq.", "eq")
            text = text.replace("pp.", "pp")
            text = text.replace("i.e.", "ie")
            text = text.replace("e.g.", "eg")
            text = text.replace("ref.", "ref")
            text = text.replace("Ref.", "Ref")
            text = text.replace("etc.", "etc")
            text = text.replace("Figs.", "Figs")
            text = text.replace("figs.", "figs")
            text = text.replace("No.", "No")
            text = text.replace("eqs.", "eqs")

            # Checks if text is a section title
            if text.lower() in permitted_titles:
                state = text
                section_counts[state] += 1
            else:
                if sentences_as_lists:
                    if section_counts[state] > 1:
                        state = state + "_" + str(section_counts[state])

                    sentence_states = []
                    sentence_states_pos = []
                    for x in sent_tokenize(text):
                        filtered_sentence, filtered_sentence_pos = preprocess_sentence(x)
                        sentence_states.append(filtered_sentence)
                        sentence_states_pos.append(filtered_sentence_pos)

                    sentences_with_states[state] = sentence_states
                    sentences_with_states_pos[state] = sentence_states_pos
                    
                    sentence_storage = []
                    for x in sent_tokenize(text):
                        sentence = preprocess_sentence(x, filter_sentence=False)
                        sentence_storage.append((sentence, sentence_index))
                        sentence_index+=1
                    sentences[state] = sentence_storage
            if state == "ABSTRACT":
                paper_abstract = text.strip()

    return sentences, sentences_with_states, sentences_with_states_pos, paper_abstract

def get_paper_as_words(tokenized_paper):
    all_words = []
    for key in tokenized_paper.keys():
        
        # For every paper section that contains content information,
        # retrieve words
        if key not in non_content_keys:
            section_content = tokenized_paper[key]
            [all_words.extend(s) for s in section_content]
            
    return all_words

def compute_sentence_freq(words, sentences):
    freq_dict = {}
    for word in words:
        freq_dict[word] = sum(1 for sent in sentences if word in sent)
    return freq_dict

def compute_tf(word: str, sentence: List):
    freq = sum(1 for sent_word in sentence if sent_word == word)
    return freq / len(sentence)

def compute_idf(word: str, no_sentences: int, freq_dict: Counter):
    if word in freq_dict:
        sentence_freq = freq_dict[word]
        return math.log10(no_sentences / sentence_freq)
    return 0

def compute_tfidf(tf: float, idf: float):
    return tf * idf

def map_pos_tag(tag: str):
    if tag.startswith('J'):
        return ['a', 's']
    if tag.startswith('V'):
        return ['v']
    if tag.startswith('N'):
        return ['n']
    if tag.startswith('R'):
        return ['r']
    return ''

def compute_sentence_weight(sentence_pos, sentence, dict_freq, no_sentences):    
    sentence_score = 0

    for word, pos in sentence_pos:

        # Map NLTK POS tag to Wordnet POS tag
        mapped_tag = map_pos_tag(pos)

        # Get synonyms of word
        synsets = []

        # Only include words that have the same POS tag
        for w in wn.synsets(word):
            if w.pos() in mapped_tag:
                synsets.append(w)

        synonyms = set(chain.from_iterable([word.lemma_names() for word in synsets]))

        # Add original word to set of synonyms
        synonyms.add(word)

        # Compute tf-idf
        for syn_word in synonyms:
            tf = compute_tf(syn_word, sentence)
            idf = compute_idf(syn_word, no_sentences, dict_freq)
            sentence_score += compute_tfidf(tf, idf)
        
    return sentence_score

def summarize_paper(tokenized_paper, tokenized_paper_pos, paper_sentences, nr_sentences=5):
    sentence_weights = []
    
    # Get word representation of paper
    paper_words = get_paper_as_words(tokenized_paper)
    
    processed_sentences = []
    processed_sentences_pos = []
    original_sentences = []
    
    # Get all sentences in paper
    for section in tokenized_paper.keys():
        
        if section not in non_content_keys:
            processed_sentences_pos.extend(tokenized_paper_pos[section])
            processed_sentences.extend(tokenized_paper[section])
            original_sentences.extend(paper_sentences[section])
    
    # For every word, compute how often they appear in a sentence
    freq_dict = compute_sentence_freq(paper_words, processed_sentences)
    no_sentences = len(processed_sentences)
    
    for tok_sentence_pos, tok_sentence, orig_sentence in zip(processed_sentences_pos, processed_sentences, original_sentences):
        
        # Compute sentence weight and store with sentence
        sentence_weight = compute_sentence_weight(tok_sentence_pos, tok_sentence, freq_dict, no_sentences)
        sentence_weights.append((orig_sentence[0], orig_sentence[1], sentence_weight))
            
    # Create a dataframe of all sentences and sort descending by weight
    sentence_weights = pd.DataFrame(sentence_weights, columns=['sentence', 'index', 'weight'])
    sentence_weights.sort_values(by=['weight'], ascending=False, inplace=True)
    
    # Select desired number of sentences and sort by order of occurence in text
    summary = sentence_weights.head(nr_sentences).sort_values(by=['index'])['sentence'].values
    
    # Join selected strings into a summary
    string_summary = ' '.join(summary)
    
    return string_summary

rouge = Rouge()

def compute_metrics(paper_abstract: np.array, generated_summary: np.array):
    rouge_scores = rouge.get_scores(generated_summary, paper_abstract, avg=True)
    print(rouge_scores)
#     return rouge_scores['rouge-1'].values(), rouge_scores['rouge-2'].values(), rouge_scores['rouge-l'].values(),

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarise papers')
    parser.add_argument('--paper_path', type=str, help='Directory where parsed papers are stored')
    parser.add_argument('--summary_path', type=str, help='Directory where generated summaries are stored')
    parser.add_argument('--sum_length', type=int, help='Percentage of paper to determine summary length')
    args = parser.parse_args()

    PAPER_PATH = args.paper_path
    SUMMARY_PATH = args.summary_path
    # PAPER_PATH = 'data/parsed_papers'
    paper_file_names = os.listdir(PAPER_PATH)

    # Define desired number of sentences for a summary
    NR_OF_SENTENCES = args.sum_length

    ground_truth_summaries = np.empty(len(paper_file_names), dtype='object')
    generated_summaries = np.empty(len(paper_file_names), dtype='object')

    for i, paper_file_name in tqdm(enumerate(paper_file_names)):
        
        # Read paper file
        filepath = f'{PAPER_PATH}/{paper_file_name}'
        with open(filepath, "r") as paper_file:
            paper_content = paper_file.read()
        
        # Tokenize paper into sentences (and sentences into separate words) and get paper abstract
        paper_sentences, tokenized_paper, tokenized_paper_pos, paper_abstract = paper_tokenize(paper_content, sentences_as_lists=True, preserve_order=True)
        ground_truth_summaries[i] = paper_abstract
        
        # Summarize paper
        # sum_length = int(NR_OF_SENTENCES * sum(len(section_sentences) for section_sentences in paper_sentences))
        # sum_length = int(NR_OF_SENTENCES * len(paper_sentences))
        # print(sum_length)

        generated_summary = summarize_paper(tokenized_paper, tokenized_paper_pos, paper_sentences, NR_OF_SENTENCES)
        generated_summaries[i] = generated_summary

        # Write summary to disk for back-up
        with open(f'{SUMMARY_PATH}/tfidf_wordnet/{paper_file_name}', 'w') as sum_file:
            sum_file.write(generated_summary)
        
    # Compute ROUGE scores
    compute_metrics(ground_truth_summaries, generated_summaries)