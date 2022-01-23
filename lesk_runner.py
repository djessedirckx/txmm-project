import argparse
from collections import defaultdict, Counter
import os
import re
from tqdm import tqdm
from typing import List

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd

PERMITTED_TITLES_SOURCE = "scientific-paper-summarisation/Data/Utility_Data/permitted_titles.txt"
non_content_keys = ['', 'MAIN-TITLE', 'HIGHLIGHTS', 'KEYPHRASES', 'ABSTRACT', 'ACKNOWLEDGEMENTS', 'ACKNOWLEDGEMENTS', 'REFERENCES']
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

def compute_sentence_weight(sentences_pos: List, text: Counter, total_words, total_words_synsets) -> int:
    
    sentence_score = 0
    
    # Iterate over all words in the sentence
    for word, pos in sentences_pos:

        total_words += 1

        # Map NLTK POS tag to Wordnet POS tag
        mapped_tag = map_pos_tag(pos)
        
        # Get word synsets that have the same POS tag
        synsets = []
        for w in wn.synsets(word):
            if w.pos() in mapped_tag:
                synsets.append(w)

        total_words_synsets += len(synsets)
        
        # If word has synsets (i.e., is known by wordnet), continue
        best_synset_score = 0
        for synset in synsets:
            
            # Get and tokenize gloss and remove stopwords and punctuation
            filtered_gloss, _ = preprocess_sentence(synset.definition())
            
            # Compute score
            score = 0
            for def_word in filtered_gloss:
                if def_word in text:
                    score += text[def_word]

            if score > best_synset_score:
                best_synset_score = score
                
        # Update sentence score
        sentence_score += best_synset_score

    return sentence_score, total_words, total_words_synsets

def summarize_paper(tokenized_paper, tokenized_paper_pos, paper_sentences, nr_sentences, total_words, total_words_synsets):
    sentence_weights = []
    
    # Get word representation of paper
    paper_words = get_paper_as_words(tokenized_paper)
    counted_words = Counter(paper_words)

    for section in tokenized_paper.keys():
        
        if section not in non_content_keys:
            section_content = tokenized_paper[section]
            section_content_pos = tokenized_paper_pos[section]
            section_sentences = paper_sentences[section]
            
            for tok_sentence, tok_sentence_pos, orig_sentence in zip(section_content, section_content_pos, section_sentences):
                
                # Compute sentence weight and store with sentence
                sentence_weight, total_words, total_words_synsets = compute_sentence_weight(tok_sentence_pos, counted_words, total_words, total_words_synsets)
                sentence_weights.append((orig_sentence[0], orig_sentence[1], sentence_weight))
            
    # Create a dataframe of all sentences and sort descending by weight
    sentence_weights = pd.DataFrame(sentence_weights, columns=['sentence', 'index', 'weight'])
    sentence_weights.sort_values(by=['weight'], ascending=False, inplace=True)

    print(sentence_weights['weight'].head(nr_sentences).mean())
    
    # Select desired number of sentences and sort by order of occurence in text
    summary = sentence_weights.head(nr_sentences).sort_values(by=['index'])['sentence'].values
    
    # Join selected strings into a summary
    string_summary = ' '.join(summary)
    
    return string_summary, total_words, total_words_synsets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarise papers')
    parser.add_argument('--paper_path', type=str, help='Directory where parsed papers are stored')
    parser.add_argument('--summary_path', type=str, help='Directory where generated summaries are stored')
    parser.add_argument('--sum_length', type=int, help='Number of sentences for a paper')
    args = parser.parse_args()

    total_words = 0
    total_words_synsets = 0

    PAPER_PATH = args.paper_path
    SUMMARY_PATH = ''
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
        generated_summary, total_words, total_words_synsets = summarize_paper(tokenized_paper, tokenized_paper_pos, paper_sentences, NR_OF_SENTENCES, total_words, total_words_synsets)
        generated_summaries[i] = generated_summary

        # Write summary to disk for analysis
        with open(f'{SUMMARY_PATH}/wordnet/{paper_file_name}', 'w') as sum_file:
            sum_file.write(generated_summary)