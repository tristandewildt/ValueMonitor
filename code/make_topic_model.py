import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import unicodedata
import re
import dateutil.parser
import _pickle as cPickle
import nltk

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct
from operator import itemgetter
from IPython.display import display, HTML
from simple_colors import *
from nltk.tag.perceptron import PerceptronTagger
tagger=PerceptronTagger()


def filter_stopwords_verbs(x, tags_to_select):
    
    pos_tagged_tokens = tagger.tag(nltk.word_tokenize(x))
    remaining_text = [s for s in pos_tagged_tokens if any(ext in s[1] for ext in tags_to_select)]
    remaining_text_untolken = ' '.join([word for word, pos in remaining_text])
    return remaining_text_untolken


def clean_df(df, columns_to_select_as_text, column_as_date, other_columns_to_keep, wordtagging, tags_to_select):
    
    df["text"] = df[columns_to_select_as_text].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df["date"] = df[column_as_date[0]].map(lambda x: dateutil.parser.parse(str(x)))
    
    df2 = df[["text", "date"] + other_columns_to_keep]
    df2.columns = ["text", "date"] + other_columns_to_keep
    
    df2["text"] = df2["text"].map(lambda x: re.sub(r'\W+', ' ', x))
    df2["text"] = df2["text"].map(lambda x: re.sub(r'http\S+', ' ', x))
    df2["text"] = df2["text"].map(lambda x: nltk.word_tokenize(x))
    df2["text"] = df2["text"].map(lambda x: " ".join(x))

    if wordtagging == True:      
        df2["text_tagged"] = df2["text"].apply(lambda x: filter_stopwords_verbs(x, tags_to_select))
    else:
        df2["text_tagged"] = df2["text"]
    return df2

def topic_int_or_string(Topic_selected, dict_anchor_words):
    
    if type(Topic_selected) == str:
        list_keys = list(dict_anchor_words.keys())
        Topic_selected_number = list_keys.index(Topic_selected)
    else:
        Topic_selected_number = Topic_selected
        
    return Topic_selected_number

def reduce_df(df, max_number_of_documents, dict_anchor_words, equilibrate):

    ''' Here we should include an option in case the dataset is not well spread over time '''
    if equilibrate == True and len(dict_anchor_words)>0:
        df_reduced = df
        
        #list_words = list(dict_anchor_words.values())
        #pattern = '|'.join([item for sublist in list_words for item in sublist])
        #df_reduced = df_reduced[df_reduced['text_tagged'].str.contains(pattern)]
        
        dict_frequencies = {}
        for key, value in dict_anchor_words.items():
            pattern = '|'.join(value)
            dict_frequencies[key] = len(df_reduced[df_reduced['text_tagged'].str.contains(pattern)])
        dict_frequencies = dict(sorted(dict_frequencies.items(), key=lambda item: item[1]))
    
        df_focused = pd.DataFrame(columns=df_reduced.columns.tolist())
        #number_of_articles_still_to_take = max_number_of_documents
        average_number_articles_to_take = round(max_number_of_documents / len(dict_anchor_words))
        for key, value in dict_frequencies.items():
            pattern = '|'.join(dict_anchor_words[key])
            df_this_value = df_reduced[df_reduced['text_tagged'].str.contains(pattern)]
            if len(df_this_value) > average_number_articles_to_take:
                df_this_value = df_this_value.sample(n=average_number_articles_to_take)
            else:
                remaining_articles_not_taken = average_number_articles_to_take - len(df_this_value)
                average_number_articles_to_take = average_number_articles_to_take + round(remaining_articles_not_taken / len(dict_anchor_words))
            #df_focused = df_focused.append(df_this_value)
            df_focused = pd.concat([df_focused,df_this_value])
        df_focused = df_focused.drop_duplicates(subset='text_tagged', keep="first")
        #number_of_documents_in_analysis = len(df_focused)
        df_reduced = df_focused
    
    else: 
        df_reduced = df.sample(min(max_number_of_documents, len(df)))
        
    #print(len(df_reduced))
    return df_reduced

def vectorize(df):
    
    vectorizer = TfidfVectorizer(
        max_df=.5,
        min_df=10,
        max_features=None,
        ngram_range=(1, 2),
        norm=None,
        binary=True,
        use_idf=False,
        sublinear_tf=False
    )

    vectorizer = vectorizer.fit(df['text_tagged'])
    tfidf = vectorizer.transform(df['text_tagged'])
    #vocab = vectorizer.get_feature_names()
    vocab = vectorizer.get_feature_names_out()
    vectorized_data = [vectorizer, tfidf, vocab]
    
    return vectorized_data

def make_topic_model(df, number_of_topics, anchors):
    
    vectorized_data = vectorize(df)
    
    tfidf = vectorized_data[1]
    vocab = vectorized_data[2]

    model = ct.Corex(n_hidden=number_of_topics, seed=42)
    model = model.fit(
            tfidf,
            words=vocab,
            anchors=anchors,
            anchor_strength=3) # Check whether this still works when there are no anchor words   

    return model

def find_best_number_of_topics(df, number_of_documents_in_analysis, min_number_of_topics, max_number_of_topics):
    ''' Think here what could be some errors that people could make with regard to input data '''
    equilibrate = False
    dict_anchor_words_empty = {}
    df_reduced = reduce_df(df, number_of_documents_in_analysis, dict_anchor_words_empty, equilibrate)
    
    interval = (max_number_of_topics - min_number_of_topics) / 4
    list_topics_to_try = np.arange(min_number_of_topics, (max_number_of_topics + interval), interval).tolist()
    list_topics_to_try = [int(i) for i in list_topics_to_try]
    
    dict_topic_correlation = {}
    dict_topic_models = {}
    
    for number_of_topics in list_topics_to_try:
        print("Working on model with "+str(number_of_topics)+" topics...") # Might need to put this as info
        
        anchors = []
        model = make_topic_model(df_reduced, number_of_topics, anchors)
        
        dict_topic_correlation[number_of_topics] = np.sum(model.tcs)
        dict_topic_models[number_of_topics] = model
        
    fig, ax1 = plt.subplots()
    
    ax1.plot(list(dict_topic_correlation.keys()),list(dict_topic_correlation.values()))
    ax1.set_xlabel('Number of topics', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Total correlation', fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()

    best_number_of_topics = max(dict_topic_correlation, key=dict_topic_correlation.get)
    
    return best_number_of_topics

def make_anchored_topic_model(df, number_of_topics, number_of_documents_in_analysis, dict_anchor_words, list_anchor_words_other_topics, list_rejected_words):
    ''' Think here what could be some errors that people could make with regard to input data '''
    equilibrate = True
    df_reduced = reduce_df(df, number_of_documents_in_analysis, dict_anchor_words, equilibrate)   
    vectorized_data = vectorize(df_reduced)
    vocab = vectorized_data[2]
    
    anchors = [[]] * number_of_topics
    
    counter = 0
    for key, value in dict_anchor_words.items():
        value_lowercase = value
        for i in range(len(value_lowercase)):
            value_lowercase[i] = value_lowercase[i].lower()
        anchors[counter] = value_lowercase
        counter += 1
    for i in list_anchor_words_other_topics:
        anchors[counter]=i
        counter += 1
    anchors[counter]=list_rejected_words
    
    anchors = [
        [a for a in topic if a in vocab]
        for topic in anchors]
    
    model = make_topic_model(df_reduced, number_of_topics, anchors)
    model_and_vectorized_data = [model, vectorized_data]
    
    return model_and_vectorized_data

def report_topics(model, dict_anchor_words, number_of_words_per_topic):
        
    list_values = []
    words_values = {}
    for key, value in dict_anchor_words.items(): 
        list_values.append(key)
        
    index_values = []
    for i in list_values:
        index_values.append(list_values.index(i))

    for i, topic_ngrams in enumerate(model.get_topics(n_words=number_of_words_per_topic)):
        topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
       
        if i in index_values:
            words_values[list_values[i]] = topic_ngrams
            print("Topic #{} ({}): {}".format(i, list_values[i], ", ".join(topic_ngrams)))
        else:
            #words_values[i] = topic_ngrams
            print("Topic #{}: {}".format(i, ", ".join(topic_ngrams)))
        words_values[i] = topic_ngrams

            
    return words_values

def report_topics_words_and_weights(model, dict_anchor_words, number_of_words_per_topic):
    
    topics_weights = {}
    list_values_int = list(range(len(dict_anchor_words)))
    list_values = list(dict_anchor_words.keys())
    for i, topic_ngrams in enumerate(model.get_topics(n_words=number_of_words_per_topic)):
        dict_words = {}
        for j in topic_ngrams:
            if j[1]> 0:
                dict_words[j[0]] = round(j[1],3)
        if i in list_values_int:
            topics_weights['Topic #'+str(i)+'# ('+str(list_values[i])+')'] = dict_words
        else:
            topics_weights['Topic #'+str(i)+'#'] = dict_words
    return topics_weights

def create_df_with_topics(df, model, vectorized_data, best_number_of_topics):
    vectorizer = vectorized_data[0]
    
    tfidf = vectorizer.transform(df['text_tagged']) 
    
    df_documents_topics = pd.DataFrame(
        model.transform(tfidf), 
        columns=[i for i in range(best_number_of_topics)]
    ).astype(float)

    df_documents_topics.index = df.index
    df_with_topics = pd.concat([df, df_documents_topics], axis=1)
      
    return df_with_topics

def export_documents_related_to_one_topic(df_with_topics, dict_anchor_words, file_name, Topic_selected):
    
    Topic_selected_number = topic_int_or_string(Topic_selected, dict_anchor_words)
    
    root = '/gdrive/My Drive/Topic_modelling_analysis/'
    name, extension = os.path.splitext(file_name)
    
    df_selected = df_with_topics[df_with_topics[Topic_selected_number] == 1] 
    df_selected = DataFrame(df_selected,columns=["text", "date"])

    df_selected.to_csv(str(root) + "save/" + str(name) + "_topic_"+str(Topic_selected)+".csv", index = False)
    
def export_topic_model(model_and_vectorized_data, dict_anchor_words, best_number_of_topics, file_name):
    
    root = os.getcwd()
    name, extension = os.path.splitext(file_name)
    
    saved_data = [model_and_vectorized_data, dict_anchor_words, best_number_of_topics]
    
    cPickle.dump(saved_data, open(str(root) + "/save/" + str(name) + "_saved_topic_model", 'wb'))

    
def find_documents_related_to_the_value_that_are_not_yet_in_the_topics(df_with_topics, model_and_vectorized_data, dict_anchor_words, list_of_words, topic_to_evaluate, number_of_words_per_topic_to_show):
        
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
       
    listToStr = ', '.join([str(elem) for elem in list_of_words])
    name_column_counts = "Number of documents found in each topic with keywords '" +str(listToStr)+"' which have not been assigned to topic "+str(topic_to_evaluate)+"."

    df_selected = df_with_topics[df_with_topics["text"].str.contains('|'.join(list_of_words))]
    df_selected = df_selected[df_selected[topic_to_evaluate_number] == 0]
    
    print(str(len(df_selected))+" documents found that contains words in the list and have not been attributed to the topic of interest.")
       
    model = model_and_vectorized_data[0]
    list_topics = list(range(len(model.get_topics())))
    list_topics.remove(topic_to_evaluate_number)
    df_column_topics = df_selected[list_topics]
    
    df_column_topics = pd.DataFrame(df_column_topics.sum(axis=0))
    df_column_topics = df_column_topics.rename(columns={0: name_column_counts})

    list_values = []
    for key, value in dict_anchor_words.items(): 
        list_values.append(key)

    words_topics = []
    for i, topic_ngrams in enumerate(model.get_topics(n_words=number_of_words_per_topic_to_show)):
        if i in list_topics:
            topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
            if len(list_values) > i:
                words_topics.append("Topic #{} ({}): {}".format(i, list_values[i], ", ".join(topic_ngrams)))
            else:
                words_topics.append("Topic #{}: {}".format(i, ", ".join(topic_ngrams)))

    df_column_topics.insert(0, "Topics", words_topics)
    df_column_topics = df_column_topics.sort_values(by=[name_column_counts], ascending=False)

    display(HTML(df_column_topics.to_html(justify = "center")))
    
def sample_documents(df_selected, random_number_documents_to_return, text_table):

    df_selected_texts = pd.DataFrame(df_selected["text"].sample(n = min(random_number_documents_to_return, len(df_selected))))
    
    df_selected_texts["text"] = df_selected_texts["text"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
    df_selected_texts = df_selected_texts.rename(columns={"text": text_table})

    display(HTML(df_selected_texts.to_html(justify = "center")))

def print_documents_related_to_the_value_that_are_not_yet_in_the_topics(df_with_topics, dict_anchor_words, list_of_words, topic_to_evaluate, topic_in_which_some_keywords_are_found, random_number_documents_to_return):
    
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
    
    df_selected = df_with_topics[df_with_topics["text_tagged"].str.contains('|'.join(list_of_words))]
    df_selected = df_selected[(df_selected[topic_to_evaluate_number] == 0) & (df_selected[topic_in_which_some_keywords_are_found] == 1)]
    
    listToStr = ', '.join([str(elem) for elem in list_of_words])
    text_table = "Random " + str(random_number_documents_to_return) + " documents in topic " + str(topic_in_which_some_keywords_are_found) + " with keywords '" + str(listToStr) + "' that have not been assigned to topic " + str(topic_to_evaluate) + "."
    
    sample_documents(df_selected, random_number_documents_to_return, text_table)
    
def print_sample_documents_related_to_topic(df_with_topics, dict_anchor_words, topic_to_evaluate, random_number_documents_to_return, model, top_x_words_of_topic_to_show_in_text):
    
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)

    df_selected = df_with_topics[df_with_topics[topic_to_evaluate_number] == 1]
    

    text_table = "Random " + str(random_number_documents_to_return) + " documents in topic " + str(topic_to_evaluate) + "."
    
    sample_documents(df_selected, random_number_documents_to_return, text_table)

def print_sample_documents_related_to_topic_with_keywords(df_with_topics, dict_anchor_words, list_of_words, topic_to_evaluate, random_number_documents_to_return):
    
    topic_to_evaluate_number = topic_int_or_string(topic_to_evaluate, dict_anchor_words)
    
    df_selected = df_with_topics[df_with_topics["text"].str.contains('|'.join(list_of_words))]
    df_selected = df_selected[df_selected[topic_to_evaluate_number] == 1]
    
    listToStr = ', '.join([str(elem) for elem in list_of_words])
    text_table = "Random " + str(random_number_documents_to_return) + " documents in topic " + str(topic_to_evaluate) + " with keywords '" + str(listToStr) + "."
    
    sample_documents(df_selected, random_number_documents_to_return, text_table)
    
def print_sample_articles_topic(df_with_topics, dict_anchor_words, topics, selected_value, size_sample, window, show_extracts, show_full_text):
    
    words_topics = topics

    #window = 10

    list_values = list(dict_anchor_words.keys())
    
    if type(selected_value) == str:
        selected_value_int = list_values.index(selected_value)
    else:
        selected_value_int = selected_value
    
    df_with_topics_to_analyse = df_with_topics.loc[df_with_topics[selected_value_int] > 0]
    
    
    sampled_df = df_with_topics_to_analyse.sample(n = min(size_sample, len(df_with_topics_to_analyse)))
    if type(selected_value) == str:
        print("Keywords related to "+str(selected_value)+" found in text:"+str(words_topics[selected_value]))
    else:
        print("Keywords related to topic "+str(selected_value))  
    
    print("")
    
    for index, row in sampled_df.iterrows():
        print('\033[1m' + 'Article '+str(index) + '\033[0m')
        if 'title' in sampled_df:
            print("Title: "+str(row['title']))
        if 'date' in sampled_df:
            print("Date: "+str(row['date']))
        if 'dataset' in sampled_df:
            print("Dataset: "+str(row['dataset']))
        
        text_combined_tagged = row['text_tagged']
        text_combined_not_tagged = row['text']
        
        tokens = text_combined_not_tagged.split() #### check here with spaces

        if show_full_text == True:
            for word in words_topics[selected_value]:
                text_combined_not_tagged = re.sub(word, '\033[1m' + '[' + str(red(word)) + ']' + '\033[0m', text_combined_not_tagged, flags=re.IGNORECASE)
            print(text_combined_not_tagged)
    
        if show_extracts == True:
            print("Values:")
            print("")
            for index in range(len(tokens)):
                if tokens[index].lower() in words_topics[selected_value]:
                    start = max(0, index-window)
                    finish = min(len(tokens), index+window+1)
                    lhs = " ".join( tokens[start:index] )
                    rhs = " ".join( tokens[index+1:finish] )
                    conc = "%s [%s] %s" % (" - "+str(lhs), '\033[1m' + str(red(tokens[index])) + '\033[0m', rhs)
                    print(conc)
                    print("")
        print("")
    
def print_sample_articles_value_and_topic(df_with_topics, dict_anchor_words, topics, selected_value, selected_topic, size_sample, window, show_extracts, show_full_text):
    
    words_topics = topics
    words_selected_topic = topics[int(selected_topic)]

    #window = 10
    
    list_values = list(dict_anchor_words.keys())
    
    if type(selected_value) == str:
        selected_value_int = list_values.index(selected_value)
    else:
        selected_value_int = selected_value
    
    df_with_topics_to_analyse = df_with_topics.loc[df_with_topics[selected_value_int] > 0]
    
    
    sampled_df = df_with_topics_to_analyse.sample(n = min(size_sample, len(df_with_topics_to_analyse)))
    if type(selected_value) == str:
        print("Keywords related to "+str(selected_value)+" found in text:"+str(words_topics[selected_value]))
    else:
        print("Keywords related to topic "+str(selected_value))  
    
    print("")
    
    for index, row in sampled_df.iterrows():
        print('\033[1m' + 'Article '+str(index) + '\033[0m')
        if 'title' in sampled_df:
            print("Title: "+str(row['title']))
        if 'Titel' in sampled_df:
            print("Title: "+str(row['Titel']))
        if 'date' in sampled_df:
            print("Date: "+str(row['date']))
        if 'dataset' in sampled_df:
            print("Dataset: "+str(row['dataset']))
        
        text_combined_tagged = row['text_tagged']
        text_combined_not_tagged = row['text']
        
        tokens = text_combined_not_tagged.split() #### check here with spaces

        if show_full_text == True:
            for word in words_topics[selected_value]:
                text_combined_not_tagged = re.sub(word, '\033[1m' + '[' + str(red(word)) + ']' + '\033[0m', text_combined_not_tagged, flags=re.IGNORECASE)
            for word in words_selected_topic:
                text_combined_not_tagged = re.sub(word, '\033[1m' + '[' + str(green(word)) + ']' + '\033[0m', text_combined_not_tagged, flags=re.IGNORECASE)
            print(text_combined_not_tagged)
    
        if show_extracts == True:
            print("Words values and topics:")
            print("")
            for index in range(len(tokens)):
                if tokens[index].lower() in words_topics[selected_value]:
                    start = max(0, index-window)
                    finish = min(len(tokens), index+window+1)
                    lhs = " ".join( tokens[start:index] )
                    rhs = " ".join( tokens[index+1:finish] )
                    conc = "%s [%s] %s" % (" - "+str(lhs), '\033[1m' + str(red(tokens[index])) + '\033[0m', rhs)
                    print(conc)
                    print("")
                if tokens[index].lower() in words_selected_topic:
                    start = max(0, index-window)
                    finish = min(len(tokens), index+window+1)
                    lhs = " ".join( tokens[start:index] )
                    rhs = " ".join( tokens[index+1:finish] )
                    conc = "%s [%s] %s" % (" - "+str(lhs), '\033[1m' + str(green(tokens[index])) + '\033[0m', rhs)
                    print(conc)
                    print("")
        print("")
        

def import_topic_model(combined_STOA_technologies_saved_topic_model, df):
    
    imported_data = combined_STOA_technologies_saved_topic_model
    df_with_topics = create_df_with_topics(df, imported_data[0][0], imported_data[0][1], imported_data[2])
    dict_anchor_words = imported_data[1]
    
    topics = imported_data [3]

    results_import = [df_with_topics, topics, dict_anchor_words]
    return(results_import)

def explore_topics_in_dataset(df_with_topics, number_of_topics_to_find, number_of_documents_in_analysis, number_of_words_per_topic, dict_anchor_words, topics, selected_value):
    
    words_values = topics

    #window = 10
    
    list_values = list(dict_anchor_words.keys())
    
    selected_value_int = list_values.index(selected_value)
    
    df_with_topics_to_analyse = df_with_topics.loc[df_with_topics[selected_value_int] > 0]

    dict_anchor_words2 = {}
    list_anchor_words_other_topics2 = []
    list_rejected_words2 = []

    #remove columns with int
    df_with_topics_to_analyse
    df_with_topics_to_analyse = df_with_topics_to_analyse[[c for c in df_with_topics_to_analyse.columns if type(c) != int]]


    tags_to_select = ['NN', 'NNP', 'NNS', 'JJ']
    df_with_topics_to_analyse["text_tagged"] = df_with_topics_to_analyse["text_tagged"].apply(lambda x: filter_stopwords_verbs(x, tags_to_select))
    
    model_and_vectorized_data = make_anchored_topic_model(df_with_topics_to_analyse, number_of_topics_to_find, min(number_of_documents_in_analysis, len(df_with_topics)), dict_anchor_words2, list_anchor_words_other_topics2, list_rejected_words2)
    topics2 = report_topics(model_and_vectorized_data[0], dict_anchor_words2,number_of_words_per_topic)
    df_with_topics = create_df_with_topics(df_with_topics_to_analyse, model_and_vectorized_data[0], model_and_vectorized_data[1], number_of_topics_to_find)

    df_with_topics_sum_dataset_short = df_with_topics[[c for c in df_with_topics.columns if type(c) == int]]
    
    get_topics = model_and_vectorized_data[0].get_topics()
    list_topics = []
    for topic_n,topic in enumerate(get_topics):
        topic = [(w,mi,s) if s > 0 else ('~'+w,mi,s) for w,mi,s in topic]
        if len(topic) > 0:
            words,mis,signs = zip(*topic)    
            topic_str = ', '.join(words)
        else:
            topic_str = ''
        list_topics.append(topic_str)

    df_with_topics_sum_dataset_short.columns = list_topics
    df_sum_dataset_short = df_with_topics_sum_dataset_short.sum(numeric_only=True)
    series_perc_dataset_short = df_sum_dataset_short.apply(lambda x: x / len(df_with_topics_sum_dataset_short) * 100)
    series_perc_dataset_short = series_perc_dataset_short.sort_values(ascending=False)
    
    dict_dataset_short = series_perc_dataset_short.to_dict()
    #plt.figure(figsize=(10,number_of_topics_to_find / 2))
    plt.barh(list(dict_dataset_short.keys()), list(dict_dataset_short.values()))
    plt.gca().invert_yaxis()
    
    plt.rcParams.update({'font.size': 12})
    plt.title('Occurence of topics in dataset')
    plt.xlabel('Percentage')
    plt.show()
