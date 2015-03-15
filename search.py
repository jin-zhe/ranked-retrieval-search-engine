#!/usr/bin/python
import collections
import operator
import getopt
import codecs
import struct
import timeit
import pickle
import math
import nltk
import sys
import io
import re

IGNORE_STOPWORDS = True     # toggling the option for ignoring stopwords
IGNORE_SINGLES = True       # toggling the option for ignoring single character tokens
RECORD_TIME = False         # toggling for recording the time taken for indexer
BYTE_SIZE = 4               # docID is in int
TOP_LIMIT = 10              # number of most relevant docIDs to fetch

"""
conducts boolean queries from queries_file and writes outputs to output_file
params:
    dictionary_file:    dictionary file produced by indexer
    postings_file:      postings file produced by indexer
    queries_file:       file of boolean queries
    output_file:        responses to boolean queries
"""
def search(dictionary_file, postings_file, queries_file, output_file):
    # open files
    dict_file = codecs.open(dictionary_file, encoding='utf-8')
    post_file = io.open(postings_file, 'rb')
    query_file = codecs.open(queries_file, encoding='utf-8')
    out_file = open(output_file, 'w')
    vect_file = open("vector_squares_sum")

    # load dictionary to memory
    loaded_dict = load_dictionary(dict_file)
    dictionary = loaded_dict[0]     # dictionary map
    indexed_docIDs = loaded_dict[1] # list of all docIDs indexed in sorted order
    dict_file.close()

    # load vector_squares_sum to memory
    vector_squares_sum = pickle.load(vect_file)
    vect_file.close()

    # process each query
    queries_list = query_file.read().splitlines()
    for i in range(len(queries_list)):
        query = queries_list[i]
        result = get_top_cosine_scores(query, dictionary, post_file, vector_squares_sum, indexed_docIDs)
        # write each result to output
        for j in range(len(result)):
            docID = str(result[j][0])
            if (j != len(result) - 1):
                docID += ' '
            out_file.write(docID)
        if (i != len(queries_list) - 1):
            out_file.write('\n')

    # close files
    post_file.close()
    query_file.close()
    out_file.close()

"""
returns 2-tuple of loaded dictionary and total df
params:
    dict_file: opened dictionary file
"""
def load_dictionary(dict_file):
    dictionary = {}                 # dictionary map loaded
    indexed_docIDs = []             # list of all docIDs indexed
    docIDs_processed = False        # if indexed_docIDs is processed

    # load each term along with its df and postings file pointer to dictionary
    for entry in dict_file.read().split('\n'):
        # if entry is not empty (last line in dictionary file is empty)
        if (entry):
            # if first line of dictionary, process list of docIDs indexed
            if (not docIDs_processed):
                indexed_docIDs = [int(docID) for docID in entry[20:-1].split(',')]
                docIDs_processed = True
            # else if dictionary terms and their attributes
            else:
                token = entry.split(" ")
                term = token[0]
                df = int(token[1])
                offset = int(token[2])
                dictionary[term] = (df, offset)

    return (dictionary, indexed_docIDs)

"""
returns the TOP_LIMIT most relevant docIDs in the result for the given query
params:
    query:          the query string e.g. 'China's economic progress'
    dictionary:     the dictionary in memory
    indexed_docIDs: the list of all docIDs indexed
"""
def get_top_cosine_scores(query, dictionary, post_file, vector_squares_sum, indexed_docIDs):
    scores = {} # (accumulator) Key: docID, Value: raw tabulated score (before cosine normalization)
    N = len(indexed_docIDs) # number of documents in collection

    # for each query term
    for term, tf_raw_query in get_query_terms(query):
        if (term not in dictionary):    continue    # skip terms not in dictionary
        tf_wt_query = 1 + math.log(tf_raw_query, 10)
        df = dictionary[term][0]
        idf = math.log(N/float(df), 10)
        w_tq = tf_wt_query * idf    # weight for term in query

        # for each document in postings list
        for docID, tf_raw_document in load_posting_list(post_file, df, dictionary[term][1]):
            tf_wt_doc = 1 + math.log(tf_raw_document, 10)
            w_td = tf_wt_doc        # weight for term in current document
            # accumulate scores
            if (docID not in scores):
                scores[docID] = (w_tq * w_td)
            else:
                scores[docID] += (w_tq * w_td)

    # length normalize all scores
    for docID, score in scores.items():
        length = math.sqrt(vector_squares_sum[docID])
        scores[docID] /= length

    # sorts list and return most relevant documents
    most_relevant = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    if (len(most_relevant) <= TOP_LIMIT):
        return most_relevant
    else:
        return most_relevant[:10]

"""
returns a list of (term, tf)
params:
    query:  the query string e.g. 'China's economic progress'
"""
def get_query_terms(query):
    stemmer = nltk.stem.porter.PorterStemmer() # instantiate stemmer
    tokens = nltk.word_tokenize(query)
    stopwords = nltk.corpus.stopwords.words('english')
    # populate query_tf
    query_tf = {} # Key: term, Value: tf
    for word in tokens:
        term = word.lower()
        if (IGNORE_STOPWORDS and term in stopwords):    continue    # if ignoring stopwords
        term = stemmer.stem(term)   # stemming
        if (term[-1] == "'"):
            term = term[:-1]        # remove apostrophe
        if (IGNORE_SINGLES and len(term) == 1):         continue    # if ignoring single terms
        if (len(term) == 0):                            continue    # ignore empty terms

        if (term not in query_tf):
            query_tf[term] = 1
        else:
            query_tf[term] += 1

    return query_tf.items()
"""
returns posting list for term corresponding to the given offset
params:
    post_file:  opened postings file
    length:     length of posting list (same as df for the term)
    offset:     byte offset which acts as pointer to start of posting list in postings file
"""
def load_posting_list(post_file, length, offset):
    post_file.seek(offset)
    posting_list = []
    for i in range(length):
        posting = post_file.read(2 * BYTE_SIZE)
        pair = struct.unpack('II', posting) # docID, tf pair
        posting_list.append(pair)
    return posting_list

"""
prints the proper command usage
"""
def print_usage():
    print ("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

dictionary_file = postings_file = queries_file = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except (getopt.GetoptError, err):
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        queries_file = a
    elif o == '-o':
        output_file = a
    else:
        assert False, "unhandled option"
if (dictionary_file == None or postings_file == None or queries_file == None or output_file == None):
    print_usage()
    sys.exit(2)

if (RECORD_TIME): start = timeit.default_timer()                    # start time
search(dictionary_file, postings_file, queries_file, output_file)   # call the search engine on queries
if (RECORD_TIME): stop = timeit.default_timer()                     # stop time
if (RECORD_TIME): print ('Querying time:' + str(stop - start))      # print time taken