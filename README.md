# Ranked Retrieval Search Engine
This is a Python implementation of indexing and searching techniques for ranked retrieval using the [Vector Space Model](http://nlp.stanford.edu/IR-book/html/htmledition/the-vector-space-model-for-scoring-1.html) (VSM). Specifically, the [lnc.ltc](http://nlp.stanford.edu/IR-book/html/htmledition/document-and-query-weighting-schemes-1.html) weighting scheme is implemented for the vector space scoring.

For the computation of cosine values, normalization by query vector length was left out intentionally as an optmization decision since it will not affect the relative cosine scores relationships between documents.

## Requirements
* [NLTK](http://www.nltk.org/) installed
* Corpus for indexing and searching with constituent documents named numerically (e.g. Reuters corpus in NLTK data)

## Indexing
* Command: `$ python index.py -i <directory-of-documents> -d <dictionary-file> -p <postings-file>`
 * `<directory-of-documents>` is the directory for the collection of documents to be indexed
 * `<dictionary-file>` is the filename of dictionary to be created by indexer
	 * lines are of the format: "<term> <df> <byte offset in postings file>"
	 * last line contains meta information and indicates all docIDs indexed in ascending order along with its vector length in the format <docID>:<vector length>
	 *	e.g. `metadata:1:213.415770684,5:72.7078088451,6:76.4752737343,9:36.1086776973,10:113.329002691,...`
 * `<postings-file>` is the filename of the postings file created by indexer
  * Non-human readable
  * raw bytes where every 8 bytes represents implicit docID, term frequency Integer pairs
* Porter stemming is used during the indexing stage

## Searching
Command: `$ python search.py -d <dictionary-file> -p <postings-file> -q <file-of-queries> -o <output-file-of-results>`
* `<dictionary-file>` and `<postings-file>` are created by the indexer as aforementioned
* `<file-of-queries>` is a text file containing the list of free text queries (i.e. what one would type into a web search bar), with one query for each line
 * An e.g. for a line of query could be `Japanese trade houses sold up to three cargoes of Thai sugar`
* `<output-file-of-results>` is the name of the output file for the search results for the given queries
 * For each query, the top 10 most relevant results will be displayed if more than 10 documents are deemed relevant throug the search, else all relevant documents will be displayed
 * For the same line number, each line in `<output-file-of-results>` is a space-delimited list of docIDs (sorted from most relevant to least relevant) corresponding to the search result for the corresponding query in `<file-of-queries>`. For e.g. `11609 13162 6 9193 1843 1395 12361 10758 9521 6412`

## Future areas for development
### Phrasal search
Phrasal queries could be supported via a bigram approach to indexing so that each entry in dictionary is a pair of terms. When we query, we also break up the query into bigrams and evaluate the score list for each bigram as individual queries. Once all bigrams in the query are evaluated, we combine all the score lists and update the scores associated with each docID if they occur in more than one of the bigram score lists. Finally, we sort this list and return the top 10 docIDs. To illustrate, suppose we have the query "please turn off your hand", we could conduct the following steps:

1. break query into bigrams: e.g. `"$ please", "please turn", "turn off", "off your", "your hand", "hand $"`
2. query each bigram as individual queries and get their lists of (docID, score) pairs:
 * "$ please"      (1, 0.7), (18, 0.3), (20, 0.2)
 * "please turn"   (1, 0.6), (18, 0.4), (22, 0.3)
 * "turn off"      (1, 0.2), (6, 0.1)
 * "off your"      (18, 0.4)
 * "your hand"     (18, 0.8)
 * "hand $"        (20, 0.4)
3. combine score lists, tabulate net scores for each docID and sort the list in descending net score. i.e. `(18, 1.9), (1, 1.5), (20, 0.6), (22, 0.3), (6, 0.1)`
4. hence, in the order of decreasing relevance, the docIDs are: `18, 1, 20, 22, 6`
 
### Web interface
A simple python flask server could be implemented to wrap this search engine in a web UI.
