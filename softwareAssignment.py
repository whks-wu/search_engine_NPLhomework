"""
Introduction to Python Programming
Software Assignment
"""
import math
from stemming.porter2 import stem
from collections import Counter
import sys
import xml.sax, string
from decimal import Decimal, getcontext
# Set Precision
getcontext().prec = 18

class DocHandler(xml.sax.ContentHandler):
    """ Custom SAX handler for extracting content from XML tags DOC, HEADLINE, TEXT and P.

    After SAX Parser we get this structure:
{
    "documents": [
        {
            "id": "doc1",
            "headline": ["first", "document", "headline"],
            "paragraphs": [
                ["paragraph", "1", "of", "first", "document"],
                ["paragraph", "2", "of", "first", "document"]
            ]
        }
    ]
}

    """

    def __init__(self):
        super().__init__()
        # Flags to track which element we're currently processing
        self.in_headline = False
        self.in_text = False
        self.in_p = False

        # Temporary collect character data
        self.current_data = ""

        # Track the current document being processed
        self.current_doc = None

        # Storage for extracted content
        self.results = {
            "documents": []
        }

    def startElement(self, name, attrs):
        """Called when an opening tag is encountered."""
        # Reset the character data buffer at the start of any element
        self.current_data = ""

        # Start a new doc when DOC tag is found
        if name == "DOC":
            # Create a doc structure to store text in every doc
            self.current_doc = {
                "id": attrs.get ("id", "unknown"),
                "headline":"",
                "paragraphs":[]
            }
            self.results["documents"].append(self.current_doc)
            #print(f"DOC ID: {self.current_doc["id"]}")

        # Set flags for content-containing elements we care about
        # This flag tells the parser, "We are now inside a HEADLINE element."
        elif name == "HEADLINE":
            self.in_headline = True
        elif name == "TEXT":
            self.in_text = True
        elif name == "P":
            self.in_p = True

    # While the flag is True, the characters method will collect any text content found inside that element:
    def characters(self, content):
        """Called for the text content between tags."""

        if self.in_headline or self.in_p or self.in_text:
            self.current_data += content

    def endElement(self, name):
        """Called when a closing tag is encountered."""
        if name == "HEADLINE":
            # Remove punctuation, convert to lowercase, and split into token
            # str.maketrans('', '', string.punctuation) maps each punctuation character to None
            # text.translate(translator) applies this mapping to the string
            translator = str.maketrans('','', string.punctuation)
            clean_content = (self.current_data.
                             translate(translator).lower().strip().split())
            if clean_content:
                # print(f"HEADLINE: {clean_content}")
                self.current_doc["headline"] = clean_content
            self.in_headline = False

        elif name == "P":
            clean_content = (self.current_data.
                             translate(str.maketrans('', '', string.punctuation)).lower().strip().split())
            if clean_content:
                # print(f"P: {clean_content}")
                self.current_doc["paragraphs"].append(clean_content)
            self.in_p = False

        elif name =="TEXT":
            clean_content = (self.current_data.
                             translate(str.maketrans('', '', string.punctuation)).lower().strip().split())
            if clean_content:
                self.current_doc["paragraphs"].append(clean_content)
            self.in_text = False
        elif name == "DOC":
            # We've finished processing this document
            self.current_doc = None


class SearchEngine:

    def __init__(self, collection_name, create, handler):
        """
        Initialize the search engine, i.e. create or read in index. If
        create=True, the search index should be created and written to
        files. If create=False, the search index should be read from
        the files. The collection_name points to the filename of the
        document collection (without the .xml at the end). Hence, you
        can read the documents from <collection_name>.xml, and should
        write / read the idf index to / from <collection_name>.idf, and
        the tf index to / from <collection_name>.tf respectively. All
        of these files must reside in the same folder as THIS file. If
        your program does not adhere to this "interface
        specification", we will subtract some points as it will be
        impossible for us to test your program automatically!
        """
        self.query_terms = None
        self.collection_name = collection_name
        self.create = create
        self.idf_values = []   # Store IDF values as a list (indexed by vocabulary)
        self.vocabulary = []  # Store vocabulary as a sorted list of terms
        self.doc_ids = []  # Store doc IDs as a sorted list
        # Store TF values as a list of lists/tuples: [[(term_idx, tf_value), ...], ...]
        # where outer list index is doc_idx, inner list contains (term_idx, tf_value) for terms in that doc
        self.tf_data = []

        if create:
            print("Reading index from file...")
            # Extract every word from each doc into text list
            all_terms = []
            for doc in handler.results["documents"]:
                self.doc_ids.append(doc["id"]) # Store doc IDs
                all_terms.extend(doc["headline"])
                for paragraph in doc["paragraphs"]:
                    all_terms.extend(paragraph)

                # Stem and create unique vocabulary
                stemmed_unique_terms = sorted(list(set(stem(term) for term in all_terms)))
                self.vocabulary = stemmed_unique_terms

                # Create a term_to_idx mapping for faster lookup during TF processing
                self.term_to_idx = {term: idx for idx, term in enumerate(self.vocabulary)}

            # ---- Compute IDF ----
            with open(f"{collection_name}.idf", "w") as f_idf:
                num_docs = Decimal(len(handler.results["documents"]))
                # The final doc_term_presence will be a list of sets.
                # Each set in this list corresponds to a specific document in collection, ordered by its index.
                # doc_term_presence[0] will be a set for Document 0.
                # doc_term_presence[1] will be a set for Document 1.
                doc_term_presence = [set() for _ in range(len(handler.results["documents"]))]

                # First pass: populate doc_term_presence sets for all documents
                for doc_idx, doc in enumerate(handler.results["documents"]):
                    for word in doc["headline"]:
                        stemmed_word = stem(word)
                        if stemmed_word in self.term_to_idx:
                            doc_term_presence[doc_idx].add(self.term_to_idx[stemmed_word])
                    for paragraph in doc["paragraphs"]:
                        for word in paragraph:
                            stemmed_word = stem(word)
                            if stemmed_word in self.term_to_idx:
                                doc_term_presence[doc_idx].add(self.term_to_idx[stemmed_word])

                # Second pass: compute IDF for each term in vocabulary
                # If your self.vocabulary had, for example, 7 terms
                # (like in our simple example:
                # "cat", "dog", "mat", "ran", "sat", "small", "the"), then len(self.vocabulary) would be 7.
                # [Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')]
                self.idf_values = [Decimal(0)] * len(self.vocabulary)
                for term_idx, stemmed_word in enumerate(self.vocabulary):
                    num_docs_containing_term = 0
                    for doc_idx in range(len(handler.results["documents"])):
                        if term_idx in doc_term_presence[doc_idx]:
                            num_docs_containing_term += 1

                    idf = Decimal.ln(
                        num_docs / num_docs_containing_term) if num_docs_containing_term > 0 else Decimal.ln(num_docs)
                    self.idf_values[term_idx] = idf
                    f_idf.write(f"{stemmed_word}\t{idf}\n")
            print("IDF computation done.")

            # ---- Compute TF ----
            with open(f"{collection_name}.tf", "w") as f_tf:
                self.tf_data = [[] for _ in range(len(handler.results["documents"]))]  # Initialize for each doc

                for doc_idx, doc in enumerate(handler.results["documents"]):
                    id_doc = doc["id"]

                    terms_one_doc = []
                    for term in doc["headline"]:
                        terms_one_doc.append(stem(term))
                    for paragraph in doc["paragraphs"]:
                        for term in paragraph:
                            terms_one_doc.append(stem(term))

                    term_counts = Counter(terms_one_doc)

                    doc_tf_entries = []  # Store (term_idx, tf_value) for this document
                    max_occur_in_doc = Decimal(max(term_counts.values(), default=1))

                    for stemmed_word, count in term_counts.items():
                        if stemmed_word in self.term_to_idx:
                            term_idx = self.term_to_idx[stemmed_word]
                            tf = Decimal(count) / max_occur_in_doc
                            doc_tf_entries.append((term_idx, tf))
                            f_tf.write(f"{id_doc}\t{stemmed_word}\t{tf}\n")
                    self.tf_data[doc_idx] = doc_tf_entries  # Store for later use in memory
            print("TF computation done.")

        else:
            print("Reading index from file...")
            # Reading the IDF and TF from files
            term_to_idx_temp = {}
            with open(f"{collection_name}.idf", "r") as f_idf:
                for idx, line in enumerate(f_idf):
                    term, idf_value = line.strip().split()
                    self.vocabulary.append(term)
                    self.idf_values.append(Decimal(idf_value))
                    term_to_idx_temp[term] = idx
            self.term_to_idx = term_to_idx_temp  # Set the mapping after loading all vocabulary

            # Read TF data and reconstruct for list-based access
            # We need to map doc_id to doc_idx
            doc_id_to_idx = {}

            with open(f"{collection_name}.tf", "r") as f_tf:
                for line in f_tf:
                    doc_id, term, tf_value_str = line.strip().split()
                    if doc_id not in doc_id_to_idx:
                        doc_id_to_idx[doc_id] = len(self.doc_ids)  # Assign new index
                        self.doc_ids.append(doc_id)
                        self.tf_data.append([])  # Initialize an empty list for this new doc

                    term_idx = self.term_to_idx.get(term)
                    if term_idx is not None:
                        doc_idx = doc_id_to_idx[doc_id]
                        self.tf_data[doc_idx].append((term_idx, Decimal(tf_value_str)))

            print("Done.")

    def execute_query(self, query_terms):
            """
            Input to this function: List of query terms

            Returns the 10 highest ranked documents together with their
            tf.idf scores, sorted by score. For instance,

            [('NYT_ENG_19950101.0001', 0.07237004260325626),
             ('NYT_ENG_19950101.0022', 0.013039249597972629), ...]

            May be less than 10 documents if there aren't as many documents
            that contain the terms.
            """

            # Apply stemming to the query terms and get their indices
            stemmed_query_term_indices = []
            for term_q in query_terms:
                stemmed_term = stem(term_q)
                if stemmed_term in self.term_to_idx:
                    stemmed_query_term_indices.append(self.term_to_idx[stemmed_term])

            # Create query tf-idf vector (query_tf_idf)
            # tf-idf = tf(term of query) * idf(same term in docs)
            query_term_counts = Counter(stemmed_query_term_indices)  # Count occurrences of indices
            max_occurrences = Decimal(max(query_term_counts.values(), default=1))  # or 1 - Avoid Empty list divide by zero

            # Compute tf for every term in query
            query_tf = {}
            for term_idx, count in query_term_counts.items():
                # Formula of tf: Number of times term occurs in query/maxOccurrences
                query_tf[term_idx] = Decimal(count) / max_occurrences

            # Create a tf-idf vector for the query as a dense list
            query_vector = [Decimal(0)] * len(self.vocabulary)
            for term_idx, tf_val in query_tf.items():
                idf_val = self.idf_values[term_idx]
                query_vector[term_idx] = tf_val * idf_val

            # Compute cosine similarity
            cosine_similarities = []
            for doc_idx, doc_id in enumerate(self.doc_ids):
                doc_tf_entries = self.tf_data[doc_idx]  # List of (term_idx, tf_value) for this doc

                dot_product = Decimal(0)
                norm_doc_sq = Decimal(0)

                # Efficiently compute dot product and doc norm for sparse tf data
                # Iterate only over terms present in the document
                for term_idx, doc_term_tf in doc_tf_entries:
                    doc_term_tfidf = doc_term_tf * self.idf_values[term_idx]
                    dot_product += query_vector[term_idx] * doc_term_tfidf
                    norm_doc_sq += doc_term_tfidf ** 2

                norm_query_sq = sum(value ** 2 for value in query_vector)

                if norm_doc_sq > 0 and norm_query_sq > 0:
                    norm_query = Decimal(math.sqrt(norm_query_sq))
                    norm_doc = Decimal(math.sqrt(norm_doc_sq))
                    cos_sim = dot_product / (norm_query * norm_doc)
                else:
                    cos_sim = Decimal(0)

                cosine_similarities.append((doc_id, cos_sim))

            # Sort the documents by cosine similarity in descending order
            sorted_similarities = sorted(
                [(doc_id, cos_sim) for doc_id, cos_sim in cosine_similarities if cos_sim > 0],
                key=lambda x: x[1], reverse=True)

            # Return the top 10 (or fewer) documents
            return sorted_similarities[:10]


    def execute_query_console(self):
            """
            When calling this, the interactive console should be started,
            ask for queries and display the search results, until the user
            simply hits enter.
            """
            while True:
                # Removes spaces (including blank lines) before and after user input,
                # so that when user hits enter, program exits
                query = input("Please enter query, terms separated by whitespace:").strip()

                if not query: # Check if query is empty, when user hits enter, then program exits
                    sys.exit()

                self.query_terms = query.split()

                results = self.execute_query(self.query_terms)
                if results:
                    print("I found the following documents:")
                    for doc_id, score in results:
                        print(f"{doc_id}: ({score:.17f})")

                else:
                    print("Sorry, I didn't find any documents for this terms.")

def parse_xml_file(file_path):
    """Parse an XML file and extract the required elements."""
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    doc_handler = DocHandler()
    parser.setContentHandler(doc_handler)
    parser.parse(file_path)
    return doc_handler

if __name__ == '__main__':
    '''
    write your code here:
    * load index / start search engine
    * start the loop asking for query terms
    * program should quit if users enters no term and simply hits enter
    '''

    # Example for how we might test your program:
    # Should also work with nyt199501 !
    document_handler = parse_xml_file("./DocumentCollections/nytsmall.xml")
    search_engine = SearchEngine("nytsmall", create=False, handler=document_handler)
    search_engine.execute_query_console()













