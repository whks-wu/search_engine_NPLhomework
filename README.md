This is a programming homework assignment, when I was in my second semester of Computer Linguistics. 
This little assignment was a search engine written in python that searched for information in a given corpus

# Search Engine Project

This project implements a basic search engine in Python, capable of indexing a collection of XML documents, calculating TF-IDF scores for terms, and executing queries to find relevant documents. It emphasizes efficiency in data processing and storage for better performance.

## Project Description

This search engine processes a collection of documents in a custom XML format. It extracts relevant text content (headlines and paragraphs), performs text cleaning and stemming, and then builds an inverted index using the Term Frequency-Inverse Document Frequency (TF-IDF) model. The index can be persisted to and loaded from disk for efficient repeated use. Users can interactively query the engine through a console interface.

## Features

* **XML Document Parsing:** Utilizes the SAX parser to efficiently extract document IDs, headlines, and paragraph content from XML files.
* **Text Pre-processing:**
    * Removes punctuation and converts text to lowercase.
    * Applies Porter2 stemming to normalize words (e.g., "running" becomes "run").
* **TF-IDF Indexing:**
    * Calculates **Term Frequency (TF)** for each term within documents, using a normalized approach (term count divided by the frequency of the most frequent term in the document).
    * Calculates **Inverse Document Frequency (IDF)** for each term across the entire document collection.
    * Uses Python's `Decimal` type for all TF-IDF calculations to ensure high numerical precision, preventing floating-point inaccuracies.
* **Efficient Index Storage:**
    * IDF values are stored in a list, directly indexed by term ID for fast lookup.
    * TF values are stored in a sparse list of lists (list of `(term_idx, tf_value)` tuples for each document), optimizing memory usage for large vocabularies.
    * Uses a two-pass approach for IDF calculation to efficiently determine document frequencies.
* **Index Persistence:** The computed TF and IDF indices are saved to `.tf` and `.idf` files, respectively, allowing the search engine to load pre-built indices without re-processing documents every time.
* **Query Execution:**
    * Processes user queries by stemming terms.
    * Calculates TF-IDF scores for query terms and documents.
    * Ranks documents based on relevance (e.g., using cosine similarity, though the specific similarity metric isn't explicitly defined in the provided snippets, TF-IDF scores are the foundation).
* **Console Interface:** Provides a simple command-line interface for users to enter queries and view ranked results.

## How to Run

1.  **Dependencies:** Ensure you have the `stemming` library installed.
    ```bash
    pip install stemming
    ```
2.  **Document Collection:** Place your XML document collection file (e.g., `nytsmall.xml`) in the `DocumentCollections/` directory relative to your script.
3.  **Run the script:**
    ```bash
    python softwareAssignment.py
    ```
    The script will first parse the XML, build/load the index, and then enter a loop, prompting you to enter query terms.

### Example Usage:

```
Please enter query, terms separated by whitespace: cat dog
I found the following documents:
doc2: (0.12345678901234567)
doc1: (0.09876543210987654)
Please enter query, terms separated by whitespace: (press Enter to exit)
```

## Key Concepts and Implementation Details

### `DocHandler` Class

* Extends `xml.sax.ContentHandler` to process XML events.
* Uses internal flags (`self.in_headline`, `self.in_p`, etc.) to track the current XML element being parsed.
* `self.current_data`: A temporary buffer that collects character data chunks between tags. It's crucial for correctly assembling text content for a single tag and is **reset to `""`** at the beginning of every `startElement` call to ensure content from different tags (or documents) doesn't mix.
* `self.results`: Stores the parsed documents in a structured dictionary format, ready for the `SearchEngine`.

### `SearchEngine` Class

* **`__init__(self, collection_name, create, handler)`:**
    * **Core Idea:** Sets up the search engine's internal state.
    * **`create=True`**: Triggers the index building process. This involves:
        * **Vocabulary Creation:** Collects all unique stemmed terms from the entire document collection and stores them in `self.vocabulary` (a sorted list) and `self.term_to_idx` (a dictionary mapping terms to their integer IDs).
        * **IDF Calculation (Two-Pass):**
            1.  **Pass 1 (`doc_term_presence`):** Creates a list of sets, where `doc_term_presence[i]` contains the `term_idx` for every unique term found in document `i`. This uses sets for efficient uniqueness and lookup.
            2.  **Pass 2:** Iterates through `self.vocabulary`. For each term, it counts its Document Frequency (how many documents contain it) by checking against the `doc_term_presence` sets. IDF is then calculated using `Decimal.ln(total_documents / document_frequency)`.
        * **TF Calculation:** For each document, it counts term occurrences, calculates a normalized TF (`count / max_occurrence_in_doc`), and stores these as `(term_idx, tf_value)` tuples in `self.tf_data[doc_idx]`.
        * **Persistence:** Writes calculated IDF and TF values to `.idf` and `.tf` files respectively.
    * **`create=False`**: Loads the pre-computed index from the `.idf` and `.tf` files into the `self.vocabulary`, `self.idf_values`, `self.tf_data`, and `self.doc_ids` structures.

* **`Decimal` Type:** Used extensively throughout the `SearchEngine` for all TF-IDF calculations. This ensures high numerical precision and avoids floating-point errors, which is critical for accurate document ranking. `getcontext().prec = 18` sets the precision to 18 decimal places.
