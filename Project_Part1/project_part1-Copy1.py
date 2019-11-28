## Import Libraries and Modules here...


class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens
        self.tf_entities

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens
        self.idf_entities

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        pass ## Replace this line with your implementation...


    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        pass



    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        pass ## Replace this line with your implementation...
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})
