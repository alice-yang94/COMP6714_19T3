## Import Libraries and Modules here...
from collections import defaultdict
import spacy
from math import log
import itertools 

class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        # {key: token/entity, value: {key: doc_id, value: normalised term frequency(token/entity, doc_id)}}
        self.tf_tokens = defaultdict(dict)
        self.tf_entities = defaultdict(dict)

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        # {key: token/entity, value: IDF(token/entity)}
        self.idf_tokens = {}
        self.idf_entities = {}
        
        # load the english language model
        self.nlp = spacy.load("en_core_web_sm")

        
    # count term frequencies for entities and tokens in the documents       
    def _count_tf(self, documents):
        # tf_count - key: term, value: {key: doc_id, value: # of this term in this doc_id}
        entity_tf_count = defaultdict(lambda:defaultdict(int))
        token_tf_count = defaultdict(lambda:defaultdict(int))

        # documents: key - doc_id; value: document_text
        for doc_id in documents.keys():
            doc_text = self.nlp(documents[doc_id])

            # decide to calculate entities first, because single word entities 
            #should not be counted as tokens, if tokens calculated first, extra deleting
            #cost to token list
            single_word_entities = defaultdict(list)

            # count tf for entities
            for ent in doc_text.ents:
                entity_tf_count[ent.text][doc_id] += 1

                # if entity is a single word
                if len(ent.text.split()) == 1:
                    single_word_entities[ent.text].append(ent.start_char)

            # count tf for tokens       
            for token in doc_text:
                if not token.is_stop and not token.is_punct:
                    is_single_entity = False
                    # ent_iob - 3: begin entity, 2: outside, 1: inside
                    if (token.ent_iob == 1 or token.ent_iob == 3) \
                      and token.text in single_word_entities.keys() \
                      and token.idx in single_word_entities[token.text]:
                        is_single_entity = True

                    if not is_single_entity:
                        token_tf_count[token.text][doc_id] += 1
        
        return entity_tf_count, token_tf_count
    
    # calculate tf and idf of tokens
    def _calc_tf_idf_token(self, token_tf_count, total_doc_no):
        for token in token_tf_count.keys():
            # calculate tf
            for doc_id in token_tf_count[token]:
                tf_token = token_tf_count[token][doc_id]
                # calculate normalised token tf
                self.tf_tokens[token][doc_id] = 1.0 + log(1.0 + log(tf_token))

            # calculate token idf
            doc_contain_token = len(token_tf_count[token])
            self.idf_tokens[token] = 1.0 + log(total_doc_no / (1.0 + doc_contain_token))
        return

    # calculate tf and idf of entities
    def _calc_tf_idf_entity(self, entity_tf_count, total_doc_no):
        for ent in entity_tf_count:
            # calculate tf
            for doc_id in entity_tf_count[ent]:
                tf_ent = entity_tf_count[ent][doc_id]

                # calculate normalised entity tf
                self.tf_entities[ent][doc_id] = 1.0 + log(tf_ent)

            # calculate entity idf
            doc_contain_ent = len(entity_tf_count[ent])
            self.idf_entities[ent] = 1.0 + log(total_doc_no / (1.0 + doc_contain_ent))
        return    
        
    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        total_doc_no = len(documents)
        
        # get term frequencies
        entity_tf_count, token_tf_count = self._count_tf(documents)
        
        # use tf to calculate normalised tf and idf
        self._calc_tf_idf_token(token_tf_count, total_doc_no)
        self._calc_tf_idf_entity(entity_tf_count, total_doc_no)

      
    # enumerate all combinations of tokens in Q and filter by DoE
    # (2. step 1)
    def _get_entities(self, tokens, DoE):
        # token_count: {key: token, value: count} count no. of tokens in query
        token_count = defaultdict(int)

        for token in tokens:
            token_count[token] += 1

        # generate an entities: list of combinations of tokens         
        possible_entities = []
        for ent_size in range(1, len(tokens) + 1):
            for indicies in itertools.combinations(range(len(tokens)), ent_size):
                ids = list(indicies)
                ent = [tokens[i] for i in ids]
                
                possible_entities.append(' '.join(ent))

        entities = [curr_doe for curr_doe in DoE.keys() if curr_doe in possible_entities]

        return entities, token_count

    # True if exceeded token count, else False
    # (2. step 3)
    def _exceed_token_count(self, subset_ents, token_count):
        # sub_token_count: {key: token, value: count} 
        # count no. of tokens in this subset
        sub_token_count = defaultdict(int)

        for ent in subset_ents:
            tokens = ent.split()
            for token in tokens:
                sub_token_count[token] += 1
                if sub_token_count[token] > token_count[token]:
                    return True
        return False
    
    # enumerate all possible subsets of entities selected, filter by token count
    # (2. step 2/3)
    def _get_subsets_ents(self, entities, token_count):
        subsets_ents = []
        for i in range(len(entities) + 1):
            for comb_subset_ents in itertools.combinations(entities, i):
                subset_ents = list(comb_subset_ents)

                if len(subset_ents) <= 1 or \
                  (len(subset_ents) > 1 and not self._exceed_token_count(subset_ents, token_count)):
                    subsets_ents.append(subset_ents)

        return subsets_ents

    # (2. step 4): split according to each filtered entity subset
    def _create_splits(self, subsets_ents, tokens):
        ents_and_tokens = defaultdict(dict)
        key = 0

        for curr_ents in subsets_ents:
            curr_tokens = tokens.copy()
            for ent in curr_ents:
                words = ent.split()
                for word in words:
                    curr_tokens.remove(word)

            ents_and_tokens[key]['tokens'] = curr_tokens
            ents_and_tokens[key]['entities'] = curr_ents
            key += 1

        return ents_and_tokens           
    
    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        # query Q: a string of words; DoE: key - entity, value - entity_id
        tokens = Q.split()
        entities, token_count = self._get_entities(tokens, DoE)    
        
        subsets_ents = self._get_subsets_ents(entities, token_count)
        ents_and_tokens = self._create_splits(subsets_ents, tokens)
        return ents_and_tokens


    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})
        max_score = -1
        max_score_split = {}
        for key, split in query_splits.items():
            curr_tokens = split['tokens']
            curr_ents = split['entities']
            scores = {}

            score_token = 0.0
            for token in curr_tokens:
                if token in self.tf_tokens.keys() and doc_id in self.tf_tokens[token].keys():
                    score_token += self.tf_tokens[token][doc_id] * self.idf_tokens[token]
            scores['tokens_score'] = score_token
            
            score_entity = 0.0
            for ent in curr_ents:
                if ent in self.tf_entities.keys() and doc_id in self.tf_entities[ent].keys():
                    score_entity += self.tf_entities[ent][doc_id] * self.idf_entities[ent]
            scores['entities_score'] = score_entity

            scorei = score_entity + 0.4 * score_token
            scores['combined_score'] = scorei
            if scorei > max_score:
                max_score = scorei
                max_score_split = split
            
        return (max_score, max_score_split)
                
                
                
                
                
                
                
                
                
                
                
                
                
   
