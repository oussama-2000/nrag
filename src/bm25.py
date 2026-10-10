"""
    bm25 (best matching 25)
    is a ranking algorithm used to determine how relevant a document is to ag given search query.
    it's an impoved version of the traditional TF-IDF
"""
# tokenized_chunks = [
#     {"id": 9, "content": ["python", "is", "easy", "python", "is", "powerful"]},
#     {"id": 0, "content": ["rag", "is", "a", "ai", "system"]},
#     {"id": 1, "content": ["retrival", "augmented", "generation"]},
#     {"id": 2, "content": ["how", "rag", "works", "?"]},
#     {"id": 3, "content": ["generation", "step"]},
# ]

import math

class BM25:
    def __init__(self, chunks):

        self.chunks = chunks

        self.k1 = 1.5

        # number of chunks 
        self.corpus_size = len(self.chunks)

        # avg chunk lenght in chunks
        self.avgcl = 0

        # chunks containing each token
        self.token_freq = {} # {token: [chunks id containing the token]}

        self.calculate_avgcl()
        # self.calculate_token_freq()
    
    def calculate_avgcl(self):
        sum_lengs = sum([len(chunk['content']) for chunk in self.chunks])
        self.avgcl =  sum_lengs / len(self.chunks)


    # def calculate_token_freq(self):
    #     for chunk in self.chunks:
    #         for token in chunk['content']:
    #             self.token_freq.setdefault(token, []).append(chunk['id'])


    def TF(self, token, chunk):

        # freq = self.token_freq.get(token, []).count(chunk['id'])
        freq = self.token_freq.get(token, {}).get(chunk['id'], 0)
        d = len(chunk['content'])

        part2 = freq + self.k1 * (
                    (d / self.avgcl)
                )
        
        return freq / part2 if part2 else 0

    def IDF(self, token):
        N = len(self.chunks)
        nt = len(set(self.token_freq.get(token, [])))

        return math.log(1 + ((N - nt + 0.5) / (nt + 0.5)))

    def score(self, query, chunk):
        score = 0
        
        for token in query:
            score += self.IDF(token) * self.TF(token, chunk)
        return score

    def get_scores(self, query):
        scores = []

        for chunk in self.chunks:
            scores.append(self.score(query, chunk))

        return scores
        
# bm25 = BM25(tokenized_chunks)


# scores = bm25.get_scores(["what", "is", "rag"])


# for _ in range(min(3, len(scores))):
#     big_score_index = scores.index(max(scores))
#     print(tokenized_chunks[big_score_index])
#     scores[big_score_index] = float("-inf")


# from rank_bm25 import BM25Okapi
# corpus = [chunk['content'] for chunk in tokenized_chunks]
# Bm = BM25Okapi(corpus)

# print(Bm.get_scores(["what", "is", "rag"]))