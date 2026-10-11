

from rank_bm25 import BM25Okapi
import re
import pickle
import numpy
from .models import MinimalSource, MinimalSearchResults, StudentSearchResults, RagDataset
from pathlib import Path
import json
import os
from .reader import Reader
from .chunkers.code import CodeChunker
from .chunkers.text import TextChunker
# from .semantic import Semantic_search
from tqdm import tqdm
import uuid
from .bm25 import BM25

class RagPipeline:
    def __init__(self):
        self.reader = Reader("data/raw/vllm-0.10.1")
        self.code_chunker = CodeChunker()
        self.text_chunker = TextChunker()
        self.resurce_paths = []
        self.chunks = {} # {"file_path+first_c": chunk}
        self.tokenized_chunks = []
        self.bm25 = None
        self.max_chunk_size = 0
        self.tokens_freq = {}
        # self.semantic_search = Semantic_search()
            
    def tokenize(self, text):
        return re.split(r"[\W_]+", text.lower())


    def clean_tokenized_chunks(self, chunks):

        cleaned_chunks = []

        with open("src/stop_words.txt", "r") as file:
            stop_words_content = file.read()
        stop_words = [word for word in stop_words_content.split()]

        chunk_id = 1

        for chunk in chunks:
            cleaned_chunk = []

            for token in chunk:

                if len(token) < 3 or not token or token in stop_words:
                    continue
                
                cleaned_chunk.append(token)

            cleaned_chunks.append({"id": chunk_id, "content": cleaned_chunk})
            chunk_id += 1
        return cleaned_chunks


    def chunk(self):

        self.resurce_paths = self.reader.read()

        for file in tqdm(self.resurce_paths, desc=f"chunking resources into chunks with size {self.max_chunk_size}"):

            if file.suffix.lower() == ".py":
                chunks = self.code_chunker.extract_chunks(str(file), self.max_chunk_size)
                for chunk in chunks:
                    self.chunks.update({f"{file}{chunk['first_character_index']}" : chunk})
      
            elif file.suffix.lower() in [".txt", ".md"]:
                chunks = self.text_chunker.extract_chunks(str(file), self.max_chunk_size)
                for chunk in chunks:
                    self.chunks.update({f"{file}{chunk['first_character_index']}" : chunk})


    def build(self):

        chunks_content = [chunk['text'] for chunk in self.chunks.values()]

        tokenized_chunks = [
            self.tokenize(chunk)
            for chunk in chunks_content
        ]

        self.tokenized_chunks = self.clean_tokenized_chunks(tokenized_chunks)

        self.bm25 = BM25(self.tokenized_chunks)
        self.tokens_freq = self.bm25.calculate_token_freq()

        # self.bm25 = BM25Okapi([chunk['content'] for chunk in self.tokenized_chunks])
    
    def save(self):
        
        data  = {
            "files": {
                path: os.path.getctime(path) for path in self.resurce_paths
            },
            "chunks": [MinimalSource(**chunk) for chunk in self.chunks.values()],
            "tokenized_chunks": self.tokenized_chunks,
            "max_chunk_size": self.max_chunk_size,
            "tokens_freq": self.tokens_freq
        }

        Path("data/processed").mkdir(exist_ok=True)

        with open("data/processed/lexical_index", "wb") as file:
            pickle.dump(data, file)
    
    
    def ingest(self, max_chunk_size):
        self.max_chunk_size = max_chunk_size
        self.chunk()
        self.build()
        self.save()
    
    def load(self, path="data/processed/lexical_index"):
        print("loading data from disk ...")

        with open(path, "rb") as file:
            data = pickle.load(file)

        self.chunks = {
            f"{chunk.file_path}{chunk.first_character_index}" :
                {
                    'file_path': chunk.file_path,
                    'first_character_index': chunk.first_character_index,
                    'last_character_index': chunk.last_character_index,
                    'text': chunk.text
                }
            for chunk in data['chunks']
        }
        self.tokenized_chunks = data["tokenized_chunks"]
        # self.bm25 = BM25Okapi([chunk['content'] for chunk in self.tokenized_chunks])
        self.bm25 = BM25(self.tokenized_chunks)
        self.bm25.tokens_freq = data['tokens_freq']



    def search(self, question, k, storing_file="", single=False):
        storing_file = Path(storing_file).name

        tokenized_query = self.tokenize(question if single else question.question)

        scores = self.bm25.get_scores(tokenized_query)

        result = []
        for _ in tqdm(range(min(k, len(self.chunks))), desc=f"querying data '{question}'") if single else range(min(k, len(self.chunks))):
            
            index = numpy.argmax(scores)

            result.append(list(self.chunks.values())[index])

            scores[index] = float("-infinity")
  
        return MinimalSearchResults(
                question_id=question.question_id if not single else str(uuid.uuid4()),
                question=question.question if not single else question,
                retrieved_sources=result,
                storing_file=storing_file
            )
        

    def get_search_result(self, dataset_path, k):

        dataset_folder = Path(dataset_path)
        datasets = dataset_folder.rglob("*.json")

        student_results: list[MinimalSearchResults] = []
        
        for file in datasets:

            with open(str(file), "r") as f:
                questions = RagDataset(**json.load(f))

                for question in tqdm(questions.rag_questions, desc=f"querying data with questions from '{dataset_path}'"):
                    chunks = self.search(question, k, str(file))
                    student_results.append(chunks)

        return StudentSearchResults(search_results=student_results, k=k)


    def save_searching_output(self, result,k, save_directory="data/search_results"):
        """
            dispatcher, saves each result to it's correct distination file (code, docs)
        """

        Path(save_directory).mkdir(exist_ok=True)

        content = {} # {file_path: {search_results: []}}

        for result in tqdm(result.search_results, desc=f"saving search results into {save_directory}"):
            sources = []

            for source in result.retrieved_sources:

                    sources.append({
                        "file_path": source.file_path,
                        "first_character_index": source.first_character_index,
                        "last_character_index": source.last_character_index,
                        "text": source.text
                    })

            try:
                content[result.storing_file]['search_results'].append({
                    "question_id": result.question_id,
                    "question": result.question,
                    "retrieved_sources": sources,
                })
            except KeyError:
                content.update({result.storing_file: {'search_results': [{
                    "question_id": result.question_id,
                    "question": result.question,
                    "retrieved_sources": sources
                    }]
                }})
            content[result.storing_file]["k"] = k

        for file in content.keys():
            with open(f"{save_directory}/{file}", "w") as f:
                json.dump(content[file], f, indent=4)
