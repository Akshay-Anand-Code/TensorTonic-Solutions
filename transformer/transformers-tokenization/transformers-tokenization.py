import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # Initialize vocab with special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            
        self.vocab_size = len(special_tokens)
    
    def build_vocab(self, texts: List[str]) -> None:
        """Builds vocabulary by splitting text and assigning IDs to new words."""
        for text in texts:
            words = text.split()
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = self.vocab_size
                    self.id_to_word[self.vocab_size] = word
                    self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """Converts words to IDs, using the <UNK> ID for missing words."""
        words = text.split()
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """Converts IDs back to words and joins them with spaces."""
        return " ".join([self.id_to_word.get(i, self.unk_token) for i in ids])

# Example Usage:
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(["hello world"])

print(f"Encoded 'hello world': {tokenizer.encode('hello world')}") 
# Output: [4, 5] (assuming PAD=0, UNK=1, BOS=2, EOS=3)

print(f"Encoded 'hello unknown': {tokenizer.encode('hello unknown')}") 
# Output: [4, 1]