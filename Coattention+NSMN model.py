#utilize glove to be the initial word representation
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
input_file = r'glove.6B.50d.txt'
output_file = r'gensim_glove.6B.50d.txt'
glove2word2vec(input_file, output_file)

# Test Glove model
model = KeyedVectors.load_word2vec_format(output_file, binary=False)
