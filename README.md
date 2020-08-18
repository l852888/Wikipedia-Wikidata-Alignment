# Wikipedia-Wikidata-Alignment
This repository is used to store the model of wiki project code.
-------------------------------------------------------------
There is not much research about the consistency between Wikidata and Wikipedia pages. Moreover, there is no solution to 
compare such Wikipedia content with Wikidata information in the past. Our goals is to create a mapping between Wikidata
and Wikipedia content, and try to measure the consistency between both of them. In this page, we utilize the NSMN+Coattention
model to discover the consistency. This is a supervised learning method.

The input is one sentence and one wikidata claim, and the output label is consistent or not.
<img src="https://github.com/l852888/Wikipedia-Wikidata-Alignment/blob/master/framework/framework.jpg" width="75%" height="75%">

Data preparation.py
-----------------------------
Do some data preparation first (utilize GloVe wrod embedding ,drop stopwords and stemming) for putting into the model.

Co-attention+NSMN.py
---------------------------
The model for dealing with our task, including model training and calculate evaluation metrics.
