Doc2Vec + KNN Text Classification

**1. Abstract**

This paper demonstrates that highly accurate text classification may be obtained through word embedding representations of documents in combination with a K-Nearest Neighbor algorithm. This study found that document embeddings with a Distributed Bag of Words model in combination with KNN not only scores higher than other TF-IDF vector KNN approaches, but rivaled the dominant SVM approach in terms of accuracy.

_Keyterms: Keyword Extraction, Text Classification, Word Embedding, Word2Vec, Term Frequency – Inverse Document Frequency, Doc2Vec, K-Nearest Neighbor_

**2. Introduction**

Word embeddings capture many semantic relationships and meanings from text. Word vectors quantify a word based on its relationship. Proximity, and usage relative to other words.

Unfortunately, word vectors do not capture the specific context of the word at the instance of usage. For example, without proper context, _chicken_ could refer to the literal animal, a coward, or dinner. But through the use of _document embeddings_, or _paragraph embeddings_, the entire document may be represented as a word embedding defined by all the words within the document. This would mean a word embedding with a specific and narrow context and setting as opposed to a general word embedding capturing all possible relationships.12

![alt text](https://github.com/ross-abramson/Doc2Vec-Classification/blob/master/readme-images/Doc-Vec-Class-Struggle.png)


   
   Figure 1: The above graph shows several word embedded vectors found in &quot;The Tale of Two Cities.&quot; Each individual word has its own individual vector direction. When averaged together, we get one singular vector representative of the entire document. In this case, that average vector resembles &quot;Class Struggle.&quot; 3



**2. Related Work**

Many authors have already incorporated and explored the use of word embeddings for text classification. In &quot;Bag-Of-Embeddings for Text Classification,&quot; the authors were able to implement a multi-prototype word embedding based on text classes on the Reuters news dataset and were able to achieve accuracy scores relatively close to Support Vector Machine models (SVM).4 Specific to this dataset, few authors have published results. In &quot;Text Classification&quot; by Nura Kawa, Kawa developed logistic regression, SVM, and KNN models for text classification using TF-IDF vectors, bag-of-words, and N-grams.5 Kawa was able to achieve above 90% accuracy for most of his models, with the exception of KNN which wasn&#39;t able to bypass 50% accuracy. In &quot;Comparison of Text Classifiers on News Articles,&quot; the authors used the BBC News set and achieved 97% with SVM, but only 88% with KNN-- Again utilizing TF-IDF vectorization only this time with more data preprocessing and feature selection techniques such as Chi-Square selection.

**3. Dataset: BBC**

The dataset used was the BBC news dataset. This dataset consists of 2225 short news articles for five categories from 2004-2005: &#39;Business,&#39; &#39;Sport,&#39; &#39;Politics,&#39; &#39;Entertainment,&#39; and &#39;Tech.&#39; The dataset was chosen for its simplicity of only five categories and because the the categories &#39;naturally&#39; feel different (i.e. sport and technology typically wouldn&#39;t be clustered together by a human annotator). Additionally, this dataset has been relatively underused. It should be noted that the articles are not evenly distributed (Ex: sport and business articles form roughly 46% of the dataset. This is relevant more so for when introducing KNN because if a certain article type is represented more, it may skew the algorithm to label articles as sport or business.



**4.**  **Preprocessing Phase**

First the user will receive all data from the BBC news articles. The articles are labeled in separate folders. The user goes through and labels all folders as falling under one of the five pre-assigned categories: Business, Sports, Politics, Entertainment, and Tech. The data is then preprocessed by removing all stop words, utilizing Named Entity Recognition, stemming words, and sending all letters to lowercase. Lastly, the words are POS tagged.

**5 Evaluating The Challenges involved in Word Embeddings**

As mentioned before, word embedding&#39;s can be incredibly effective, yet they still perform significantly worst than the typical SVM and LDA models which can achieve in some cases more than 90% correctness. The reasons and issues further explained are: the algorithm may pick up on connections not considered with human annotators and the topic of a documents can be ambiguous or have multiple categories.

**Facebook is business, technology, and entertainment: ****Most News Documents aren&#39;t just one category.**

The issue with pinpointing the correct category is that a category could have a justifiably equal connection to multiple categories. Take for example this title from one document: &quot;Ink helps drive democracy in Asia.&quot; The remainder of that article talked about the relationship of ink and the coming election in a certain area of Asia. The majority of readers would inherently think the category of the article is politics, but the article is actually talking about the technology of ink and thus is labeled as &quot;tech.&quot; Esoteric category labeling was found abundantly within the news corpus. To visualize the space of these documents, Gensim&#39;s Doc2Vec library was used along with PCA and K-Means. The results show that it may be possible to use a KNN approach of labeling new incoming documents which place on the outer rims of the cluster. But for those documents which fall closer to the center, the model would struggle to solve.

 ![alt text](https://github.com/ross-abramson/Doc2Vec-Classification/blob/master/readme-images/PCA-char-labels.png)


    Figure 3: PCA Representation with single character labels. Notice the central overlap.

 ![alt text](https://github.com/ross-abramson/Doc2Vec-Classification/blob/master/readme-images/PCA-Cluster-coloring.png)


   Figure 4: K-means of PCA Representation. Centroids are darkened circular centers. Notice how the center overlaps.

**6. The Model: Document Embedding + KNN**

Through vectorizing the entire document into a word embedding space, we capture the context and theme of the entire document as opposed to individual word vectors. The unlabeled document embedded vector is then mapped to a graph along with other pre-categorized documents. The unlabeled document takes on the label of the K nearest labeled documents. This is the k-nearest neighbor model with document vectors as inputs.

The results show a viable KNN classification approach that beats the traditional KNN utilizing TF-IDF vector algorithm on the same BBC dataset.

 ![alt text](https://github.com/ross-abramson/Doc2Vec-Classification/blob/master/readme-images/Algorithm-BPMN.png)

     Figure 5: A higher level overview of the algorithm

**6.1 KNN Overview.**

KNN functions by first plotting several pre-labeled inputs onto an N-dimensional graph. An unlabeled input will be plotted within the graph. We check for its closest K neighbors. The label of the unmarked input will be whatever is found to be the majority category amongst the closest K neighbors.

First the documents are vectorized using Gensim&#39;s Doc2Vec with a Bag-Of-Word model. After modeling of the documents is finished, 90 percent of the articles are randomly chosen to be used as the training set and the remaining 10 percent are the test set. The test set is then labeled according to the KNN algorithm. The document model uses a vector size of 100 dimensions, a window size of 10, and is trained over 100 iterations with a minimum word frequency of 1 to capture all words.

**6.3 Computing Average Top Accuracy**

The program took 1000 randomly chosen training and test sets and computed the average of the accuracy given a k. The best results were found to be around K=35 and K=105 with an accuracy score, Precision, Recall, and F1-Score averaging about 95%

 ![alt text](https://github.com/ross-abramson/Doc2Vec-Classification/blob/master/readme-images/Accuaracy-average-graph.png)


     Figure 6: Graph of average accuracy score per given k. Average accuracy was computed over a 1000 iterations of randomly selected testing and train sets. Accuracy is the worst when searching for the first closest neighbors at K=1 and peaks at K =35 and 105.

**7. Results of KNN Approach**

The KNN scored on average more than 94% accuracy amongst the top 120 K scores. The best K values were found to be 35 and 105, both achieving more than 95% accuracy, and, in some cases, even 98%. The highest scoring Recall and F1-Score was found to be Sports documents. This would make sense as sports seems, on face value, to share the least in common with the other four categories. Sports was also found to be the highest scoring category within those articles. Technology conversely generally scored the worst in recall and F1-Score.



_Word Embedding KNN Model Compared to Others_

| **Model** | **Accuracy** |
| --- | --- |
| TF-IDF + N-Gram + BoW KNN | 49% |
| TF-IDF + Feature Selection KNN | 88.5509% |
| **Document Embedding BoW + KNN** |  **95.5157%** |
| SVM | 97.6744% |



 ![alt text](https://github.com/ross-abramson/Doc2Vec-Classification/blob/master/readme-images/Precision-Recall-F1.png)


     Figure 7: Precision, Recall, and F1 Score per each article type. The left most column is the first character of each category. Support is the number of article types per test set. The bottom row shows an average of 95% amongst the three.

**8. KNN Error Analysis**

First, the core issue with document classification is that some articles are highly ambiguous. It may be that an article shares similarities to both business and politics. The KNN approach has removed most of that error but not entirely.


Second, the algorithm is highly dependent upon the initial training set. Take for example the following two confusion matrices. Both tables utilize a K value of 35, but one scored much better than the other, nearing a 98% correct rate. The reason is solely due to the randomness of what documents are chosen to be the training and test set.

![alt text](https://github.com/ross-abramson/Doc2Vec-Classification/blob/master/readme-images/Accuracy-Issue1.png)
![alt text](https://github.com/ross-abramson/Doc2Vec-Classification/blob/master/readme-images/Accuracy-Issue-2.png)

**9. Conclusion and Future Work**

This paper has shown the advantage of using KNN and Document Embeddings as opposed to the traditional TF-IDF vector approach. While not yet outperforming SVM, this approach still comes within striking distance. The next phases of this algorithm will incorporate other datasets commonly used, specifically the Reuters dataset, for comparison with other mainstream algorithms. Additionally, I will introduce another metric of comparison between neighbors found and the test article, such as weighing closer neighbors as being more important than farther away neighboring articles.



**11. Work Cited**

1

#
 Kusner, Sun, Kolkin, and Weinberger. &quot;From Word Embeddings to Document Distances&quot; Washington University in St. Louis. [http://proceedings.mlr.press/v37/kusnerb15.pdf](http://proceedings.mlr.press/v37/kusnerb15.pdf)

2

#
 Le, Quoc and Mikilov, Tomas. &quot;Distributed Representations of Sentences and Documents&quot; Google. [https://cs.stanford.edu/~quocle/paragraph\_vector.pdf](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

3

#
 Tamir, Mike Phd. &quot;Short Introduction to Using Word2Vec for Text Classification. LinkedIn. [https://www.linkedin.com/pulse/short-introduction-using-word2vec-text-classification-mike/](https://www.linkedin.com/pulse/short-introduction-using-word2vec-text-classification-mike/)

4

#
 Jin, Zhang, Chen, and Xia. &quot;Bag-of-Embeddings for Text Classification&quot;. Joint Conferce on Artificial Intelligence. [https://www.ijcai.org/Proceedings/16/Papers/401.pdf](https://www.ijcai.org/Proceedings/16/Papers/401.pdf)

5

#
 Kawa, Nura. &quot;Text Classification&quot; University of California, Berkley. [https://www.stat.berkeley.edu/~aldous/Research/Ugrad/Nura\_Kawa\_report.pdf](https://www.stat.berkeley.edu/~aldous/Research/Ugrad/Nura_Kawa_report.pdf)
