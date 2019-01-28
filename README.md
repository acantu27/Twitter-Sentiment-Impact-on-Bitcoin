# Twitter Sentiment Impact on Bitcoin Pricing
Natural Language Processing <sup>[1][1]</sup> (NLP) used in conjunction with a Support Vector Machine <sup>[2][2]</sup> 
(SVM) to classify the sentiment of tweets to determine a correlation with Bitcoin prices.

*Note: Not a full/complete project*

## General Information
Preprocessing is handled by utilizing NLP techniques provided by the Natural Language Toolkit<sup>[3][3]</sup> (NLTK) to 
normalize textual data. Textual data is then converted into a Vector Space Model (VSM) with Term Frequency-Inverse 
Document Frequency<sup>[4][4]</sup> (TF-IDF). A SVM is then used for binary classification of 
tweets to determine positive and non-positive sentiment of tweets. Tweets can be acquired using [TweetStreamer](https://github.com/acantu27/TweetStreamer).

## In Progress
- [x] Revised Stop Word Removal
- [ ] Negation Handling
- [ ] Emoji Support and Scoring

## References
1. [Natural Language Processing Wikipedia](https://en.wikipedia.org/wiki/Natural_language_processing)
2. [Support Vector Machine](https://google.com)
3. [Natural Language Toolkit Official](https://www.nltk.org/)
4. [TF-IDF](http://www.tfidf.com/)
5. [Sentiment 140 Training](http://help.sentiment140.com/for-students)

[1]: https://en.wikipedia.org/wiki/Natural_language_processing
[2]: https://google.com
[3]: https://www.nltk.org/
[4]: http://www.tfidf.com/
