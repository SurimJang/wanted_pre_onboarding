<h1 align="center"> 원티드 프리온보딩 코스 AI/ML 사전과제 </h1>

## About The Project
This is a pre-submission task for the Wanted Pre-Onboarding AI/ML course. It contains following two class implementations:

* `Tokenizer` :  Builds a vocabulary dictionary by tokenizing words and returns an integer index value.
* `TfidfVectorizer` :  Returns the TF-IDF calculated through the learned vocabulary as a list.

## Prerequisites
The following open source packages are used in this project:

* `Numpy`
* `re`
* `string`

## Examples

* `Tokenizer`
```Python
  >>> import Tokenizer
  >>> sequences = [
  ...              "I like this movie, it's funny.",
  ...              "I hate this movie.", 
  ...              "This was awesome! I like it.",
  ...              "Nice one. I love it."
                  ]
  >>> tokenizer = Tokenizer()
  >>> X = tokenizer.fit_transform(sequences)
  >>> print(tokenizer.word_dict)
  ...   {'oov': 0, 'i': 1, 'like': 2, 'this': 3, 'movie': 4, 'its': 5, 'funny': 6, 'hate': 7, 'was': 8, 'awesome': 9, 'it': 10, 'nice': 11, 'one': 12, 'love': 13}
  >>> print(X) 
  ...   [[1, 2, 3, 4, 5, 6], [1, 7, 3, 4], [3, 8, 9, 1, 2, 10], [11, 12, 1, 13, 10]]
```

* `TfidfVectorizer`
```Python
  >>> import TfidfVectorizer
  >>> sequences = [
  ...              "I like this movie, it's funny.",
  ...              "I hate this movie.", 
  ...              "This was awesome! I like it.",
  ...              "Nice one. I love it."
                  ]
  >>> tokenizer = Tokenizer()
  >>> tfidfvectorizer = TfidfVectorizer(tokenizer)
  >>> X = tfidfvectorizer.fit_transform(sequences)
  >>> print(tfidfvectorizer.idf)
  ...   [-0.2231435513142097, 0.28768207245178085, 0.0, ..., 0.6931471805599453, 0.6931471805599453, 0.6931471805599453]
  >>> print([[round(num,2) for num in lst] for lst in X])
  ...   [[-0.22, 0.29, 0.0, ..., nan, nan, nan], ..., [-0.22, nan, nan, nan, ..., 0.69, 0.69, 0.69]]
```
