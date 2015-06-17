# Punkt

[![Build Status](https://travis-ci.org/ferristseng/rust-punkt.svg)](https://travis-ci.org/ferristseng/rust-punkt)
[![](http://meritbadge.herokuapp.com/punkt)](https://crates.io/crates/punkt)

Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence tokenization.
Results have been compared with small and large texts that have been tokenized with NLTK's library;
although, the results are not guaranteed to be the same because the Rust implementation 
forgoes the usage of regexes, and instead prefers to use custom tokenizers internally.

## Usage

The library allows you to train the tokenizer yourself or use pre-trained data.

To use pre-trained data from English texts (other languages available):

```rust
```

The paper describing the Punkt algorithm also states that the tokenizer can learn all of the 
necessary information from the document it is tokenizing. 

To emulate this behavior:

```rust
```

## Customization

`rust-punkt` exposes a number of traits to customize how the trainer, sentence tokenizer, 
and internal tokenizers work. The default settings, which are nearly identical, to the 
ones available in the Python library are available in `punkt::params::Default`.

## Benchmarks

Specs of my machine:

  * i5-4460 @ 3.20 x 4
  * 8 GB RAM
  * Fedora 20
  * SSD

```
```

Python results for sentence tokenization, and training on the document (the first 3 tests mirrored from above):

The following script was used to benchmark NLTK.

  * `f0` is the contents of the file that is being tokenized.
  * `s` is an instance of a `PunktSentenceTokenizer`.
  * `timed` is the total time it takes to run `tests` number of tests.

*`False` is being passed into `tokenize` to prevent NLTK from aligning sentence boundaries. This functionality 
is currently unimplemented.*

```python
timed = timeit.timeit('s.train(f0); [s for s in s.tokenize(f0, False)]', 'from bench import s, f0', number=tests)
print(timed)
print(timed / tests)
```

```
```
