# Punkt

[![Build Status](https://travis-ci.org/ferristseng/rust-punkt.svg)](https://travis-ci.org/ferristseng/rust-punkt)
[![](http://meritbadge.herokuapp.com/punkt)](https://crates.io/crates/punkt)

Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence tokenization.
Includes a word tokenizer that tokenizes words based on regexes defined in Python's
NLTK library. Results have been compared with small and large texts that have been
tokenized with NLTK's library.

Note: The output of this library may be slightly different than that of NLTK's implementation 
of the algorithm, due to the fact that the WordTokenizer was developed with performancein mind, and 
avoids the usage of regular expressions. 

## Usage

The library allows you to train the tokenizer yourself or use pre-trained data.

To use pre-trained data from English texts (other languages available):

```rust
extern crate punkt;

use punkt::trainer::{TrainingData, Trainer};
use punkt::tokenizer::SentenceTokenizer;

let doc: &str = ...;
let data = TrainingData::english();

for sent in SentenceTokenizer::new(doc, &data) {
  println!("{:?}", sent);
}
```

The paper describing the Punkt algorithm also states that the tokenizer can learn all of the 
necessary information from the document it is tokenizing. 

To emulate this behavior:

```rust
extern crate punkt;

use std::default::Default;

use punkt::trainer::Trainer;
use punkt::tokenizer::SentenceTokenizer;

let doc: &str = ...;
let mut data = Default::default();

// Trainer requires a mutable reference to data, so it must be instantiated 
// within a block, to release the mutable borrow.
{
  let mut trainer = Trainer::new(&mut data);
  trainer.train(doc);
  trainer.finalize();
}

for sent in SentenceTokenizer::new(doc, &data) {
  println!("{:?}", sent);
}
```

You can also manually train the tokenizer:

```rust
extern crate punkt;

use std::default::Default;

use punkt::trainer::Trainer;
use punkt::tokenizer::SentenceTokenizer;

let docs: Vec<&str> = vec![...];
let mut data = Default::default();

{
  let mut trainer = Trainer::new(&mut data);

  for d in docs.iter() {
    trainer.train(*d);
    trainer.finalize();
  }
}

for sent in SentenceTokenizer::new(doc, &data) {
  println!("{:?}", sent);
}

```

## Benchmarks

Specs of my machine:

  * i5-4460 @ 3.20 x 4
  * 8 GB RAM
  * Fedora 20
  * SSD

```
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_long   ... bench: 150262059 ns/iter (+/- 2273208)
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_medium ... bench:   1032174 ns/iter (+/- 37794)
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_short  ... bench:    814174 ns/iter (+/- 55152)
test tokenizer::word::word_tokenizer_bench_long                             ... bench:  15028316 ns/iter (+/- 266758)
test tokenizer::word::word_tokenizer_bench_medium                           ... bench:    337740 ns/iter (+/- 21585)
test tokenizer::word::word_tokenizer_bench_short                            ... bench:    283756 ns/iter (+/- 18985)
test tokenizer::word::word_tokenizer_bench_very_long                        ... bench:  54302238 ns/iter (+/- 1068666)
test trainer::trainer::bench_trainer_long                                   ... bench:  31247160 ns/iter (+/- 1100315)
test trainer::trainer::bench_trainer_medium                                 ... bench:    758197 ns/iter (+/- 50642)
test trainer::trainer::bench_trainer_short                                  ... bench:    580214 ns/iter (+/- 47159)
test trainer::trainer::bench_trainer_very_long                              ... bench: 108127674 ns/iter (+/- 1459951)
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
long    - 1.3272813240997494 s = 1.32728 x 10^9 ns   ~ 8.83310137524x improvement 
medium  - 0.007431562599958852 s = 7.43156 x 10^6 ns ~ 7.19991009268x improvement
short   - 0.005576989498998349 s = 5.57699 x 10^6 ns ~ 6.84987484248x improvement
```
