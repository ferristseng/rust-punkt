# Punkt

[![Build Status](https://travis-ci.org/ferristseng/rust-punkt.svg)](https://travis-ci.org/ferristseng/rust-punkt)

Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence tokenization.
Includes a word tokenizer that tokenizes words based on regexes defined in Python's
NLTK library. Results have been compared with small and large texts that have been
tokenized with NLTK's library.

Note: The output of this library may be slightly different than that of NLTK's implementation 
of the algorithm, due to the fact that the WordTokenizer was developed with speed in mind, and 
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
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_long   ... bench: 130466365 ns/iter (+/- 6549054)
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_medium ... bench:    927021 ns/iter (+/- 60709)
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_short  ... bench:    698511 ns/iter (+/- 55804)
test tokenizer::word::word_tokenizer_bench_long                             ... bench:  10334680 ns/iter (+/- 506852)
test tokenizer::word::word_tokenizer_bench_medium                           ... bench:    216752 ns/iter (+/- 14028)
test tokenizer::word::word_tokenizer_bench_short                            ... bench:    184602 ns/iter (+/- 14329)
test tokenizer::word::word_tokenizer_bench_very_long                        ... bench:  35592132 ns/iter (+/- 1385917)
test trainer::trainer::bench_trainer_long                                   ... bench:  26580077 ns/iter (+/- 1304612)
test trainer::trainer::bench_trainer_medium                                 ... bench:    637396 ns/iter (+/- 49960)
test trainer::trainer::bench_trainer_short                                  ... bench:    488168 ns/iter (+/- 9631)
test trainer::trainer::bench_trainer_very_long                              ... bench:  89846860 ns/iter (+/- 2098575)
```

For sentence tokenization, and training on the input document, the Rust implementation runs roughly 10x faster than the Python implementations. I used timeit on the `tokenize` method after calling `train` to benchmark this.
