# Punkt

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
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_long   ... bench: 126307934 ns/iter (+/- 2396747)
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_medium ... bench:    873410 ns/iter (+/- 32674)
test tokenizer::sentence::bench_sentence_tokenizer_train_on_document_short  ... bench:    680775 ns/iter (+/- 21910)
test tokenizer::word::word_tokenizer_bench_long                             ... bench:  10532662 ns/iter (+/- 349243)
test tokenizer::word::word_tokenizer_bench_medium                           ... bench:    223920 ns/iter (+/- 10958)
test tokenizer::word::word_tokenizer_bench_short                            ... bench:    187889 ns/iter (+/- 17145)
test tokenizer::word::word_tokenizer_bench_very_long                        ... bench:  36701973 ns/iter (+/- 1131175)
test trainer::trainer::bench_trainer_long                                   ... bench:  25637329 ns/iter (+/- 794873)
test trainer::trainer::bench_trainer_medium                                 ... bench:    616952 ns/iter (+/- 41141)
test trainer::trainer::bench_trainer_short                                  ... bench:    478615 ns/iter (+/- 29167)
test trainer::trainer::bench_trainer_very_long                              ... bench:  87842418 ns/iter (+/- 1961972)
```

For sentence tokenization, and training on the input document, the Rust implementation runs roughly 10x faster than the Python implementations. I used timeit on the `tokenize` method after calling `train` to benchmark this.
