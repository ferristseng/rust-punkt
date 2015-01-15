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

```
TODO
```