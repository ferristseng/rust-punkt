# punkt

[![Build Status](https://travis-ci.org/ferristseng/rust-punkt.svg)](https://travis-ci.org/ferristseng/rust-punkt)
[![Crates.io](https://img.shields.io/crates/v/punkt.svg)](https://crates.io/crates/punkt)
[![Docs.rs](https://docs.rs/punkt/badge.svg)](https://docs.rs/punkt/)

## Status

*I am no longer maintaining this library. Please contact me or create an issue if you would like to become a maintainer.*

## Overview

Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence
tokenization. Results have been compared with small and large texts that have
been tokenized using NLTK.

## Training

Training data can be provided to a `SentenceTokenizer` for better
results. Data can be acquired manually by training with a `Trainer`,
or using already compiled data from NLTK (example: `TrainingData::english()`).

## Typical Usage

The punkt algorithm allows you to derive all the necessary data to perform
sentence tokenization from the document itself.

```rust
#
let trainer: Trainer<Standard> = Trainer::new();
let mut data = TrainingData::new();

trainer.train(doc, &mut data);

for s in SentenceTokenizer::<Standard>::new(doc, &data) {
  println!("{:?}", s);
}
```

`rust-punkt` also provides pretrained data that can be loaded for certain languages.

```rust
#
#
let data = TrainingData::english();
```

`rust-punkt` also allows training data to be incrementally gathered.

```rust
#
let trainer: Trainer<Standard> = Trainer::new();
let mut data = TrainingData::new();

for d in docs.iter() {
  trainer.train(d, &mut data);

  for s in SentenceTokenizer::<Standard>::new(d, &data) {
    println!("{:?}", s);
  }
}
```

## Customization

`rust-punkt` exposes a number of traits to customize how the trainer, sentence tokenizer,
and internal tokenizers work. The default settings, which are nearly identical, to the
ones available in the Python library are available in `punkt::params::Standard`.

To modify only how the trainer works:

```rust
#
struct MyParams;

impl DefinesInternalPunctuation for MyParams {}
impl DefinesNonPrefixCharacters for MyParams {}
impl DefinesNonWordCharacters for MyParams {}
impl DefinesPunctuation for MyParams {}
impl DefinesSentenceEndings for MyParams {}

impl TrainerParameters for MyParams {
  const ABBREV_LOWER_BOUND: f64 = 0.3;
  const ABBREV_UPPER_BOUND: f64 = 8f64;
  const IGNORE_ABBREV_PENALTY: bool = false;
  const COLLOCATION_LOWER_BOUND: f64 = 7.88;
  const SENTENCE_STARTER_LOWER_BOUND: f64 = 35f64;
  const INCLUDE_ALL_COLLOCATIONS: bool = false;
  const INCLUDE_ABBREV_COLLOCATIONS: bool = true;
  const COLLOCATION_FREQUENCY_LOWER_BOUND: f64 = 0.8f64;
}
```

To fully modify how everything works:

```rust
#
struct MyParams;

impl DefinesSentenceEndings for MyParams {
  // const SENTENCE_ENDINGS: &'static Set<char> = &phf_set![...];
}

impl DefinesInternalPunctuation for MyParams {
  // const INTERNAL_PUNCTUATION: &'static Set<char> = &phf_set![...];
}

impl DefinesNonWordCharacters for MyParams {
  // const NONWORD_CHARS: &'static Set<char> = &phf_set![...];
}

impl DefinesPunctuation for MyParams {
  // const PUNCTUATION: &'static Set<char> = &phf_set![...];
}

impl DefinesNonPrefixCharacters for MyParams {
  // const NONPREFIX_CHARS: &'static Set<char> = &phf_set![...];
}

impl TrainerParameters for MyParams {
  // const ABBREV_LOWER_BOUND: f64 = ...;
  // const ABBREV_UPPER_BOUND: f64 = ...;
  // const IGNORE_ABBREV_PENALTY: bool = ...;
  // const COLLOCATION_LOWER_BOUND: f64 = ...;
  // const SENTENCE_STARTER_LOWER_BOUND: f64 = ...;
  // const INCLUDE_ALL_COLLOCATIONS: bool = ...;
  // const INCLUDE_ABBREV_COLLOCATIONS: bool = true;
  // const COLLOCATION_FREQUENCY_LOWER_BOUND: f64 = ...;
}
```

## Benchmarks

Specs of my machine:

  * i5-4460 @ 3.20 x 4
  * 8 GB RAM
  * Fedora 20
  * SSD

```
test tokenizer::bench_sentence_tokenizer_train_on_document_long   ... bench: 129,877,668 ns/iter (+/- 6,935,294)
test tokenizer::bench_sentence_tokenizer_train_on_document_medium ... bench:     901,867 ns/iter (+/- 12,984)
test tokenizer::bench_sentence_tokenizer_train_on_document_short  ... bench:     702,976 ns/iter (+/- 13,554)
test tokenizer::word_tokenizer_bench_long                         ... bench:  14,897,528 ns/iter (+/- 689,138)
test tokenizer::word_tokenizer_bench_medium                       ... bench:     339,535 ns/iter (+/- 21,692)
test tokenizer::word_tokenizer_bench_short                        ... bench:     281,293 ns/iter (+/- 3,256)
test tokenizer::word_tokenizer_bench_very_long                    ... bench:  54,256,241 ns/iter (+/- 1,210,575)
test trainer::bench_trainer_long                                  ... bench:  27,674,731 ns/iter (+/- 550,338)
test trainer::bench_trainer_medium                                ... bench:     681,222 ns/iter (+/- 31,713)
test trainer::bench_trainer_short                                 ... bench:     527,203 ns/iter (+/- 11,354)
test trainer::bench_trainer_very_long                             ... bench:  98,221,585 ns/iter (+/- 5,297,733)

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
long    - 1.3414202709775418 s   = 1.34142 x 10^9 ns ~ 10.3283365927x improvement 
medium  - 0.007250561956316233 s = 7.25056 x 10^6 ns ~ 8.03950245027x improvement
short   - 0.005532620595768094 s = 5.53262 x 10^6 ns ~ 7.870283759x   improvement
```

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
