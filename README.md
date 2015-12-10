# Punkt

[![Build Status](https://travis-ci.org/ferristseng/rust-punkt.svg)](https://travis-ci.org/ferristseng/rust-punkt)
[![](http://meritbadge.herokuapp.com/punkt)](https://crates.io/crates/punkt)

Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence 
tokenization. Results have been compared with small and large texts that have 
been tokenized using NLTK. 

## Usage

*For full examples, see rust-punkt/examples*

The punkt algorithm allows you to derive all the necessary data to perform 
sentence tokenization from the document itself. 

```rust
let doc = "I bought $5.50 worth of apples from the store. I gave them to my dog when I came home.";
let trainer: Trainer<Standard> = Trainer::new();
let mut data = TrainingData::new();

trainer.train(doc, &mut data);

for s in SentenceTokenizer::<Standard>::new(doc, &data) {
  println!("{:?}", s);
}
```

`rust-punkt` also provides pretrained data that can be loaded for certain languages.

```rust
let data = TrainingData::english();
...
```

`rust-punkt` also allows training data to be incrementally gathered.

```rust
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

*For a full example, see rust-punkt/examples/custom-parameters.rs*

`rust-punkt` exposes a number of traits to customize how the trainer, sentence tokenizer, 
and internal tokenizers work. The default settings, which are nearly identical, to the 
ones available in the Python library are available in `punkt::params::Standard`.

To modify only how the trainer works:

```rust
struct MyParams;

impl DefaultCharacterDefinitions for MyParams { }

impl TrainerParameters for MyParams {
  ...
}
```

To fully modify how everything works:

```rust
struct MyParams;

impl DefinesSentenceEndings for MyParams { 
  ...
}

impl DefinesInternalPunctuation for MyParams {
  ...
}

impl DefinesNonWordCharacters for MyParams { 
  ...
}

impl DefinesPunctuation for MyParams {
  ...
}

impl DefinesNonPrefixCharacters for MyParams {
  ...
}

impl TrainerParameters for MyParams {
  ...
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