# Punkt

[![Build Status](https://travis-ci.org/ferristseng/rust-punkt.svg)](https://travis-ci.org/ferristseng/rust-punkt)
[![](http://meritbadge.herokuapp.com/punkt)](https://crates.io/crates/punkt)

Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence 
tokenization. Results have been compared with small and large texts that have 
been tokenized with NLTK's library. 

## Usage

*For full examples, see rust-punkt/examples*

The punkt algorithm allows you to derive all the necessary data to perform 
sentence tokenization from the document itself. 

```rust
let doc = "I bought $5.50 worth of apples from the store. I gave them to my dog when I came home.";
let trainer: Trainer<Default> = Trainer::new();
let mut data = TrainingData::new();

trainer.train(doc, &mut data);

for s in SentenceTokenizer::<Default>::new(doc, &data) {
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
let trainer: Trainer<Default> = Trainer::new();
let mut data = TrainingData::new();

for d in docs.iter() {
  trainer.train(d, &mut data);

  for s in SentenceTokenizer::<Default>::new(d, &data) {
    println!("{:?}", s);
  }
}
```

## Customization

*For a full example, see rust-punkt/examples/custom-parameters.rs*

`rust-punkt` exposes a number of traits to customize how the trainer, sentence tokenizer, 
and internal tokenizers work. The default settings, which are nearly identical, to the 
ones available in the Python library are available in `punkt::params::Default`.

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