#![feature(proc_macro_hygiene)]

extern crate phf;
extern crate punkt;

use punkt::params::*;
use punkt::{SentenceTokenizer, Trainer, TrainingData};
use phf::phf_set;

struct MyParams;

impl DefinesInternalPunctuation for MyParams {}
impl DefinesNonPrefixCharacters for MyParams {}
impl DefinesNonWordCharacters for MyParams {
  const NONWORD_CHARS: &'static Set<char> = &phf_set![
    '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '[', '\u{201c}', '\u{201d}'
  ];
}
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
  const COLLOCATION_FREQUENCY_LOWER_BOUND: f64 = 0.8;
}

// The article in this example has some unicode characters in it that are not
// defined in the default settings. The above custom parameters modify some
// of the parameters for the trainer, and add in the unicode characters present
// in the article, to provide better results.
fn main() {
  println!("\n-- Trained using custom parameters --\n");

  let doc = include_str!("../test/raw/ny-times-article-02.txt");
  let trainer: Trainer<MyParams> = Trainer::new();
  let mut data = TrainingData::new();

  trainer.train(doc, &mut data);

  for s in SentenceTokenizer::<MyParams>::new(doc, &data) {
    println!("{:?}", s);
  }
}
