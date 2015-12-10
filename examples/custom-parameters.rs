#![feature(plugin)]
#![plugin(phf_macros)]

extern crate phf;
extern crate punkt;

use punkt::params::*;
use punkt::{SentenceTokenizer, Trainer, TrainingData};


static NONWORD_CHARS : Set<char> = phf_set![
  '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '[',
  '\u{201c}', '\u{201d}'
];


struct MyParams;

impl DefinesSentenceEndings for MyParams { 
  #[inline(always)] fn sentence_endings() -> &'static Set<char> { 
    Standard::sentence_endings()
  }
}

impl DefinesInternalPunctuation for MyParams {
  #[inline(always)] fn internal_punctuation() -> &'static Set<char> {
    Standard::sentence_endings()
  }
}

impl DefinesNonWordCharacters for MyParams { 
  #[inline(always)] fn nonword_chars() -> &'static Set<char> {
    &NONWORD_CHARS
  }
}

impl DefinesPunctuation for MyParams {
  #[inline(always)] fn punctuation() -> &'static Set<char> {
    Standard::punctuation()
  }
}

impl DefinesNonPrefixCharacters for MyParams {
  #[inline(always)] fn nonprefix_chars() -> &'static Set<char> {
    Standard::nonprefix_chars()
  }
}

impl TrainerParameters for MyParams {
  #[inline(always)] fn abbrev_lower_bound() -> f64 { 0.3 }
  #[inline(always)] fn abbrev_upper_bound() -> f64 { 8f64 }
  #[inline(always)] fn ignore_abbrev_penalty() -> bool { false }
  #[inline(always)] fn collocation_lower_bound() -> f64 { 7.88 }
  #[inline(always)] fn sentence_starter_lower_bound() -> f64 { 35f64 }
  #[inline(always)] fn include_all_collocations() -> bool { false }
  #[inline(always)] fn include_abbrev_collocations() -> bool { true }
  #[inline(always)] fn collocation_frequency_lower_bound() -> f64 { 0.8f64 }
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