//! # Overview
//!
//! Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence 
//! tokenization. Results have been compared with small and large texts that have 
//! been tokenized using NLTK. 
//!
//! # Training
//!
//! Training data can be provided to a `SentenceTokenizer` for better 
//! results. Data can be acquired manually by training with a `Trainer`, 
//! or using already compiled data from NLTK (example: `TrainingData::english()`).
//!
//! # Typical Usage
//! 
//! The punkt algorithm allows you to derive all the necessary data to perform 
//! sentence tokenization from the document itself. 
//!
//! ```
//! # use punkt::params::Default;
//! # use punkt::{Trainer, TrainingData, SentenceTokenizer};
//! # 
//! # let doc = "I bought $5.50 worth of apples from the store. I gave them to my dog when I came home.";
//! let trainer: Trainer<Default> = Trainer::new();
//! let mut data = TrainingData::new();
//! 
//! trainer.train(doc, &mut data);
//! 
//! for s in SentenceTokenizer::<Default>::new(doc, &data) {
//!   println!("{:?}", s);
//! }
//! ```
//!
//! `rust-punkt` also provides pretrained data that can be loaded for certain languages.
//! 
//! ```
//! # #![allow(unused_variables)]
//! #
//! # use punkt::TrainingData;
//! #
//! let data = TrainingData::english();
//! ```
//!
//! `rust-punkt` also allows training data to be incrementally gathered.
//!
//! ```
//! # use punkt::params::Default;
//! # use punkt::{Trainer, TrainingData, SentenceTokenizer};
//! #
//! # let docs = ["This is a sentence with a abbrev. in it."];
//! let trainer: Trainer<Default> = Trainer::new();
//! let mut data = TrainingData::new();
//! 
//! for d in docs.iter() {
//!   trainer.train(d, &mut data);
//!
//!   for s in SentenceTokenizer::<Default>::new(d, &data) {
//!     println!("{:?}", s);
//!   }
//! }
//! ```
//!
//! # Customization
//!
//! `rust-punkt` exposes a number of traits to customize how the trainer, sentence tokenizer, 
//! and internal tokenizers work. The default settings, which are nearly identical, to the 
//! ones available in the Python library are available in `punkt::params::Default`.
//! 
//! To modify only how the trainer works:
//! 
//! ```
//! # use punkt::params::{DefaultCharacterDefinitions, TrainerParameters};
//! #
//! struct MyParams;
//!
//! impl DefaultCharacterDefinitions for MyParams { }
//!
//! impl TrainerParameters for MyParams {
//! # fn abbrev_lower_bound() -> f64 { 0.3 }
//! # fn abbrev_upper_bound() -> f64 { 8f64 }
//! # fn ignore_abbrev_penalty() -> bool { false }
//! # fn collocation_lower_bound() -> f64 { 7.88 }
//! # fn sentence_starter_lower_bound() -> f64 { 35f64 }
//! # fn include_all_collocations() -> bool { false }
//! # fn include_abbrev_collocations() -> bool { true }
//! # fn collocation_frequency_lower_bound() -> f64 { 0.8f64 }
//! }
//! ```
//!
//! To fully modify how everything works:
//! 
//! ```
//! # use punkt::params::*;
//! #
//! struct MyParams;
//! 
//! impl DefinesSentenceEndings for MyParams { 
//! # fn sentence_endings() -> &'static Set<char> { Default::sentence_endings() }
//! }
//! 
//! impl DefinesInternalPunctuation for MyParams {
//! # fn internal_punctuation() -> &'static Set<char> { Default::sentence_endings() }
//! }
//!
//! impl DefinesNonWordCharacters for MyParams { 
//! # fn nonword_chars() -> &'static Set<char> { Default::nonword_chars() } 
//! }
//! 
//! impl DefinesPunctuation for MyParams {
//! # fn punctuation() -> &'static Set<char> { Default::punctuation() }
//! }
//!
//! impl DefinesNonPrefixCharacters for MyParams {
//! # fn nonprefix_chars() -> &'static Set<char> { Default::nonprefix_chars() }
//! }
//! 
//! impl TrainerParameters for MyParams {
//! # fn abbrev_lower_bound() -> f64 { 0.3 }
//! # fn abbrev_upper_bound() -> f64 { 8f64 }
//! # fn ignore_abbrev_penalty() -> bool { false }
//! # fn collocation_lower_bound() -> f64 { 7.88 }
//! # fn sentence_starter_lower_bound() -> f64 { 35f64 }
//! # fn include_all_collocations() -> bool { false }
//! # fn include_abbrev_collocations() -> bool { true }
//! # fn collocation_frequency_lower_bound() -> f64 { 0.8f64 }
//! }
//! ```

#![feature(plugin, str_char, collections)]
#![allow(dead_code)]
#![cfg_attr(test, feature(test, path_ext, fs_walk))]
#![plugin(phf_macros)]
#![warn(missing_docs)]

extern crate phf;
extern crate num;
extern crate rustc_serialize;
extern crate freqdist;
#[cfg(test)] extern crate test;

mod trainer;
mod util;
mod token;
mod tokenizer;
mod prelude;

pub use trainer::{Trainer, TrainingData};
pub use tokenizer::{SentenceTokenizer, SentenceByteOffsetTokenizer};

/// Contains traits for configuring all tokenizers, and the trainer. Also 
/// contains default parameters for tokenizers, and the trainer.
pub mod params {
  pub use prelude::{DefinesSentenceEndings, DefinesInternalPunctuation, 
    DefinesNonWordCharacters, DefinesPunctuation, DefinesNonPrefixCharacters,
    TrainerParameters, DefaultCharacterDefinitions, Default, Set};
}


#[cfg(test)] fn get_test_scenarios(
  dir_path: &str, 
  raw_path: &str
) -> Vec<(Vec<String>, String, String)> {
  #![allow(unused_must_use)]
  
  use std::fs;
  use std::fs::PathExt;
  use std::path::Path;
  use std::io::Read;

  let mut tests = Vec::new();

  for path in fs::walk_dir(&Path::new(dir_path)).unwrap() {
    let fpath = path.unwrap().path();

    if fpath.is_file() {
      let mut exp_strb = String::new();
      let mut raw_strb = String::new();

      // Files in the directory with raw articles must match the file names of 
      // articles in the directory with test outcomes.
      let rawp = Path::new(raw_path).join(fpath.file_name().unwrap());

      fs::File::open(&fpath).unwrap().read_to_string(&mut exp_strb);
      fs::File::open(&rawp).unwrap().read_to_string(&mut raw_strb);

      // Expected results, split by newlines.
      let exps: Vec<String> = exp_strb.split('\n').map(|s| s.to_string()).collect();

      tests.push((exps, raw_strb, format!("{:?}", fpath.file_name().unwrap())));
    }
  }

  tests // Returns (Expected cases, File contents, File name)
}