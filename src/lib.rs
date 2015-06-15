//! # Overview
//!
//! Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence 
//! tokenization. Results have been compared with small and large texts that have 
//! been tokenized with NLTK's library. 
//!
//! # Training
//!
//! Training data can be provided to a `SentenceTokenizer` for better 
//! results. Data can be acquired manually by training with a `Trainer`, 
//! or using already compiled data from NLTK (example: `TrainingData::english()`).
//!
//! # Customization
//!
//!
//! # Typical Usage
//!
//!

#![feature(plugin, str_char, std_misc, collections, hash)]
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
pub use tokenizer::SentenceTokenizer;

/// Contains traits for configuring all tokenizers, and the trainer. Also 
/// contains default parameters for tokenizers, and the trainer.
pub mod params {
  pub use prelude::{DefinesSentenceEndings, DefinesInternalPunctuation, 
    DefinesNonWordCharacters, DefinesPunctuation, DefinesNonPrefixCharacters,
    TrainerParameters, DefaultCharacterDefinitions, Default};
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