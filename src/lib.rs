//! # Overview
//!
//! Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence tokenization.
//! Includes a word tokenizer that tokenizes words based on regexes defined in Python's 
//! NLTK library. Results have been compared with small and large texts that have been 
//! tokenized with NLTK's library. For usage, check out `PunktSentenceTokenizer`.
//!
//! # Training
//!
//! Training data can be provided to a `PunktSentenceTokenizer` for better results. Data 
//! can be acquired manually by training with a `PunktTrainer`, or using already compiled
//! data from NLTK (example: `PunktData::english()`).
//!
//! Training parameters can be specified using `PunktTrainerParameters`. The defaults 
//! are from NLTK, but customized threshold values and flags can be set.


#![feature(plugin, str_char, std_misc, collections)]
#![cfg_attr(test, allow(unused_must_use))]
#![cfg_attr(test, feature(test, path_ext, fs_walk))]
#![plugin(phf_macros)]
#![warn(missing_docs)]

extern crate phf;
extern crate num;
extern crate rustc_serialize;
extern crate freqdist;
extern crate collections;
#[cfg(test)] extern crate test;
#[macro_use] extern crate log;

/// Trainer to train a `SentenceTokenizer`. This module also contains 
/// default data to use, that was trained for a variety of different languages.
pub mod trainer;

/// Tokenizer module. Contains different tokenizers, and customization parameters.
pub mod tokenizer;

mod util;
mod ortho;
mod token;

pub use token::SentenceWordToken;