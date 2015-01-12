#![allow(unstable)]
#![feature(plugin)]

/// # Overview
///
/// Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence tokenization.
/// Includes a word tokenizer that tokenizes words based on regexes defined in Python's 
/// NLTK library. Results have been compared with small and large texts that have been 
/// tokenized with NLTK's library. For usage, check out `PunktSentenceTokenizer`.
///
/// # Training
///
/// Training data can be provided to a `PunktSentenceTokenizer` for better results. Data 
/// can be acquired manually by training with a `PunktTrainer`, or using already compiled
/// data from NLTK (example: `PunktData::english()`).
///
/// Training parameters can be specified using `PunktTrainerParameters`. The defaults 
/// are from NLTK, but customized threshold values and flags can be set.

#[cfg(test)] extern crate test;

extern crate xxhash;
extern crate phf;
#[plugin] #[no_link] extern crate phf_mac;

extern crate "rustc-serialize" as rustc_serialize;
extern crate freqdist;
extern crate collections;
#[plugin] extern crate log;

pub mod token;
pub mod tokenizer;
pub mod trainer;

mod util;
mod ortho;