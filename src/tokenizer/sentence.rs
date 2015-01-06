use std::default::Default;

use util::annotate_first_pass;
use trainer::TrainingData;
use tokenizer::periodctxt::PeriodContextTokenizer;

use phf::Set;

static DEFAULT: SentenceTokenizerParams = SentenceTokenizerParams {
  sent_end: &phf_set!['.', '?', '!']
};

pub struct SentenceTokenizerParams {
  sent_end: &'static Set<char>
}

impl Default for &'static SentenceTokenizerParams {
  fn default() -> &'static SentenceTokenizerParams {
    &DEFAULT
  }
}

pub struct SentenceTokenizer<'a> {
  doc: &'a str,
  iter: PeriodContextTokenizer<'a>,
  data: &'a TrainingData,
  algn: bool, 
  last: uint
}

impl<'a> Iterator for SentenceTokenizer<'a> {
  type Item = &'a str; 

  fn next(&mut self) -> Option<&'a str> {
    None
  }

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    (self.doc.len() / 10, None)
  }
}

struct SentenceWordTokenizer;