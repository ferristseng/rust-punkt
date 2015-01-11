use std::default::Default;

use util;
use trainer::TrainingData;
use token::prelude::WordTokenWithFlagsOps;
use tokenizer::{WordTokenizer, WordTokenizerParameters};
use tokenizer::periodctxt::PeriodContextTokenizer;

pub struct SentenceTokenizer<'a> {
  doc: &'a str,
  iter: PeriodContextTokenizer<'a>,
  data: &'a TrainingData,
  algn: bool, 
  last: usize,
  pub params: &'a WordTokenizerParameters
}

impl<'a> SentenceTokenizer<'a> {
  fn new(
    doc: &'a str, 
    data: &'a TrainingData
  ) -> SentenceTokenizer<'a> {
    SentenceTokenizer {
      doc: doc,
      iter: PeriodContextTokenizer::new(doc),
      data: data,
      algn: false,
      last: 0,
      params: Default::default()
    }
  }
}

impl<'a> Iterator for SentenceTokenizer<'a> {
  type Item = &'a str; 

  fn next(&mut self) -> Option<&'a str> {
    loop {
      match self.iter.next() {
        Some((slice, tok_start, ws_start, slice_end)) => {
          let mut has_sentence_break = false;

          // Get word tokens in the slice. If any of them has a sentence break,
          // then set the flag `has_sentence_break`.
          for mut t in WordTokenizer::new(slice) {
            util::annotate_first_pass(&mut t, self.data, self.iter.params.sent_end);

            if t.is_sentence_break() { has_sentence_break = true }
          }

          // If there is a token with a sentence break, it is the end of 
          // a sentence. Set the beginning of the next sentence to the start 
          // of the start of the token, or the end of the slice if the token is 
          // punctuation. Then return the sentence.
          if has_sentence_break {
            let start = self.last;

            return if tok_start == slice_end {
              self.last = slice_end;
              Some(self.doc.slice(start, slice_end))
            } else {
              self.last = tok_start;
              Some(self.doc.slice(start, ws_start))
            }
          }
        }
        None => break
      }
    }

    None
  }

  #[inline]
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.doc.len() / 10, None)
  }
}

#[test]
fn test_sentence_tokenizer() {
  let data = TrainingData::english();
  let mut stok = SentenceTokenizer::new(
    include_str!("../../test/raw/npr-article-01.txt"), 
    &data);

  println!("");

  for s in stok {
    println!("[{}]", s.escape_default());
  }

  assert!(false);
}