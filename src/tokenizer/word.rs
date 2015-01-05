use std::default::Default;

use token::word::SentenceWordToken;

use phf::Set;

static DEFAULT: WordTokenizerParameters = WordTokenizerParameters {
  non_word: &phf_set![
    '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '['
  ],
  sent_end: &phf_set!['.', '?', '!']
};

pub struct WordTokenizerParameters {
  non_word: &'static Set<char>,
  sent_end: &'static Set<char>
}

impl Default for &'static WordTokenizerParameters {
  #[inline]
  fn default() -> &'static WordTokenizerParameters {
    &DEFAULT
  }
}

pub struct WordTokenizer<'a> {
  doc: &'a str,
  pos: uint,
  params: &'a WordTokenizerParameters
}

impl<'a> Iterator for WordTokenizer<'a> {
  type Item = SentenceWordToken;

  fn next(&mut self) -> Option<SentenceWordToken> {
    None
  }

  fn size_hint(&self) -> (uint, Option<uint>) {
    (self.doc.len() / 5, Some(self.doc.len() / 3))
  }
}
