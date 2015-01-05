use std::ops::Slice;
use std::default::Default;

use token::word::SentenceWordToken;
use token::prelude::WordToken;

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

static STATE_SENT_END: u8 = 0b001;
static STATE_TOKN_BEG: u8 = 0b010;
static STATE_CAPT_TOK: u8 = 0b100;

impl<'a> Iterator for WordTokenizer<'a> {
  type Item = SentenceWordToken;

  fn next(&mut self) -> Option<SentenceWordToken> {
    let mut tstart = self.pos;
    let mut nstart = self.pos;
    let mut state: u8 = 0;

    macro_rules! return_token(
      () => (
        return Some(SentenceWordToken::new(
          tstart,
          self.doc.slice_or_fail(&tstart, &self.pos),
          false,
          false,
          false));
      )
    );

    while self.pos < self.doc.len() {
      let cur = self.doc.char_at(self.pos);

      self.pos += cur.len_utf8();
      
      match cur {
        // A sentence ending has not been encountered yet, and one 
        // was encountered. Start searching for a token.
        c if state & STATE_SENT_END == 0 &&
             self.params.sent_end.contains(&c) => 
        {
          state |= STATE_SENT_END;
        }
        // Hit a sentence end character, but not yet at token begin.
        // If whitespace is hit, then capturing of token can begin.
        // If a non-word is hit, then return that punctuation. Otherwise,
        // no match was made, continue. 
        c if state & STATE_SENT_END != 0 && 
             state & STATE_TOKN_BEG == 0 => 
        {
          if c.is_whitespace() {
            tstart = self.pos;
            state |= STATE_TOKN_BEG;
          } else if self.params.sent_end.contains(&c) {
            return_token!()
          } else if !self.params.sent_end.contains(&c) { 
            state ^= STATE_SENT_END;
          }
        }
        // Capturing the whitespace before a token, and a non-whitespace
        // is encountered. Start capturing that token.
        c if state & STATE_SENT_END != 0 &&
             state & STATE_TOKN_BEG != 0 && 
             state & STATE_CAPT_TOK == 0 =>
        {
          if !c.is_whitespace() {
            state |= STATE_CAPT_TOK;         
          }
        }
        // Whitespace after a token has been encountered. Final state -- return.
        c if state & STATE_CAPT_TOK != 0 
             && c.is_whitespace() =>
        {
          return_token!()
        }
        // Skip if not in a state at all.
        _ => ()
      }
    }

    None
  }

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    (self.doc.len() / 5, Some(self.doc.len() / 3))
  }
}

#[test]
fn test_word_tokenizer() {
  let mut iter = WordTokenizer {
    pos: 0,
    doc: include_str!("../../test/ny-times-article-01-raw.txt"),
    params: Default::default()
  };

  let collect: Vec<SentenceWordToken> = iter.collect();

  for c in collect.iter() {
    println!("--[{}]--", c.token().escape_default());
  }

  assert!(false);
}
