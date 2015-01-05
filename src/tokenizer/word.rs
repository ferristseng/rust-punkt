use std::ops::Slice;
use std::default::Default;

use token::SentenceWordToken;
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

impl<'a> WordTokenizer<'a> {
  #[inline]
  pub fn new(doc: &'a str) -> WordTokenizer<'a> {
    WordTokenizer {
      doc: doc, 
      pos: 0,
      params: Default::default()
    }
  }

  #[inline]
  pub fn with_parameters(
    doc: &'a str, 
    params: &'a WordTokenizerParameters
  ) -> WordTokenizer<'a> {
    WordTokenizer {
      doc: doc,
      pos: 0,
      params: params
    }
  }
}

const STATE_SENT_END: u8 = 0b001;
const STATE_TOKN_BEG: u8 = 0b010;
const STATE_CAPT_TOK: u8 = 0b100;

impl<'a> Iterator for WordTokenizer<'a> {
  type Item = (SentenceWordToken<'a>, SentenceWordToken<'a>);

  fn next(&mut self) -> Option<(SentenceWordToken<'a>, SentenceWordToken<'a>)> {
    let mut tstart = self.pos;
    let mut nstart = self.pos;
    let mut retpos = None;
    let mut state: u8 = 0;

    while self.pos < self.doc.len() {
      let cur = self.doc.char_at(self.pos);
      
      macro_rules! return_token(
        () => (
          {
            let end = self.pos;

            // Return to the last encountered sentence ending character, 
            // if any are encountered during capturing of a token.
            self.pos = retpos.unwrap_or(self.pos + cur.len_utf8());

            return Some((
              SentenceWordToken::new(self.doc.slice_or_fail(&tstart, &end)),
              SentenceWordToken::new(self.doc.slice_or_fail(&nstart, &end))));
          }
        )
      );

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
            nstart = self.pos;
            state |= STATE_CAPT_TOK;         
          }
        }
        // Capturing a token, and a sentence ending token is encountered. 
        // This token needs to be revisited, so set retpos to this position.
        c if state & STATE_CAPT_TOK != 0 &&
             self.params.sent_end.contains(&c) =>
        {
          retpos = Some(self.pos);
        }
        // Whitespace after a token has been encountered. Final state -- return.
        c if state & STATE_CAPT_TOK != 0 && 
             c.is_whitespace() =>
        {
          return_token!()
        }
        // Skip if not in a state at all.
        _ => ()
      }

      self.pos += cur.len_utf8();
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
  let iter = WordTokenizer {
    pos: 0,
    doc: include_str!("../../test/raw/ny-times-article-01.txt"),
    params: Default::default()
  };

  let collect: Vec<(SentenceWordToken, SentenceWordToken)> = iter.collect();

  println!("");

  for &(ref c0, ref c1) in collect.iter() {
    println!(
      "('{}', '{}')", 
      c0.original().escape_default(), 
      c1.original().escape_default());
  }

  assert!(true);
}
