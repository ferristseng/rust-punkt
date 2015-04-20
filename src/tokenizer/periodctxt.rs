use std::default::Default;

use phf::Set;

static DEFAULT: PeriodContextTokenizerParameters = PeriodContextTokenizerParameters {
  non_word: &phf_set![
    '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '['
  ],
  sent_end: &phf_set!['.', '?', '!']
};

pub struct PeriodContextTokenizerParameters {
  pub non_word: &'static Set<char>,
  pub sent_end: &'static Set<char>
}

impl Default for &'static PeriodContextTokenizerParameters {
  #[inline]
  fn default() -> &'static PeriodContextTokenizerParameters {
    &DEFAULT
  }
}

pub struct PeriodContextTokenizer<'a> {
  doc: &'a str,
  pos: usize,
  pub params: &'a PeriodContextTokenizerParameters
}

impl<'a> PeriodContextTokenizer<'a> {
  #[inline]
  pub fn new(doc: &'a str) -> PeriodContextTokenizer<'a> {
    PeriodContextTokenizer::with_parameters(doc, Default::default())
  }

  #[inline]
  pub fn with_parameters(
    doc: &'a str,
    params: &'a PeriodContextTokenizerParameters
  ) -> PeriodContextTokenizer<'a> {
    PeriodContextTokenizer {
      doc: doc,
      pos: 0,
      params: params
    }
  }

  /// Performs a lookahead to see if a sentence ending character is actually
  /// the end of the token. If it is the end, `None` is returned. Otherwise,
  /// return `Some(x)` where `x` is the new position to iterate to.
  fn lookahead_is_token(&self) -> Option<usize> {
    let mut pos = self.pos;

    while pos < self.doc.len() {
      let cur = self.doc.char_at(pos);

      match cur {
        // A whitespace is reached before a sentence ending character
        // that could signal the continuation of a token.
        c if c.is_whitespace() => return None,
        // A sentence ending is reached. Check if it could be the beginning
        // of a new token (if there is a space after it, or if the next
        // character is puntuation).
        c if self.params.sent_end.contains(&c) => {
          let nxt = self.doc.char_at(pos + cur.len_utf8());

          if nxt.is_whitespace() || self.params.non_word.contains(&nxt) {
            break;
          }
        }
        _ => ()
      }

      pos += cur.len_utf8();
    }

    Some(pos)
  }
}

const STATE_SENT_END: u8 = 0b00000001; // Hit a sentence end state.
const STATE_TOKN_BEG: u8 = 0b00000010; // Token began state.
const STATE_CAPT_TOK: u8 = 0b00000100; // Start capturing token state.
const STATE_UPDT_STT: u8 = 0b10000000; // Update the start token flag.
const STATE_UPDT_RET: u8 = 0b01000000; // Update the position at end flag.

impl<'a> Iterator for PeriodContextTokenizer<'a> {
  // (Entire slice of section, beginning of next break (if there is one),
  // start of whitespace before next token, end of entire slice)
  type Item = (&'a str, usize, usize, usize);

  fn next(&mut self) -> Option<(&'a str, usize, usize, usize)> {
    let mut astart = self.pos;
    let mut wstart = self.pos;
    let mut nstart = self.pos;
    let mut state: u8 = 0;

    while self.pos < self.doc.len() {
      let cur = self.doc.char_at(self.pos);
      
      macro_rules! return_token(
        () => (
          {
            let end = self.pos;

            // Return to the start of a any next token that occured 
            // with a sentence ending.
            if state & STATE_UPDT_RET != 0 {
              self.pos = nstart;
            }

            return Some((
              &self.doc[astart..end],
              nstart,
              wstart,
              end));
          }
        )
      );

      match cur {
        // A sentence ending was encountered. Set the appropriate state.
        // This is done anytime a sentence ender is encountered. It should not
        // affect capturing. 
        c if self.params.sent_end.contains(&c) => 
        {
          state |= STATE_SENT_END;

          // If an update is needed on the starting position of the entire token
          // update it, and toggle the flag back.
          if state & STATE_UPDT_STT != 0 {
            astart = self.pos;
            state ^= STATE_UPDT_STT;
          }

          // Capturing a token, and a sentence ending token is encountered. 
          // This token needs to be revisited, so set retpos to this position.
          if state & STATE_CAPT_TOK != 0 {
            state |= STATE_UPDT_RET;
          }
        }
        // A sentence ending has not yet been countered. If a whitespace is 
        // encountered, the start of the token needs to be updated. Set a flag 
        // to state this fact. If a non-whitespace is encountered, and the start 
        // needs to be updated, then actually update the start position.
        c if state & STATE_SENT_END == 0 => {
          if c.is_whitespace() {
            state |= STATE_UPDT_STT;
          } else if state & STATE_UPDT_STT != 0 {
            astart = self.pos;
            state ^= STATE_UPDT_STT;
          }
        }
        // Hit a sentence end character already, but not yet at token begin.
        // If whitespace is hit, then capturing of token can begin.
        // If a non-word token is hit, then return.
        // Otherwise, no match was made, continue. 
        c if state & STATE_SENT_END != 0 && 
             state & STATE_TOKN_BEG == 0 => 
        {
          if c.is_whitespace() {
            state |= STATE_TOKN_BEG;
            wstart = self.pos;
          } else if self.params.non_word.contains(&c) {
            // Setup positions for the return macro.
            self.pos += c.len_utf8();
            nstart = self.pos;

            match self.lookahead_is_token() {
              Some(x) => self.pos = x,
              None    => return_token!() 
            }
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
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.doc.len() / 5, Some(self.doc.len() / 3))
  }
}

#[test]
fn periodctxt_tokenizer_compare_nltk() {
  for (expected, raw, file) in super::get_test_scenarios("test/word-periodctxt/", "test/raw/") {
    for ((t, _, _, _), e)  in PeriodContextTokenizer::new(&raw[..]).zip(expected) {
      let t = t.replace("\n", r"\n").replace("\r", "");
      let e = e.replace("\r", "");

      assert!(t == e, "{} - you: [{}] != exp: [{}]", file, t, e);
    }
  }
}