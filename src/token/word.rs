use std::hash::Hash;
use std::borrow::BorrowFrom;
use std::fmt::{Show, Formatter, Result};

use xxhash::XXState;

use token::prelude::{
  WordToken,
  WordTokenWithPeriod,
  WordTokenWithFlags,
  WordTokenWithFlagsOps};

pub struct SentenceWordToken {
  start: uint,
  inner: String,
  flags: u8
}

impl_flags!(SentenceWordToken, u8);

impl SentenceWordToken {
  pub fn new(
    start: uint,
    slice: &str,
    is_ellipsis: bool,
    is_paragraph_start: bool,
    is_newline_start: bool
  ) -> SentenceWordToken {
    debug_assert!(slice.len() > 0);

    // Add a period to any tokens without a period. This is an optimization 
    // to avoid creating an entirely new token when searching through the HashSet.
    let mut tok = if slice.as_bytes()[slice.len() - 1] == b'.' {
      let mut tok = SentenceWordToken {
        start: start,
        inner: String::with_capacity(slice.len()),
        flags: 0x0
      };

      tok.set_has_final_period(true);
      tok
    } else {
      SentenceWordToken {
        start: start,
        inner: String::with_capacity(slice.len() + 1),
        flags: 0x0
      }
    };

    // Builds a normalized version of the slice from the slice.
    for c in slice.chars() { 
      tok.inner.push(c.to_lowercase())
    }
    
    tok.set_is_ellipsis(is_ellipsis);
    tok.set_is_paragraph_start(is_paragraph_start);
    tok.set_is_newline_start(is_newline_start);

    tok
  }
}

impl WordTokenWithPeriod for SentenceWordToken {
  #[inline]
  fn token_with_period(&self) -> &str {
    self.inner.as_slice()
  }
}

impl Show for SentenceWordToken {
  fn fmt(&self, fmt: &mut Formatter) -> Result {
    write!(fmt, "{}", self.token()) 
  }
}

impl Eq for SentenceWordToken { }

impl PartialEq for SentenceWordToken {
  #[inline]
  fn eq(&self, other: &SentenceWordToken) -> bool {
    self.token() == other.token()
  }
}

impl Hash<XXState> for SentenceWordToken {
  #[inline]
  fn hash(&self, state: &mut XXState) {
    self.token().hash(state)
  }
}

impl BorrowFrom<SentenceWordToken> for str {
  #[inline]
  fn borrow_from(owned: &SentenceWordToken) -> &str {
    owned.token()
  }
}
