use std::hash::Hash;
use std::borrow::BorrowFrom;
use std::fmt::{Show, Formatter, Result};

use xxhash::XXHasher;

use token::prelude::{
  WordToken,
  WordTokenWithPeriod,
  WordTokenWithFlags,
  WordTokenWithFlagsOps};

pub struct SentenceWordToken<'a> {
  slice: &'a str,
  inner: String,
  flags: u8
}

impl<'a> SentenceWordToken<'a> {
  /// Creates a new sentence word token from a string slice (copies the string).
  /// This constructor will set the `has_final_period` flag, but nothing else. 
  /// In most instances, this initialization should be sufficient.
  pub fn new(slice: &'a str) -> SentenceWordToken {
    debug_assert!(slice.len() > 0);

    let first = slice.char_at(0);

    // Add a period to any tokens without a period. This is an optimization 
    // to avoid creating an entirely new token when searching through the HashSet.
    let mut tok = if slice.as_bytes()[slice.len() - 1] == b'.' {
      let mut tok = SentenceWordToken {
        slice: slice,
        inner: String::with_capacity(slice.len()),
        flags: 0x0
      };

      tok.set_has_final_period(true);
      tok
    } else {
      SentenceWordToken {
        slice: slice,
        inner: String::with_capacity(slice.len() + 1),
        flags: 0x0
      }
    };

    // Builds a normalized version of the slice from the slice.
    for c in slice.chars() { 
      tok.inner.push(c.to_lowercase())
    }

    if !tok.has_final_period() {
      tok.inner.push('.');
    }

    // Check if the first character is uppercase or lowercase.
    if first.is_uppercase() {
      tok.set_is_uppercase(true);
    } else if first.is_lowercase() {
      tok.set_is_lowercase(true);
    }

    tok
  }

  /// Returns the original string (the one that was used to create)
  /// this token.
  #[inline]
  pub fn original(&self) -> &str {
    self.slice
  }
}

impl<'a> WordTokenWithFlags for SentenceWordToken<'a> {
  type Flags = u8;

  #[inline]
  fn flags(&self) -> &u8 {
    &self.flags
  }

  #[inline]
  fn flags_mut(&mut self) -> &mut u8 {
    &mut self.flags
  }
}

impl<'a> WordTokenWithPeriod for SentenceWordToken<'a> {
  #[inline]
  fn token_with_period(&self) -> &str {
    self.inner.as_slice()
  }
}

impl<'a> Show for SentenceWordToken<'a> {
  fn fmt(&self, fmt: &mut Formatter) -> Result {
    write!(fmt, "{}", self.slice) 
  }
}

impl<'a> Eq for SentenceWordToken<'a> { }

impl<'a> PartialEq for SentenceWordToken<'a> {
  #[inline]
  fn eq(&self, other: &SentenceWordToken) -> bool {
    self.token() == other.token()
  }
}

impl<'a> Hash<XXHasher> for SentenceWordToken<'a> {
  #[inline]
  fn hash(&self, state: &mut XXHasher) {
    self.token().hash(state)
  }
}

impl<'a> BorrowFrom<SentenceWordToken<'a>> for str {
  #[inline]
  fn borrow_from<'b>(owned: &'b SentenceWordToken) -> &'b str {
    owned.token()
  }
}