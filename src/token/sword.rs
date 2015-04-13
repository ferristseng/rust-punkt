use std::hash::{Hash, Hasher};
use std::borrow::Borrow;
use std::fmt::{Debug, Display, Formatter, Result};

use token::prelude::{
  WordToken,
  WordTokenWithPeriod,
  WordTokenWithFlags,
  WordTokenWithFlagsOps};

/// A word token within a sentence. The token contains a reference to the original
/// slice in the document that the token was constructed from, and can be acquired 
/// with `original()`.
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
      for c0 in c.to_lowercase() {
        tok.inner.push(c0);
      }
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
    &self.inner[..]
  }
}

impl<'a> Display for SentenceWordToken<'a> {
  #[inline]
  fn fmt(&self, fmt: &mut Formatter) -> Result {
    Debug::fmt(&self, fmt)
  }
}

impl<'a> Debug for SentenceWordToken<'a> {
  #[inline]
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

impl<'a> Hash for SentenceWordToken<'a> {
  #[inline]
  fn hash<H>(&self, state: &mut H) where H: Hasher {
    self.token().hash(state)
  }
}

impl<'a> Borrow<str> for SentenceWordToken<'a> {
  #[inline]
  fn borrow<'b>(&'b self) -> &'b str {
    self.token()
  }
}
