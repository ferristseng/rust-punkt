use std::hash::{Hash, Hasher};
use std::fmt::{Debug, Display, Formatter, Result};

use token::prelude::{
  WordTokenWithPeriod, 
  WordTypeToken,
  WordTokenWithFlags, 
  WordTokenWithFlagsOps,
  WordTokenWithFlagsOpsExt};

/// A Training Token is used during training, and includes an extended number of 
/// flags, to allow caching of a greater number of derived properties about the token.
pub struct TrainingToken {
  inner: String,
  flags: u16 
}

impl_flags!(TrainingToken, u16);

impl TrainingToken {
  /// Creates a new TrainingToken, from a string slice, and some metadata that 
  /// can be acquired while parsing. This constructor will properly set some flags 
  /// on the token. Note, the original string can not be recovered from this token.
  pub fn new(
    slice: &str,
    is_ellipsis: bool,
    is_paragraph_start: bool,
    is_newline_start: bool
  ) -> TrainingToken {
    debug_assert!(slice.len() > 0);

    let first = slice.char_at(0);
    let mut has_punct = false;

    // Add a period to any tokens without a period. This is an optimization 
    // to avoid creating an entirely new token when searching through the HashSet.
    let mut tok = if slice.as_bytes()[slice.len() - 1] == b'.' {
      let mut tok = TrainingToken {
        inner: String::with_capacity(slice.len()),
        flags: 0x00
      };

      tok.set_has_final_period(true);
      tok
    } else {
      TrainingToken {
        inner: String::with_capacity(slice.len() + 1),
        flags: 0x00
      }
    };

    // These are mutually exclusive. Only need to set one, although 
    // still potentially need to run both checks.
    if is_str_numeric(slice) {
      tok.set_is_numeric(true);
    } else if is_str_initial(slice) {
      tok.set_is_initial(true);
    }

    // Builds a normalized version of the slice from the slice.
    // Also, determine if a sentence has any punctuation, and is not 
    // entirely punctuation.
    for c in slice.chars() { 
      for c0 in c.to_lowercase() {
        tok.inner.push(c0);
      }

      if c.is_alphabetic() || c == '_' {
        tok.set_is_non_punct(true);
      } else if !c.is_digit(10) {
        has_punct = true;
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

    tok.set_is_alphabetic(!has_punct);
    tok.set_is_ellipsis(is_ellipsis);
    tok.set_is_paragraph_start(is_paragraph_start);
    tok.set_is_newline_start(is_newline_start);

    tok
  }
}

impl WordTokenWithPeriod for TrainingToken {
  #[inline]
  fn token_with_period(&self) -> &str {
    &self.inner[..]
  }
}

impl Display for TrainingToken {
  #[inline]
  fn fmt(&self, fmt: &mut Formatter) -> Result {
    Debug::fmt(&self, fmt)
  }
}

impl Debug for TrainingToken {
  #[inline]
  fn fmt(&self, fmt: &mut Formatter) -> Result {
    write!(fmt, "{}", self.typ()) 
  }
}

impl Eq for TrainingToken { }

impl PartialEq for TrainingToken {
  #[inline]
  fn eq(&self, other: &TrainingToken) -> bool {
    self.typ() == other.typ()
  }
}

impl Hash for TrainingToken {
  #[inline]
  fn hash<H>(&self, state: &mut H) where H: Hasher {
    self.typ().hash(state)
  }
}

/// A number can start with a negative sign ('-'), and be followed by digits
/// or isolated periods, commas, or dashes. 
/// Note: It's assumed that multi-chars are taken out of the input when creating word 
/// tokens, so a numeric word token SHOULD not have a multi-char within it as
/// its received. This assumption should be fulfilled by the parser generating 
/// these word tokens. If it isn't some weird outputs are possible (such as "5.4--5").
#[inline]
fn is_str_numeric(tok: &str) -> bool {
  let mut digit_found = false;
  let mut pos = 0usize;

  for c in tok.chars() {
    match c {
      // A digit was found. Note this to confirm punctuation
      // within the number is valid or not.
      _ if c.is_digit(10) => {
        digit_found = true
      }
      // A delimeter was found. This is valid as long as 
      // a digit was also found prior.
      ',' | '.' | '-' if digit_found => (),
      // A comma or period was found as the first character, or 
      // after a negative sign. This is a valid token.
      ',' | '.' if pos == 0 || pos == 1 => (),
      // A negative sign is found. 
      '-' if pos == 0 => (),
      // A non numeric token was encountered in the string that 
      // isn't a valid one. Return false.
      _ => return false
    }

    pos += c.len_utf8(); 
  }

  digit_found
}

/// Tests if the token is an initial. An initial is a 2 character grouping
/// where the first character is a letter (non-digit, non-symbol), and the 
/// next is a period.
#[inline]
fn is_str_initial(tok: &str) -> bool {
  let mut iter = tok.chars();

  (match (iter.next(), iter.next())
  {
    (Some(c), Some('.')) if c.is_alphabetic() => true,
    _ => false
  }) && iter.next().is_none()
}
