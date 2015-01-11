use std::rc::Rc;
use std::hash::Hash;
use std::borrow::BorrowFrom;
use std::fmt::{Show, Formatter, Result};

use xxhash::XXState;

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
  pub fn new(
    slice: &str,
    is_ellipsis: bool,
    is_paragraph_start: bool,
    is_newline_start: bool
  ) -> TrainingToken {
    debug_assert!(slice.len() > 0);

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
    for c in slice.chars() { 
      tok.inner.push(c.to_lowercase())
    }
    
    if !tok.has_final_period() {
      tok.inner.push('.');
    }

    tok.set_is_ellipsis(is_ellipsis);
    tok.set_is_paragraph_start(is_paragraph_start);
    tok.set_is_newline_start(is_newline_start);

    tok
  }
}

impl WordTokenWithPeriod for TrainingToken {
  #[inline]
  fn token_with_period(&self) -> &str {
    self.inner.as_slice()
  }
}

impl Show for TrainingToken {
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

impl Hash<XXState> for TrainingToken {
  #[inline]
  fn hash(&self, state: &mut XXState) {
    self.typ().hash(state)
  }
}

impl BorrowFrom<Rc<TrainingToken>> for str {
  #[inline]
  fn borrow_from(owned: &Rc<TrainingToken>) -> &str {
    owned.typ()
  }
}

/// A number can start with a negative sign ('-'), and be followed by digits
/// or isolated periods, commas, or dashes. 
/// Note: It's assumed that multi-chars are taken out the input when creating word 
/// tokens, so a numeric word token SHOULD not have a multi-char within it as
/// its received. This assumption should be fulfilled by the parser generating 
/// these word tokens. If it isn't some weird outputs are possible (such as "5.4--5").
#[inline]
fn is_str_numeric(tok: &str) -> bool {
  let mut digit_found = false;
  let mut pos = 0us;

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

/// Tests if the token is an initial. 
#[inline]
fn is_str_initial(tok: &str) -> bool {
  let mut iter = tok.chars();

  match (iter.next(), iter.next())
  {
    (Some(c), Some('.')) if c.is_alphabetic() => true,
    _ => false
  }
}
