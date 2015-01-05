use std::rc::Rc;
use std::hash::Hash;
use std::borrow::BorrowFrom;
use std::fmt::{Show, Formatter, Result};

use xxhash::XXState;

use token::prelude::{
  WordTokenWithPeriod, 
  WordTypeToken,
  WordTokenWithFlags, 
  WordTokenWithFlagsOps};

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
