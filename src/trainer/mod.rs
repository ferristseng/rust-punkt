pub use trainer::trainer::Trainer;
pub use trainer::data::TrainingData;

use token::TrainingToken;
use token::prelude::WordTypeToken;

use std::rc::Rc;
use std::ops::Deref;
use std::borrow::Borrow;

mod col;
mod math;
mod data;
mod trainer;

/// Wrapper around a reference counted Training Token, 
/// to use as a key for frequency distributions, to avoid the 
/// copying of possibly redundant strings.
#[derive(Eq, PartialEq, Hash, Clone)]
struct TrainingTokenKey(Rc<TrainingToken>);

impl TrainingTokenKey {
  #[inline]
  pub fn new(tt: TrainingToken) -> TrainingTokenKey {
    TrainingTokenKey(Rc::new(tt))
  }
}

impl Deref for TrainingTokenKey {
  type Target = TrainingToken;

  #[inline]
  fn deref<'a>(&'a self) -> &'a TrainingToken {
    self.borrow()
  }
}

impl Borrow<str> for TrainingTokenKey {
  #[inline]
  fn borrow(&self) -> &str {
    let tt: &TrainingToken = self.borrow(); tt.typ()
  }
}

impl Borrow<TrainingToken> for TrainingTokenKey {
  #[inline]
  fn borrow(&self) -> &TrainingToken {
    let TrainingTokenKey(ref tt) = *self; tt
  }
}
