use std::ops::Deref;
use std::hash::Hash;

use xxhash::XXState;

use token::prelude::{WordToken, WordTypeToken};

/// A collocation. A normal Tuple can not be used, because a collocation
/// as defined by NLTK requires a special hash function. 
#[derive(Show)]
pub struct Collocation<T> {
  l: T,
  r: T
}

impl<T> Collocation<T> {
  #[inline]
  pub fn new(l: T, r: T) -> Collocation<T> {
    Collocation { l: l, r: r }
  }

  #[inline]
  pub fn left(&self) -> &T {
    &self.l
  }

  #[inline]
  pub fn right(&self) -> &T {
    &self.r
  }
}

impl<T: WordTypeToken, D: Deref<Target = T>> Eq for Collocation<D> { }

impl<T: WordTypeToken, D: Deref<Target = T>> Hash<XXState> for Collocation<D> {
  #[inline]
  fn hash(&self, state: &mut XXState) {
    (*self.l).typ_without_period().hash(state); 
    (*self.r).typ_without_break_or_period().hash(state);
  }
}

impl<T: WordTypeToken, D: Deref<Target = T>> PartialEq for Collocation<D> {
  #[inline]
  fn eq(&self, x: &Collocation<D>) -> bool {
    (*self.l).typ_without_period() == (*x.l).typ_without_period() &&
    (*self.r).typ_without_break_or_period() == (*x.r).typ_without_break_or_period()
  }
}