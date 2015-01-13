use std::ops::Deref;
use std::hash::Hash;

use xxhash::XXHasher;

use token::prelude::WordTypeToken;

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

impl<T, D> Eq for Collocation<D> 
  where T: WordTypeToken, D: Deref<Target = T> 
{ }

impl<T, D> Hash<XXHasher> for Collocation<D>
  where T: WordTypeToken, D: Deref<Target = T> 
{
  #[inline]
  fn hash(&self, state: &mut XXHasher) {
    (*self.l).typ_without_period().hash(state); 
    (*self.r).typ_without_break_or_period().hash(state);
  }
}

impl<T, D> PartialEq for Collocation<D> 
  where T: WordTypeToken, D: Deref<Target = T> 
{
  #[inline]
  fn eq(&self, x: &Collocation<D>) -> bool {
    (*self.l).typ_without_period() == (*x.l).typ_without_period() &&
    (*self.r).typ_without_break_or_period() == (*x.r).typ_without_break_or_period()
  }
}
