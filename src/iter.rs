/// Iterates over every PuntkToken from the supplied iterator and returns 
/// the immediate following token. Returns None for the following token on the 
/// last token.
pub struct ConsecutiveTokenIterator<'a, T: 'a, I: Iterator<Item = &'a T>> {
  iter: I,
  last: Option<&'a T>
}

impl<'a, T: 'a, I: Iterator<Item = &'a T>> Iterator
for ConsecutiveTokenIterator<'a, T, I>
{
  type Item = (&'a T, Option<&'a T>); 

  #[inline]
  fn next(&mut self) -> Option<(&'a T, Option<&'a T>)> {
    match self.last {
      Some(tok) => {
        self.last = self.iter.next();

        Some((tok, self.last))
      }
      None => match self.iter.next() {
        Some(tok) => {
          self.last = self.iter.next();

          Some((tok, self.last))
        }
        None => None
      }
    }
  }

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    self.iter.size_hint()
  }
}

#[inline]
pub fn consecutive_token_iter<'a, T, I: Iterator<Item = &'a T>>(
  iter: I
) -> ConsecutiveTokenIterator<'a, T, I> {
  ConsecutiveTokenIterator { iter: iter, last: None }
}

/// Iterates over every Token from the supplied iterator, and returns the 
/// immediate following token. Returns None for the following token on the last 
/// token.
pub struct ConsecutiveTokenMutIterator<'a, T: 'a, I: Iterator<Item = &'a mut T>> {
  iter: I,
  last: Option<*mut T>
}

impl<'a, T: 'a, I: Iterator<Item = &'a mut T>> Iterator
for ConsecutiveTokenMutIterator<'a, T, I>
{
  type Item = (&'a mut T, Option<&'a mut T>);

  #[inline]
  fn next(&mut self) -> Option<(&'a mut T, Option<&'a mut T>)> {
    match self.last {
      Some(tok) => {
        self.last = self.iter.next().map(|t| t as *mut T);

        unsafe { Some((&mut *tok, self.last.map(|t| &mut *t))) }
      }
      None => match self.iter.next() {
        Some(tok) => {
          self.last = self.iter.next().map(|t| t as *mut T);

          unsafe { Some((&mut *tok, self.last.map(|t| &mut *t))) }
        }
        None => None
      }
    }
  }

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    self.iter.size_hint()
  }
}

#[inline]
pub fn consecutive_token_iter_mut<'a, T, I: Iterator<Item = &'a mut T>>(
  iter: I
) -> ConsecutiveTokenMutIterator<'a, T, I> {
  ConsecutiveTokenMutIterator { iter: iter, last: None }
}