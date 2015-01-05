#![feature(macro_rules, phase, default_type_params, associated_types)]

#![allow(dead_code)]

/// # Overview
///
/// Implementation of Tibor Kiss' and Jan Strunk's Punkt algorithm for sentence tokenization.
/// Includes a word tokenizer that tokenizes words based on regexes defined in Python's 
/// NLTK library. Results have been compared with small and large texts that have been 
/// tokenized with NLTK's library. For usage, check out `PunktSentenceTokenizer`.
///
/// # Training
///
/// Training data can be provided to a `PunktSentenceTokenizer` for better results. Data 
/// can be acquired manually by training with a `PunktTrainer`, or using already compiled
/// data from NLTK (example: `PunktData::english()`).
///
/// Training parameters can be specified using `PunktTrainerParameters`. The defaults 
/// are from NLTK, but customized threshold values and flags can be set.

extern crate xxhash;
extern crate phf;
#[phase(plugin)] extern crate phf_mac;

extern crate "rustc-serialize" as rustc_serialize;
extern crate freqdist;
extern crate collections;
#[cfg(test)] extern crate test;
#[phase(plugin, link)] extern crate log;

pub mod token;
pub mod tokenizer;
//pub mod trainer;

/*
mod prelude {
  use trainer::Data;
  use tokenizer::Token;

  pub trait PunktFirstPassAnnotater {
    fn data(&self) -> &Data;

    /// Peforms a first pass annotation on a Token.
    #[inline]
    fn annotate_first_pass(&self, t: &mut Token) {
      let is_split_abbrev = t
        .typ()
        .rsplitn(1, '-')
        .next()
        .map(|s| self.data().contains_abbrev(s))
        .unwrap_or(false);

      // Since this is where abbreviations are decided, `has_final_period` should 
      // return whether or not the original token has a period.
      if t.has_final_period() && 
         !t.is_ellipsis() &&
         (self.data().contains_abbrev(t.typ_without_period()) || 
          is_split_abbrev)
      {
        t.set_abbrev(true);
      }
    }
  }
}
*/

mod ortho {
  use phf::Map;

  pub type OrthographicContext = u8;

  /// Context that a token can be in.
  #[derive(Show, Eq, PartialEq)]
  pub enum OrthographyPosition {
    Initial,
    Internal,
    Unknown
  }

  impl OrthographyPosition {
    pub fn as_byte(&self) -> u8 {
      match *self {
        OrthographyPosition::Initial   => 0b01000000,
        OrthographyPosition::Internal  => 0b00100000,
        OrthographyPosition::Unknown   => 0b01100000
      }
    }
  }

  // Orthographic Constants
  // 
  // Used to describe the orthographic contexts in which a 
  // word can occur.
  pub const BEGIN_UC    : OrthographicContext = 0b00000010;
  pub const MIDDLE_UC   : OrthographicContext = 0b00000100;
  pub const UNKNOWN_UC  : OrthographicContext = 0b00001000;
  pub const BEGIN_LC    : OrthographicContext = 0b00010000;
  pub const MIDDLE_LC   : OrthographicContext = 0b00100000;
  pub const UNKNOWN_LC  : OrthographicContext = 0b01000000;
  pub const ORTHO_UC    : OrthographicContext = BEGIN_UC | MIDDLE_UC | UNKNOWN_UC; 
  pub const ORTHO_LC    : OrthographicContext = BEGIN_LC | MIDDLE_LC | UNKNOWN_LC;

  /// Map mapping a combination of LetterCase and OrthographyPosition 
  /// to an OrthographicConstant describing orthographic attributes about the 
  /// token. The chars (in ASCII) map to the result of ORing certains
  /// OrthographyPosition and LetterCase with one another.
  pub static ORTHO_MAP: Map<u8, OrthographicContext> = phf_map! {
    b'B' => BEGIN_UC,   // 66
    b'"' => MIDDLE_UC,  // 34
    b'b' => UNKNOWN_UC, // 98
    b'A' => BEGIN_LC,   // 65
    b'!' => MIDDLE_LC,  // 33
    b'a' => UNKNOWN_LC  // 97
  };
}
 
/*
mod iter {
  /// Iterates over every PuntkToken from the supplied iterator and returns 
  /// the immediate following token. Returns None for the following token on the 
  /// last token.
  pub struct ConsecutiveTokenIterator<'a, T: 'a, I: Iterator<&'a T>> {
    iter: I,
    last: Option<&'a T>
  }

  impl<'a, T: 'a, I: Iterator<&'a T>> Iterator<(&'a T, Option<&'a T>)>
  for ConsecutiveTokenIterator<'a, T, I>
  {
    /// Returns pairs of consecutive tokens.
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

  /// Constructor for a consecutive token iterator.
  #[inline]
  pub fn consecutive_token_iter<'a, T, I: Iterator<&'a T>>(
    iter: I
  ) -> ConsecutiveTokenIterator<'a, T, I> {
    ConsecutiveTokenIterator { iter: iter, last: None }
  }

  /// Iterates over every Token from the supplied iterator, and returns the 
  /// immediate following token. Returns None for the following token on the last 
  /// token.
  pub struct ConsecutiveTokenMutIterator<'a, T: 'a, I: Iterator<&'a mut T>> {
    iter: I,
    last: Option<*mut T>
  }

  impl<'a, T: 'a, I: Iterator<&'a mut T>> Iterator<(&'a mut T, Option<&'a mut T>)>
  for ConsecutiveTokenMutIterator<'a, T, I>
  {
    /// Returns pairs of consecutive tokens.
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

  /// Constructor for a mutable consecutive token iterator.
  #[inline]
  pub fn consecutive_token_iter_mut<'a, T, I: Iterator<&'a mut T>>(
    iter: I
  ) -> ConsecutiveTokenMutIterator<'a, T, I> {
    ConsecutiveTokenMutIterator { iter: iter, last: None }
  }
}
*/
