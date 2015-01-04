use std::rc::Rc;
use std::ops::{Slice, Deref};
use std::hash::Hash;
use std::borrow::BorrowFrom;
use std::fmt::{Show, Formatter, Result};

use phf::Set;
use xxhash::XXState;

#[cfg(debug)]
enum PunktCategory {
  Abbreviation,
  SentenceBreak,
  Ellipsis
}

#[cfg(debug)]
impl Show for PunktCategory {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(
      f,
      "{}",
      match *self {
        PunktCategory::Abbreviation  => "<A>",
        PunktCategory::SentenceBreak => "<S>",
        PunktCategory::Ellipsis      => "<E>"
      })
  }
}

/// Characters that denote a sentence boundary.
pub static SENTENCE_TERMINATORS: Set<char> = phf_set! { '!', '.', '?' };

/// A slice from a document that has a start position (its index),
/// and an associated length. From these properties, the original 
/// document slice can be recreated (given the original document)
/// still is in scope.
pub trait DocumentIndexedSlice {
  fn len(&self) -> uint;
  fn start(&self) -> uint;
}

/// An object that is a slice from a document. The slice can be 
/// retrieved from the original document.
pub trait DocumentSlice {
  fn as_doc_slice<'a>(&self, doc: &'a str) -> &'a str;
}

/// All indexed document slices are document slices. The document slice
/// can be recreated by slicing from start to start + len.
impl<T: DocumentIndexedSlice> DocumentSlice for T {
  #[inline]
  fn as_doc_slice<'a>(&self, doc: &'a str) -> &'a str {
    doc.slice(self.start(), self.start() + self.len())
  }
}
/// Possible cases a letter can be in. OR (|) can be applied to these with 
/// a OrthographyPosition to get a corrosponding OrthographicContext from 
/// OrthoMap.
#[derive(Show, Eq, PartialEq, Copy)]
pub enum LetterCase {
  Upper,
  Lower,
  Unknown,
}

impl LetterCase {
  pub fn as_byte(&self) -> u8 {
    match *self {
      LetterCase::Upper   => 0b00000010,
      LetterCase::Lower   => 0b00000001,
      LetterCase::Unknown => 0b00000011
    }
  }
}

/// A tokenizer used by a PunktTokenizer.
/// The flags represent certain traits about where the word token and its location
/// within the string as well as properties about its type (ellipsis, abbreviation...).
/// The token holds onto the position where it began in the document, rather than a 
/// slice of the token itself. This allows the future code to be more flexible. The
/// user should BARELY have to interact with PunktToken if at all.
pub struct PunktToken {
  start: uint,
  inner: String,
  flags: u16
}

// Flags that can be set. These describe certain properties about the Token.
const HAS_FINAL_PERIOD  : u16 = 0b0000000000000001;
const IS_ELLIPSIS       : u16 = 0b0000000000000010;
const IS_INITIAL        : u16 = 0b0000000000000100;
const IS_ABBREV         : u16 = 0b0000000000001000;
const IS_PARAGRAPH_START: u16 = 0b0000000000010000;
const IS_NEWLINE_START  : u16 = 0b0000000000100000;
const IS_SENTENCE_BREAK : u16 = 0b0000000001000000;
const IS_NUMERIC        : u16 = 0b0000000010000000;
const IS_NON_PUNCT      : u16 = 0b0000000100000000;
const IS_UPPERCASE      : u16 = 0b0000001000000000;
const IS_LOWERCASE      : u16 = 0b0000010000000000;
const HAS_DIGIT         : u16 = 0b0000100000000000;
const HAS_PUNCT         : u16 = 0b0001000000000000;
const OVERRIDE_BREAK    : u16 = 0b1000000000000000;

/// This string is defined by NLTK. It's necessary as all numeric data 
/// from NLTK data uses this string as a representation. 
static NUMBER_STR            : &'static str = "##number##";
static NUMBER_STR_WITH_PERIOD: &'static str = "##number##.";

impl PunktToken {
  /// Create a new PunktToken. The PunktToken is almost purely a data object and 
  /// does not decide much about the properties of a token. Most of the properties 
  /// of the PunktToken must be determined by the parsing algorithm. A PunktToken check
  /// to see if it is numeric, and if it ends in a period or sentence terminator is
  /// done upon creation. The algorithm requires a normalized version of the token 
  /// to be created. A token can never be EMPTY.
  pub fn new(
    tok: &str, 
    start: uint,
    is_ellipsis: bool, 
    is_paragraph_start: bool, 
    is_newline_start: bool
  ) -> PunktToken {
    // The parser should never try to create empty tokens. They don't matter for the 
    // Punkt algorithm.
    debug_assert!(tok.len() > 0);

    let mut flags = 0b0000000000000000;
    let first_char = tok.char_at(0);

    if is_ellipsis { 
      flags |= IS_ELLIPSIS; 
    }

    if is_paragraph_start { 
      flags |= IS_PARAGRAPH_START; 
    }

    if is_newline_start { 
      flags |= IS_NEWLINE_START 
    }

    // If the token is only a character, and its only token 
    // is a sentence break, it is a sentence break.
    if tok.len() == 1 &&
       SENTENCE_TERMINATORS.contains(&first_char) 
    { 
      flags |= IS_SENTENCE_BREAK 
    }

    if PunktToken::is_token_numeric(tok) {
      flags |= IS_NUMERIC;
    } else if PunktToken::is_token_initial(tok) {
      flags |= IS_INITIAL;
    } 

    if tok.ends_with(".") {
      flags |= HAS_FINAL_PERIOD; 
    }

    if first_char.is_uppercase() {
      flags |= IS_UPPERCASE;
    } else if first_char.is_lowercase() {
      flags |= IS_LOWERCASE;
    }

    // Add a period to any tokens without a period. This is an optimization 
    // to avoid creating an entirely new token when searching through the HashSet.
    let mut inner = if flags & HAS_FINAL_PERIOD > 0 { 
      String::with_capacity(tok.len()) 
    } else {
      String::with_capacity(tok.len() + 1)
    };

    // Checks if a token has any non-alphabetic chars or digits.
    // If so, we define alphabetic tokens as ones that do not
    // contains punctuation or digits. Also builds the string.
    for c in tok.chars() {
      inner.push(c.to_lowercase());

      if c.is_digit(10) {
        flags |= HAS_DIGIT;
      } else if c.is_alphabetic() || c == '_' {
        flags |= IS_NON_PUNCT;
      } else {
        flags |= HAS_PUNCT;
      }
    }

    if flags & HAS_FINAL_PERIOD == 0 {
      inner.push('.');
    }

    // The only flag that IS NOT set is the abbreviation flag. This flag can't 
    // be determined until later (when the likelihood of a token being a abbreviation
    // is calculated). An override flag can be set later to override a sentence 
    // starter decision.
    PunktToken { 
      start: start,
      inner: inner, 
      flags: flags 
    }
  }

  /// A number can start with a negative sign ('-'), and be followed by digits
  /// or isolated periods, commas, or dashes. 
  /// Note: It's assumed that multi-chars are taken out the input when creating word 
  /// tokens, so a numeric word token SHOULD not have a multi-char within it as
  /// its received. This assumption should be fulfilled by the parser generating 
  /// these word tokens. If it isn't some weird outputs are possible (such as "5.4--5").
  #[inline]
  fn is_token_numeric(tok: &str) -> bool {
    let mut is_numeric = true;
    let mut digit_found = false;
    let mut pos = 0u;

    for c in tok.chars() {
      match c {
        '-' if pos == 0 || digit_found    => (),
          _ if c.is_digit(10)             => digit_found = true, 
        ',' | '.' if pos == 0 || pos == 1 => (),
        ',' | '.' | '-' if digit_found    => (),
          _                               => { is_numeric = false; break }
      }

      pos += c.len_utf8(); 
    }

    is_numeric && digit_found
  }

  /// Tests if the token is an initial. 
  #[inline]
  fn is_token_initial(tok: &str) -> bool {
    let mut is_initial = false;

    if tok.len() > 1 {
      let mut iter = tok.chars();

      match (iter.next(), iter.next())
      {
        (Some(c), Some('.')) if c.is_alphabetic() => is_initial = true,
                                                _ => ()
      }
    }

    is_initial
  }
}

impl PunktToken {
  /// This returns the ORIGINAL token that is normalized (all lowercase). 
  #[inline]
  pub fn token(&self) -> &str {
    if self.has_final_period_internal() {
      self.inner.as_slice()
    } else {
      self.inner.slice_to_or_fail(&(self.inner.len() - 1))
    }
  }

  /// Returns the ORIGINAL token with a period.
  #[inline]
  pub fn token_with_period(&self) -> &str {
    self.inner.as_slice()
  }

  /// Returns the ORIGINAL token without a period.
  #[inline]
  pub fn token_without_period(&self) -> &str {
    if self.has_final_period_internal() {
      self.token().slice_to(self.inner.len() - 1)
    } else {
      self.token()
    }
  }

  /// Returns the type. Usually this is the token, unless the string is 
  /// numeric. Otherwise it returns '##number##'. This is a constant string 
  /// that NLTK uses and MUST be used in order to properly use the data that NLTK
  /// provides.
  #[inline]
  pub fn typ(&self) -> &str {
    if self.is_numeric() { NUMBER_STR } else { self.token() }
  }

  /// Returns the type with a period appended onto it.
  #[inline]
  pub fn typ_with_period(&self) -> &str {
    if self.is_numeric() { 
      NUMBER_STR_WITH_PERIOD 
    } else { 
      self.token_with_period() 
    }
  }

  /// Returns the type without any period appended it.
  #[inline]
  pub fn typ_without_period(&self) -> &str {
    if self.len() > 1 && self.has_final_period() {
      self.typ_with_period().slice_to(self.typ_with_period().len() - 1)
    } else {
      self.typ()
    }
  }

  /// Returns the type without a period if it is a sentence break, 
  /// otherwise returns the regular type.
  #[inline]
  pub fn typ_without_break_or_period(&self) -> &str {
    if self.is_sentence_break() {
      self.typ_without_period()
    } else {
      self.typ()
    }
  }

  /// Marks this token as an abbreviation or not as one depending on the specified 
  /// boolean.
  pub fn set_abbrev(&mut self, b: bool) {
    if b { 
      self.flags |= IS_ABBREV 
    } 
    else { 
      self.flags ^= IS_ABBREV 
    }
  }

  /// Marks this token as a sentence break overriding any decision made during the
  /// intialization process.
  pub fn set_sentence_break(&mut self, b: bool) {
    if b {
      self.flags |= OVERRIDE_BREAK
    } else {
      self.flags ^= OVERRIDE_BREAK
    }
  }

  /// Returns whether or not a token is initial. This is the cached result from initialization
  #[inline]
  pub fn is_initial(&self) -> bool {
    IS_INITIAL & self.flags != 0
  }

  /// Returns whether or not a token is an ellipsis. This is the cached result from 
  /// initialization.
  #[inline]
  pub fn is_ellipsis(&self) -> bool { 
    IS_ELLIPSIS & self.flags != 0 
  }

  /// Returns whether or not a token has a final period. Used internally, since abbreviations
  /// and ellipsis are considered to NOT have a final periods.
  #[inline]
  fn has_final_period_internal(&self) -> bool {
    HAS_FINAL_PERIOD & self.flags != 0
  }
  
  /// Checks if the token has a final period and is NOT an abbreviation or ellipsis.
  #[inline]
  pub fn has_final_period(&self) -> bool { 
    self.has_final_period_internal() && 
    !self.is_abbrev() && 
    !self.is_ellipsis()
  }

  /// Returns whether or not a token is the start of a paragraph. This is generally
  /// determined during parsing.
  #[inline]
  pub fn is_paragraph_start(&self) -> bool { 
    IS_PARAGRAPH_START & self.flags != 0 
  }

  /// Returns whether or not a token is the start of a newline. This is generally determined
  /// during parsing.
  #[inline]
  pub fn is_newline_start(&self) -> bool { 
    IS_NEWLINE_START & self.flags != 0 
  }

  /// Used internally to check if a token is a sentence break. The actual definition of an 
  /// actual sentence break varies depending on some other flags.
  #[inline]
  fn is_sentence_break_internal(&self) -> bool {
    IS_SENTENCE_BREAK & self.flags != 0
  }

  /// Returns whether or not a token is a sentence break.
  #[inline]
  pub fn is_sentence_break(&self) -> bool { 
    self.is_sentence_break_internal() ||  
    self.has_final_period() || 
    OVERRIDE_BREAK & self.flags != 0
  }

  /// Returns whether or not a token is numeric. 
  #[inline]
  pub fn is_numeric(&self) -> bool {
    IS_NUMERIC & self.flags != 0
  }

  /// Returns whether or not a token is an abbreviation.
  #[inline]
  pub fn is_abbrev(&self) -> bool {
    IS_ABBREV & self.flags != 0
  }

  /// Returns whether or not a token is alphabetic.
  #[inline]
  pub fn is_alphabetic(&self) -> bool {
    HAS_PUNCT & self.flags == 0 && !self.is_numeric()
  }

  /// Returns whether or not a token begins with an uppercase character.
  #[inline]
  pub fn is_uppercase(&self) -> bool {
    IS_UPPERCASE & self.flags != 0
  }

  /// Returns whether or not a token begins with a lowercase character.
  #[inline]
  pub fn is_lowercase(&self) -> bool {
    IS_LOWERCASE & self.flags != 0
  }

  /// Returns whether or not a token has only alphabetic characters or is numeric.
  #[inline]
  pub fn is_non_punct(&self) -> bool {
    IS_NON_PUNCT & self.flags != 0 || self.is_numeric()
  }

  /// Returns an enumeration representing the casing of the first character of the token.
  #[inline]
  pub fn first_case(&self) -> LetterCase {
    if self.is_uppercase() {
      LetterCase::Upper
    } else if self.is_lowercase() {
      LetterCase::Lower
    } else {
      LetterCase::Unknown
    }
  }

  /// Returns the number of characters in the ORIGINAL token. This is a O(n) operation. 
  /// Only use if TOTALLY necessary.
  #[inline]
  pub fn char_len(&self) -> uint {
    self.token().chars().count()
  }

  /// Gets the tag associated with the token. This is used for 
  /// debugging.
  #[cfg(debug)]
  fn tags(&self) -> Vec<PunktCategory> {
    let mut tags = Vec::with_capacity(3);

    if self.is_abbrev() {
      tags.push(PunktCategory::Abbreviation);
    }

    if self.is_sentence_break() {
      tags.push(PunktCategory::SentenceBreak);
    }

    if self.is_ellipsis() {
      tags.push(PunktCategory::Ellipsis);
    }

    tags
  }
}

#[cfg(debug)]
impl Show for PunktToken {
  fn fmt(&self, fmt: &mut Formatter) -> Result {
    write!(fmt, "{} {}", self.typ(), self.tags()) 
  }
}

#[cfg(not(debug))]
impl Show for PunktToken {
  fn fmt(&self, fmt: &mut Formatter) -> Result {
    write!(fmt, "{}", self.typ()) 
  }
}

impl Eq for PunktToken { }

impl PartialEq for PunktToken {
  /// Compares the types of tokens with one another.
  #[inline]
  fn eq(&self, other: &PunktToken) -> bool {
    self.typ() == other.typ()
  }
}

impl Hash<XXState> for PunktToken {
  #[inline]
  fn hash(&self, state: &mut XXState) {
    self.typ().hash(state)
  }
}

impl DocumentIndexedSlice for PunktToken {
  /// The starting index of the token in the original document.
  #[inline]
  fn start(&self) -> uint {
    self.start
  }

  /// Returns the length of the ORIGINAL token.
  #[inline]
  fn len(&self) -> uint {
    self.token().len()
  }
}

impl BorrowFrom<Rc<PunktToken>> for str {
  #[inline]
  fn borrow_from(owned: &Rc<PunktToken>) -> &str {
    owned.deref().typ()
  }
}

impl Deref for PunktToken {
  type Target = str;

  #[inline]
  fn deref(&self) -> &str {
    self.token_without_period()
  }
}

#[cfg(test)]
mod punkt_token_tests {
  use super::PunktToken;

  #[test]
  fn smoke_test_is_numeric_true() {
    let nums = vec!("3.4", "-,4.5", "-4-5", "-4,567.487", "4,456,678", "4.4.4.4.10", "11");

    for n in nums.iter() {
      assert!(PunktToken::is_token_numeric(*n), "failed {}", *n);
    }
  }

  #[test]
  fn smoke_test_is_numeric_false() {
    let nums = vec!("hello", "4,a", "4.a.C", "5-A", "-5-AB", "-5,b", ",");

    for n in nums.iter() {
      assert!(!PunktToken::is_token_numeric(*n), "failed {}", *n);
    }
  }

  #[test]
  fn smoke_test_set_abbrev() {
    let mut tok = PunktToken::new("hello", 0, false, false, false);

    assert!(!tok.is_abbrev());

    tok.set_abbrev(true);

    assert!(tok.is_abbrev());

    tok.set_abbrev(false);

    assert!(!tok.is_abbrev());
  }

  #[test]
  fn smoke_test_flags() {
    let tok = PunktToken::new("Test.", 0, false, false, false);

    assert!(!tok.is_abbrev());
    assert!(!tok.is_initial());
    assert!(tok.is_uppercase());
    assert!(!tok.is_lowercase());
    assert!(tok.has_final_period());
    assert!(tok.is_non_punct());
  }

  #[test]
  fn smoke_test_is_alphabetic_pass() {
    let toks = vec!("abc", "hello", "CaSeS", "TEST", "OkAy");

    for t in toks.iter() {
      let pt = PunktToken::new(*t, 0, false, false, false);

      assert!(pt.is_alphabetic());
    }
  }

  #[test]
  fn smoke_test_is_alphabetic_fail() {
    let toks = vec!("abc!", "test11!", "dashed-token", "nOn!!");

    for t in toks.iter() {
      let pt = PunktToken::new(*t, 0, false, false, false);

      assert!(!pt.is_alphabetic(), "failed: {}", t);
    }
  }
}
