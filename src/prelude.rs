use std::ops::Deref;

use phf::Set;
use phf::Map;


/// Defines a set of punctuation that can end a sentence.
pub trait DefinesSentenceEndings {
  fn sentence_endings() -> &'static Set<char>;

  #[inline(always)] fn is_sentence_ending(c: &char) -> bool { 
    Self::sentence_endings().contains(c)
  }
}


/// Defines a set of punctuation that can occur within a word.
pub trait DefinesInternalPunctuation {
  fn internal_punctuation() -> &'static Set<char>;

  #[inline(always)] fn is_internal_punctuation(c: &char) -> bool {
    Self::internal_punctuation().contains(c)
  }
}


/// Defines a set of characters that can not occur inside of a word.
pub trait DefinesNonWordCharacters {
  fn nonword_chars() -> &'static Set<char>;

  #[inline(always)] fn is_nonword_char(c: &char) -> bool {
    Self::nonword_chars().contains(c)
  }
}


/// Defines punctuation that can occur within a sentence.
pub trait DefinesPunctuation {
  fn punctuation() -> &'static Set<char>;

  #[inline(always)] fn is_punctuation(c: &char) -> bool {
    Self::punctuation().contains(c)
  }
}


/// Defines a set of a characters that can not start a word.
pub trait DefinesNonPrefixCharacters {
  fn nonprefix_chars() -> &'static Set<char>; 

  #[inline(always)] fn is_nonprefix_char(c: &char) -> bool {
    Self::nonprefix_chars().contains(c)
  }
}


static INTERNAL_PUNCT  : Set<char> = phf_set![',', ':', ';', '\u{2014}'];
static SENTENCE_ENDINGS: Set<char> = phf_set!['.', '?', '!'];
static PUNCTUATION     : Set<char> = phf_set![';', ':', ',', '.', '!', '?'];
static NONWORD_CHARS   : Set<char> = phf_set![
  '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '['
];
static NONPREFIX_CHARS : Set<char> = phf_set![
  '(', '"', '`', '{', '[', ':', ';', '&', '#', '*', '@', ')', '}', ']', '-', ','
];


/// Mixin that will give the default implementations for 
/// `DefinesSentenceEndings`, `DefinesInternalPunctuation`, 
/// `DefinesNonWordCharacter`, `DefinesEndingPunctuation`,
/// and `DefinesNonPrefixCharacters`.
pub trait DefaultCharacterDefinitions { }

impl<T> DefinesSentenceEndings for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn sentence_endings() -> &'static Set<char> { 
    &SENTENCE_ENDINGS 
  }
}

impl<T> DefinesInternalPunctuation for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn internal_punctuation() -> &'static Set<char> {
    &INTERNAL_PUNCT
  }
}

impl<T> DefinesNonWordCharacters for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn nonword_chars() -> &'static Set<char> {
    &NONWORD_CHARS
  }
}

impl<T> DefinesPunctuation for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn punctuation() -> &'static Set<char> {
    &PUNCTUATION
  }
}

impl<T> DefinesNonPrefixCharacters for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn nonprefix_chars() -> &'static Set<char> {
    &NONPREFIX_CHARS
  }
}


pub trait CaseInsensitiveEq {
  /// Equals method that ignores case.
  fn case_insensitive_eq(&self, other: &Self) -> bool;
}

impl<T> CaseInsensitiveEq for T where T : Deref<Target = str> {
  #[inline(always)] fn case_insensitive_eq(&self, other: &T) -> bool {
    self
      .chars()
      .flat_map(|c| c.to_lowercase())
      .zip(other.chars().flat_map(|c| c.to_lowercase()))
      .fold(true, |acc, (c0, c1)| acc && c0 == c1)
  }
}


pub type OrthographicContext = u8;


/// Context that a token can be in.
#[derive(Debug, Eq, PartialEq)]
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


pub const BEG_UC: OrthographicContext = 0b00000010;
pub const MID_UC: OrthographicContext = 0b00000100;
pub const UNK_UC: OrthographicContext = 0b00001000;
pub const BEG_LC: OrthographicContext = 0b00010000;
pub const MID_LC: OrthographicContext = 0b00100000;
pub const UNK_LC: OrthographicContext = 0b01000000;
pub const ORT_UC: OrthographicContext = BEG_UC | MID_UC | UNK_UC; 
pub const ORT_LC: OrthographicContext = BEG_LC | MID_LC | UNK_LC;


/// Map mapping a combination of LetterCase and OrthographyPosition 
/// to an OrthographicConstant describing orthographic attributes about the 
/// token. The chars (in ASCII) map to the result of ORing certains
/// OrthographyPosition and LetterCase with one another.
pub static ORTHO_MAP: Map<u8, OrthographicContext> = phf_map! {
  b'B' => BEG_UC, // 66
  b'"' => MID_UC, // 34
  b'b' => UNK_UC, // 98
  b'A' => BEG_LC, // 65
  b'!' => MID_LC, // 33
  b'a' => UNK_LC  // 97
};


/// Possible cases a letter can be in. OR (|) can be applied to these with 
/// a OrthographyPosition to get a corrosponding OrthographicContext from 
/// OrthoMap.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum LetterCase {
  Upper,
  Lower,
  Unknown,
}

impl LetterCase {
  #[inline(always)] pub fn as_byte(&self) -> u8 {
    match *self {
      LetterCase::Upper   => 0b00000010,
      LetterCase::Lower   => 0b00000001,
      LetterCase::Unknown => 0b00000011
    }
  }
}


#[test] fn case_insensitive_eq_str_test() {
  assert!("ABC".case_insensitive_eq(&"abc"));
  assert!("hElLo TeSt".case_insensitive_eq(&"hello TEST"));
  assert!("".case_insensitive_eq(&""));
  assert!("ÀÁÂÃÄ".case_insensitive_eq(&"àáâãä"));
}