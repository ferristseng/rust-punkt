// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use phf;

/// Type for character sets.
pub type Set<T> = phf::Set<T>;

/// Defines a set of punctuation that can end a sentence.
pub trait DefinesSentenceEndings {
  /// The set of characters that constitute a sentence ending.
  const SENTENCE_ENDINGS: &'static Set<char> = &phf_set!['.', '?', '!'];

  /// Checks if a character is a sentence ending.
  #[inline]
  fn is_sentence_ending(c: &char) -> bool {
    Self::SENTENCE_ENDINGS.contains(c)
  }
}

/// Defines a set of punctuation that can occur within a word.
pub trait DefinesInternalPunctuation {
  /// The set of legal punctuation characters that can occur within a word.
  const INTERNAL_PUNCTUATION: &'static Set<char> = &phf_set![',', ':', ';', '\u{2014}'];

  /// Checks if a character is a legal punctuation character that can occur
  /// within a word.
  #[inline]
  fn is_internal_punctuation(c: &char) -> bool {
    Self::INTERNAL_PUNCTUATION.contains(c)
  }
}

/// Defines a set of characters that can not occur inside of a word.
pub trait DefinesNonWordCharacters {
  /// The set of characters that can not occur inside of a word.
  const NONWORD_CHARS: &'static Set<char> = &phf_set![
    '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '['
  ];

  /// Checks if a character is one that can not occur inside of a word.
  #[inline]
  fn is_nonword_char(c: &char) -> bool {
    Self::NONWORD_CHARS.contains(c)
  }
}

/// Defines punctuation that can occur within a sentence.
pub trait DefinesPunctuation {
  /// The set of legal punctuation marks.
  const PUNCTUATION: &'static Set<char> = &phf_set![';', ':', ',', '.', '!', '?'];

  /// Checks if a characters is a legal punctuation mark.
  #[inline]
  fn is_punctuation(c: &char) -> bool {
    Self::PUNCTUATION.contains(c)
  }
}

/// Defines a set of a characters that can not start a word.
pub trait DefinesNonPrefixCharacters {
  /// The set of characters that can not start a word.
  const NONPREFIX_CHARS: &'static Set<char> = &phf_set![
    '(', '"', '`', '{', '[', ':', ';', '&', '#', '*', '@', ')', '}', ']', '-', ','
  ];

  /// Checks if a character can start a word.
  #[inline]
  fn is_nonprefix_char(c: &char) -> bool {
    Self::NONPREFIX_CHARS.contains(c)
  }
}

/// Configurable parameters for a trainer.
pub trait TrainerParameters: DefinesSentenceEndings + DefinesInternalPunctuation {
  /// Lower bound score for a token to be considered an abbreviation.
  const ABBREV_LOWER_BOUND: f64 = 0.3;

  /// Upper bound score for a token to be considered an abbreviation.
  const ABBREV_UPPER_BOUND: f64 = 5f64;

  /// Disables the abbreviation penalty which exponentially penalizes occurances
  /// of words without a trailing period.
  const IGNORE_ABBREV_PENALTY: bool = false;

  /// Lower bound score for two tokens to be considered a collocation
  const COLLOCATION_LOWER_BOUND: f64 = 7.88;

  /// Lower bound score for a token to be considered a sentence starter.
  const SENTENCE_STARTER_LOWER_BOUND: f64 = 30f64;

  /// Include all pairs where the first token ends with a period.
  const INCLUDE_ALL_COLLOCATIONS: bool = false;

  /// Include all pairs where the first is an abbreviation. Overridden by
  /// `include_all_collocations`.
  const INCLUDE_ABBREV_COLLOCATIONS: bool = false;

  /// Minimum number of times a bigram appears in order to be considered a
  /// collocation.
  const COLLOCATION_FREQUENCY_LOWER_BOUND: f64 = 1f64;
}

/// Standard settings for all tokenizers, and trainers.
pub struct Standard;

impl DefinesInternalPunctuation for Standard {}
impl DefinesNonPrefixCharacters for Standard {}
impl DefinesNonWordCharacters for Standard {}
impl DefinesPunctuation for Standard {}
impl DefinesSentenceEndings for Standard {}
impl TrainerParameters for Standard {}

pub type OrthographicContext = u8;

#[derive(PartialEq, Eq)]
pub enum OrthographyPosition {
  Initial,
  Internal,
  Unknown,
}

impl OrthographyPosition {
  pub fn as_byte(&self) -> u8 {
    match *self {
      OrthographyPosition::Initial => 0b01000000,
      OrthographyPosition::Internal => 0b00100000,
      OrthographyPosition::Unknown => 0b01100000,
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

/// Map relating a combination of LetterCase and OrthographyPosition
/// to an OrthographicConstant describing orthographic attributes about the
/// token. The chars (in ASCII) map to the result of ORing the byte
/// representation of an OrthographyPosition and LetterCase together.
pub static ORTHO_MAP: phf::Map<u8, OrthographicContext> = phf_map! {
  b'B' => BEG_UC, // 66
  b'"' => MID_UC, // 34
  b'b' => UNK_UC, // 98
  b'A' => BEG_LC, // 65
  b'!' => MID_LC, // 33
  b'a' => UNK_LC  // 97
};

pub enum LetterCase {
  Upper,
  Lower,
  Unknown,
}

impl LetterCase {
  #[inline(always)]
  pub fn as_byte(&self) -> u8 {
    match *self {
      LetterCase::Upper => 0b00000010,
      LetterCase::Lower => 0b00000001,
      LetterCase::Unknown => 0b00000011,
    }
  }
}
