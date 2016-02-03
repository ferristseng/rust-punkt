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
  fn sentence_endings() -> &'static Set<char>;

  /// Checks if a character is a sentence ending.
  #[inline(always)]
  fn is_sentence_ending(c: &char) -> bool {
    Self::sentence_endings().contains(c)
  }
}


/// Defines a set of punctuation that can occur within a word.
pub trait DefinesInternalPunctuation {
  /// The set of legal punctuation characters that can occur within a word.
  fn internal_punctuation() -> &'static Set<char>;

  /// Checks if a character is a legal punctuation character that can occur
  /// within a word.
  #[inline(always)]
  fn is_internal_punctuation(c: &char) -> bool {
    Self::internal_punctuation().contains(c)
  }
}


/// Defines a set of characters that can not occur inside of a word.
pub trait DefinesNonWordCharacters {
  /// The set of characters that can not occur inside of a word.
  fn nonword_chars() -> &'static Set<char>;

  /// Checks if a character is one that can not occur inside of a word.
  #[inline(always)]
  fn is_nonword_char(c: &char) -> bool {
    Self::nonword_chars().contains(c)
  }
}


/// Defines punctuation that can occur within a sentence.
pub trait DefinesPunctuation {
  /// The set of legal punctuation marks.
  fn punctuation() -> &'static Set<char>;

  /// Checks if a characters is a legal punctuation mark.
  #[inline(always)]
  fn is_punctuation(c: &char) -> bool {
    Self::punctuation().contains(c)
  }
}


/// Defines a set of a characters that can not start a word.
pub trait DefinesNonPrefixCharacters {
  /// The set of characters that can not start a word.
  fn nonprefix_chars() -> &'static Set<char>;

  /// Checks if a character can start a word.
  #[inline(always)]
  fn is_nonprefix_char(c: &char) -> bool {
    Self::nonprefix_chars().contains(c)
  }
}


/// Configurable parameters for a trainer.
pub trait TrainerParameters : DefinesSentenceEndings + DefinesInternalPunctuation {
  /// Lower bound score for a token to be considered an abbreviation.
  fn abbrev_lower_bound() -> f64;

  /// Upper bound score for a token to be considered an abbreviation.
  fn abbrev_upper_bound() -> f64;

  /// Disables the abbreviation penalty which exponentially penalizes occurances
  /// of words without a trailing period.
  fn ignore_abbrev_penalty() -> bool;

  /// Lower bound score for two tokens to be considered a collocation
  fn collocation_lower_bound() -> f64;

  /// Lower bound score for a token to be considered a sentence starter.
  fn sentence_starter_lower_bound() -> f64;

  /// Include all pairs where the first token ends with a period.
  fn include_all_collocations() -> bool;

  /// Include all pairs where the first is an abbreviation. Overridden by
  /// `include_all_collocations`.
  fn include_abbrev_collocations() -> bool;

  /// Minimum number of times a bigram appears in order to be considered a
  /// collocation.
  fn collocation_frequency_lower_bound() -> f64;
}


static INTERNAL_PUNCT: Set<char> = phf_set![',', ':', ';', '\u{2014}'];
static SENTENCE_ENDINGS: Set<char> = phf_set!['.', '?', '!'];
static PUNCTUATION: Set<char> = phf_set![';', ':', ',', '.', '!', '?'];
static NONWORD_CHARS: Set<char> = phf_set!['?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'',
                                           '(', '{', '['];
static NONPREFIX_CHARS: Set<char> = phf_set!['(', '"', '`', '{', '[', ':', ';', '&', '#', '*',
                                             '@', ')', '}', ']', '-', ','];


/// Mixin that will give the default implementations for
/// `DefinesSentenceEndings`, `DefinesInternalPunctuation`,
/// `DefinesNonWordCharacter`, `DefinesEndingPunctuation`,
/// and `DefinesNonPrefixCharacters`.
pub trait DefaultCharacterDefinitions { }

impl<T> DefinesSentenceEndings for T where T: DefaultCharacterDefinitions
{
  #[inline(always)]
  fn sentence_endings() -> &'static Set<char> {
    &SENTENCE_ENDINGS
  }
}

impl<T> DefinesInternalPunctuation for T where T: DefaultCharacterDefinitions
{
  #[inline(always)]
  fn internal_punctuation() -> &'static Set<char> {
    &INTERNAL_PUNCT
  }
}

impl<T> DefinesNonWordCharacters for T where T: DefaultCharacterDefinitions
{
  #[inline(always)]
  fn nonword_chars() -> &'static Set<char> {
    &NONWORD_CHARS
  }
}

impl<T> DefinesPunctuation for T where T: DefaultCharacterDefinitions
{
  #[inline(always)]
  fn punctuation() -> &'static Set<char> {
    &PUNCTUATION
  }
}

impl<T> DefinesNonPrefixCharacters for T where T: DefaultCharacterDefinitions
{
  #[inline(always)]
  fn nonprefix_chars() -> &'static Set<char> {
    &NONPREFIX_CHARS
  }
}


/// Standard settings for all tokenizers, and trainers.
pub struct Standard;

impl DefaultCharacterDefinitions for Standard {}

impl TrainerParameters for Standard {
  #[inline(always)]
  fn abbrev_lower_bound() -> f64 {
    0.3
  }
  #[inline(always)]
  fn abbrev_upper_bound() -> f64 {
    5f64
  }
  #[inline(always)]
  fn ignore_abbrev_penalty() -> bool {
    false
  }
  #[inline(always)]
  fn collocation_lower_bound() -> f64 {
    7.88
  }
  #[inline(always)]
  fn sentence_starter_lower_bound() -> f64 {
    30f64
  }
  #[inline(always)]
  fn include_all_collocations() -> bool {
    false
  }
  #[inline(always)]
  fn include_abbrev_collocations() -> bool {
    false
  }
  #[inline(always)]
  fn collocation_frequency_lower_bound() -> f64 {
    1f64
  }
}


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
