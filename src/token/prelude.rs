#[cfg(test)] use token::sword::SentenceWordToken;
#[cfg(test)] use token::training::TrainingToken;

// Flags that can be set. These describe certain properties about the Token.
// These 6 flags only use the lower 8 bits.
const HAS_FINAL_PERIOD  : u16 = 0b0000000000000001;
const IS_ELLIPSIS       : u16 = 0b0000000000000010;
const IS_ABBREV         : u16 = 0b0000000000000100;
const IS_SENTENCE_BREAK : u16 = 0b0000000000001000;
const IS_PARAGRAPH_START: u16 = 0b0000000000010000;
const IS_NEWLINE_START  : u16 = 0b0000000000100000;
const IS_UPPERCASE      : u16 = 0b0000000001000000;
const IS_LOWERCASE      : u16 = 0b0000000010000000;

// These flags only use the upper 8 bits.
const IS_INITIAL        : u16 = 0b1000000000000000;
const IS_NUMERIC        : u16 = 0b0100000000000000;
const IS_NON_PUNCT      : u16 = 0b0010000000000000;
const IS_ALPHABETIC     : u16 = 0b0000010000000000;

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

/// A word token with a representation of the token WITH a period appended
/// onto the end.
pub trait WordTokenWithPeriod {
  fn token_with_period(&self) -> &str;
}

/// A representable word token. 
pub trait WordToken {
  fn token(&self) -> &str;

  /// The length of the token.
  #[inline]
  fn len(&self) -> usize {
    self.token().len()
  }
}

impl<F, T> WordToken for T
  where T: WordTokenWithPeriod + 
           WordTokenWithFlags<Flags = F> + 
           WordTokenWithFlagsOps<F>
{
  /// Returns the original token (which can be reconstructed given the 
  /// flags on the token are correct).
  #[inline]
  fn token(&self) -> &str {
    if self.has_final_period() {
      self.token_with_period()
    } else {
      &self.token_with_period()[..self.token_with_period().len() - 1]
    }
  }
}

/// A word token with a representation of the token WITHOUT a period (if 
/// the original word token contained one).
pub trait WordTokenWithoutPeriod {
  fn token_without_period(&self) -> &str;
}

impl<F, T> WordTokenWithoutPeriod for T 
  where T: WordToken + 
           WordTokenWithFlags<Flags = F> + 
           WordTokenWithFlagsOps<F>
{
  /// Returns the token without a period on the end (if it had one), given 
  /// that the flags on the token are correct.
  #[inline]
  fn token_without_period(&self) -> &str {
    if self.has_final_period() {
      &self.token()[..self.len() - 1]
    } else {
      self.token()
    }
  }
}

/// A word token that has a type associated with it. The type can be 
/// represented with or without a period or without a sentence break.
pub trait WordTypeToken: WordToken {
  fn typ(&self) -> &str;
  fn typ_with_period(&self) -> &str;
  fn typ_without_period(&self) -> &str;
  fn typ_without_break_or_period(&self) -> &str;
}

impl<F, T> WordTypeToken for T
  where T: WordTypeToken + 
           WordTokenWithPeriod + 
           WordTokenWithFlags<Flags = F> + 
           WordTokenWithFlagsOpsExt<F> + 
           WordTokenWithFlagsOps<F> 
{
  /// Returns the type of the token. If the token is numeric (determined by flags), 
  /// returns a string defined by NLTK: `##number##`, otherwise returns the 
  /// token.
  #[inline]
  fn typ(&self) -> &str {
    if self.is_numeric() { "##number##" } else { self.token() }
  }

  /// Returns the type of the token with a period appended to it. Returns 
  /// `##number##.` if the token is numeric (determined by flags), otherwise 
  /// returns the original token with a period appended to it.
  #[inline]
  fn typ_with_period(&self) -> &str {
    if self.is_numeric() { "##number##." } else { self.token_with_period() }
  }

  /// Returns the type of the token without a period appended to it. Will return 
  /// `.`, if it is the only character in the string; otherwise, will slice type 
  /// to exclude the final period. 
  #[inline]
  fn typ_without_period(&self) -> &str {
    if self.token().len() > 1 && self.has_final_period() {
      &self.typ_with_period()[..self.typ_with_period().len() - 1]
    } else {
      self.typ()
    }
  }

  #[inline]
  fn typ_without_break_or_period(&self) -> &str {
    if self.is_sentence_break() {
      self.typ_without_period()
    } else {
      self.typ()
    }
  }
}

/// A token with flags that describe physical properties of the WordToken. 
/// These should be set either during construction, or during a processing 
/// stage.
pub trait WordTokenWithFlags: WordToken {
  type Flags;

  fn flags(&self) -> &<Self as WordTokenWithFlags>::Flags;
  fn flags_mut(&mut self) -> &mut <Self as WordTokenWithFlags>::Flags;
}

/// Operations performable on a WordToken with flags. 
pub trait WordTokenWithFlagsOps<T>: WordTokenWithFlags<Flags = T> {
  fn is_ellipsis(&self) -> bool;
  fn is_abbrev(&self) -> bool;
  fn is_sentence_break(&self) -> bool;
  fn has_final_period(&self) -> bool;
  fn is_paragraph_start(&self) -> bool;
  fn is_newline_start(&self) -> bool;
  fn is_uppercase(&self) -> bool;
  fn is_lowercase(&self) -> bool;

  fn first_case(&self) -> LetterCase {
    if self.is_lowercase() {
      LetterCase::Lower
    } else if self.is_uppercase() {
      LetterCase::Upper
    } else {
      LetterCase::Unknown
    }
  }

  fn set_is_ellipsis(&mut self, b: bool);
  fn set_is_abbrev(&mut self, b: bool);
  fn set_is_sentence_break(&mut self, b: bool);
  fn set_has_final_period(&mut self, b: bool);
  fn set_is_paragraph_start(&mut self, b: bool);
  fn set_is_newline_start(&mut self, b: bool);
  fn set_is_uppercase(&mut self, b: bool);
  fn set_is_lowercase(&mut self, b: bool);
}

/// Default implementation for a WordToken with flags, where the flags are 
/// a single byte.
impl<T> WordTokenWithFlagsOps<u8> for T 
  where T: WordTokenWithFlags<Flags = u8>
{
  #[inline]
  fn is_uppercase(&self) -> bool {
    *self.flags() & IS_UPPERCASE as u8 != 0
  }

  #[inline]
  fn is_lowercase(&self) -> bool {
    *self.flags() & IS_LOWERCASE as u8 != 0
  }

  #[inline]
  fn is_ellipsis(&self) -> bool {
    *self.flags() & IS_ELLIPSIS as u8 != 0
  }

  #[inline]
  fn is_abbrev(&self) -> bool {
    *self.flags() & IS_ABBREV as u8 != 0
  }

  #[inline]
  fn is_sentence_break(&self) -> bool {
    *self.flags() & IS_SENTENCE_BREAK as u8 != 0
  }

  #[inline]
  fn has_final_period(&self) -> bool {
    *self.flags() & HAS_FINAL_PERIOD as u8 != 0
  }

  #[inline]
  fn is_paragraph_start(&self) -> bool {
    *self.flags() & IS_PARAGRAPH_START as u8 != 0
  }

  #[inline]
  fn is_newline_start(&self) -> bool {
    *self.flags() & IS_NEWLINE_START as u8 != 0
  }

  #[inline]
  fn set_is_ellipsis(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_ELLIPSIS as u8;
    } else if self.is_ellipsis() {
      *self.flags_mut() ^= IS_ELLIPSIS as u8;
    }
  }

  #[inline]
  fn set_is_abbrev(&mut self, b: bool) { 
    if b {
      *self.flags_mut() |= IS_ABBREV as u8;
    } else if self.is_abbrev() {
      *self.flags_mut() ^= IS_ABBREV as u8;
    }
  }

  #[inline]
  fn set_is_sentence_break(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_SENTENCE_BREAK as u8;
    } else if self.is_sentence_break() {
      *self.flags_mut() ^= IS_SENTENCE_BREAK as u8;
    }
  }

  #[inline]
  fn set_has_final_period(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= HAS_FINAL_PERIOD as u8;
    } else if self.has_final_period() {
      *self.flags_mut() ^= HAS_FINAL_PERIOD as u8;
    }
  }

  #[inline]
  fn set_is_paragraph_start(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_PARAGRAPH_START as u8;
    } else if self.is_paragraph_start() {
      *self.flags_mut() ^= IS_PARAGRAPH_START as u8;
    }
  }

  #[inline]
  fn set_is_newline_start(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_NEWLINE_START as u8;
    } else if self.is_newline_start() {
      *self.flags_mut() ^= IS_NEWLINE_START as u8;
    }
  }

  #[inline]
  fn set_is_uppercase(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_UPPERCASE as u8;
    } else if self.is_uppercase() {
      *self.flags_mut() ^= IS_UPPERCASE as u8;
    }
  }

  #[inline]
  fn set_is_lowercase(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_LOWERCASE as u8;
    } else if self.is_lowercase() {
      *self.flags_mut() ^= IS_LOWERCASE as u8;
    }
  }
}

/// Default implementation for a WordToken with flags, where the flags are u16.
impl<T> WordTokenWithFlagsOps<u16> for T 
  where T: WordTokenWithFlags<Flags = u16>
{
  #[inline]
  fn is_uppercase(&self) -> bool {
    *self.flags() & IS_UPPERCASE != 0
  }

  #[inline]
  fn is_lowercase(&self) -> bool {
    *self.flags() & IS_LOWERCASE != 0
  }

  #[inline]
  fn is_ellipsis(&self) -> bool {
    *self.flags() & IS_ELLIPSIS != 0
  }

  #[inline]
  fn is_abbrev(&self) -> bool {
    *self.flags() & IS_ABBREV != 0
  }

  #[inline]
  fn is_sentence_break(&self) -> bool {
    *self.flags() & IS_SENTENCE_BREAK != 0
  }

  #[inline]
  fn has_final_period(&self) -> bool {
    *self.flags() & HAS_FINAL_PERIOD != 0
  }

  #[inline]
  fn is_paragraph_start(&self) -> bool {
    *self.flags() & IS_PARAGRAPH_START != 0
  }

  #[inline]
  fn is_newline_start(&self) -> bool {
    *self.flags() & IS_NEWLINE_START != 0
  }

  #[inline]
  fn set_is_ellipsis(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_ELLIPSIS;
    } else if self.is_ellipsis() {
      *self.flags_mut() ^= IS_ELLIPSIS;
    }
  }

  #[inline]
  fn set_is_abbrev(&mut self, b: bool) { 
    if b {
      *self.flags_mut() |= IS_ABBREV;
    } else if self.is_abbrev() {
      *self.flags_mut() ^= IS_ABBREV;
    }
  }

  #[inline]
  fn set_is_sentence_break(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_SENTENCE_BREAK;
    } else if self.is_sentence_break() {
      *self.flags_mut() ^= IS_SENTENCE_BREAK;
    }
  }

  #[inline]
  fn set_has_final_period(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= HAS_FINAL_PERIOD;
    } else if self.has_final_period() {
      *self.flags_mut() ^= HAS_FINAL_PERIOD;
    }
  }

  #[inline]
  fn set_is_paragraph_start(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_PARAGRAPH_START;
    } else if self.is_paragraph_start() {
      *self.flags_mut() ^= IS_PARAGRAPH_START;
    }
  }

  #[inline]
  fn set_is_newline_start(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_NEWLINE_START;
    } else if self.is_newline_start() {
      *self.flags_mut() ^= IS_NEWLINE_START;
    }
  }

  #[inline]
  fn set_is_uppercase(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_UPPERCASE;
    } else if self.is_uppercase() {
      *self.flags_mut() ^= IS_UPPERCASE;
    }
  }

  #[inline]
  fn set_is_lowercase(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_LOWERCASE;
    } else if self.is_lowercase() {
      *self.flags_mut() ^= IS_LOWERCASE;
    }
  }
}

/// Extended operations on a WordToken with flags. The flags must be expanded,
/// (not u8), to account for the different flag parameters (u16 has a default 
/// implementation).
pub trait WordTokenWithFlagsOpsExt<T>: WordTokenWithFlags<Flags = T> {
  fn is_numeric(&self) -> bool;
  fn is_initial(&self) -> bool;
  fn is_non_punct(&self) -> bool;
  fn is_alphabetic(&self) -> bool;

  fn set_is_numeric(&mut self, b: bool);
  fn set_is_initial(&mut self, b: bool);
  fn set_is_non_punct(&mut self, b: bool);
  fn set_is_alphabetic(&mut self, b: bool);
}

/// Default implementation for a WordToken with flags, where the flags are u16.
impl<T> WordTokenWithFlagsOpsExt<u16> for T 
  where T: WordTokenWithFlags<Flags = u16>
{
  #[inline]
  fn is_numeric(&self) -> bool {
    *self.flags() & IS_NUMERIC != 0
  }

  #[inline]
  fn is_initial(&self) -> bool {
    *self.flags() & IS_INITIAL != 0
  }

  #[inline]
  fn is_non_punct(&self) -> bool {
    *self.flags() & IS_NON_PUNCT != 0
  }

  #[inline]
  fn is_alphabetic(&self) -> bool {
    *self.flags() & IS_ALPHABETIC != 0
  }

  #[inline]
  fn set_is_numeric(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_NUMERIC;
    } else if self.is_numeric() {
      *self.flags_mut() ^= IS_NUMERIC;
    }
  }

  #[inline]
  fn set_is_initial(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_INITIAL;
    } else if self.is_initial() {
      *self.flags_mut() ^= IS_INITIAL;
    }
  }

  #[inline]
  fn set_is_non_punct(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_NON_PUNCT;
    } else if self.is_non_punct() {
      *self.flags_mut() ^= IS_NON_PUNCT;
    }
  }

  #[inline]
  fn set_is_alphabetic(&mut self, b: bool) {
    if b {
      *self.flags_mut() |= IS_ALPHABETIC;
    } else if self.is_alphabetic() {
      *self.flags_mut() ^= IS_ALPHABETIC;
    }
  }
}

macro_rules! perform_flag_test(
  ($tok:expr, $f:ident, $t:ident) => (
    {
      $tok.$f(true);
      assert!($tok.$t());
      $tok.$f(false);
      assert!(!$tok.$t());
    }
  )
);

#[test]
fn test_training_token_flags() {
  let mut tok = TrainingToken::new("test", false, false, false);

  tok.set_is_non_punct(false);
  tok.set_is_lowercase(false);
  tok.set_is_alphabetic(false);
  
  assert_eq!(*tok.flags(), 0);

  perform_flag_test!(tok, set_is_ellipsis, is_ellipsis);
  perform_flag_test!(tok, set_is_abbrev, is_abbrev);
  perform_flag_test!(tok, set_has_final_period, has_final_period);
  perform_flag_test!(tok, set_is_paragraph_start, is_paragraph_start);
  perform_flag_test!(tok, set_is_newline_start, is_newline_start);
  perform_flag_test!(tok, set_is_uppercase, is_uppercase);
  perform_flag_test!(tok, set_is_lowercase, is_lowercase);
  perform_flag_test!(tok, set_is_numeric, is_numeric);
  perform_flag_test!(tok, set_is_initial, is_initial);
  perform_flag_test!(tok, set_is_non_punct, is_non_punct);
  perform_flag_test!(tok, set_is_alphabetic, is_alphabetic);
}

#[test]
fn test_sentence_word_token_flags() {
  let mut tok = SentenceWordToken::new("test");
  
  tok.set_is_lowercase(false);

  assert_eq!(*tok.flags(), 0);

  perform_flag_test!(tok, set_is_ellipsis, is_ellipsis);
  perform_flag_test!(tok, set_is_abbrev, is_abbrev);
  perform_flag_test!(tok, set_has_final_period, has_final_period);
  perform_flag_test!(tok, set_is_paragraph_start, is_paragraph_start);
  perform_flag_test!(tok, set_is_newline_start, is_newline_start);
  perform_flag_test!(tok, set_is_uppercase, is_uppercase);
  perform_flag_test!(tok, set_is_lowercase, is_lowercase);
}
