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


static internal_punct  : Set<char> = phf_set![',', ':', ';', '\u{2014}'];
static sentence_endings: Set<char> = phf_set!['.', '?', '!'];
static punctuation     : Set<char> = phf_set![';', ':', ',', '.', '!', '?'];
static nonword_chars   : Set<char> = phf_set![
  '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '['
];
static nonprefix_chars : Set<char> = phf_set![
  '(', '"', '`', '{', '[', ':', ';', '&', '#', '*', '@', ')', '}', ']', '-', ','
];


/// Mixin that will give the default implementations for 
/// `DefinesSentenceEndings`, `DefinesInternalPunctuation`, 
/// `DefinesNonWordCharacter`, `DefinesEndingPunctuation`,
/// and `DefinesNonPrefixCharacters`.
pub trait DefaultCharacterDefinitions { }

impl<T> DefinesSentenceEndings for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn sentence_endings() -> &'static Set<char> { 
    &sentence_endings 
  }
}

impl<T> DefinesInternalPunctuation for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn internal_punctuation() -> &'static Set<char> {
    &internal_punct
  }
}

impl<T> DefinesNonWordCharacters for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn nonword_chars() -> &'static Set<char> {
    &nonword_chars
  }
}

impl<T> DefinesPunctuation for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn punctuation() -> &'static Set<char> {
    &punctuation
  }
}

impl<T> DefinesNonPrefixCharacters for T where T : DefaultCharacterDefinitions {
  #[inline(always)] fn nonprefix_chars() -> &'static Set<char> {
    &nonprefix_chars
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