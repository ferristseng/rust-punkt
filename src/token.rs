use std::ops::Deref;
use std::hash::{Hash, Hasher};

use prelude::CaseInsensitiveEq;


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


#[derive(Clone)]
pub struct Token<'a> {
  inner: &'a str,
  flags: u16 
}

impl<'a> Token<'a> {
  pub fn new(slice: &'a str, is_el: bool, is_pg: bool, is_nl: bool) -> Token {
    debug_assert!(slice.len() > 0);

    let first = slice.char_at(0);
    let mut has_punct = false;
    let mut tok = Token {
      inner: slice,
      flags: 0x00
    };

    if slice.as_bytes()[slice.len() - 1] == b'.' { 
      tok.set_has_final_period(true); 
    }

    // These are mutually exclusive. Only need to set one, although 
    // still potentially need to run both checks.
    if is_str_numeric(slice) {
      tok.set_is_numeric(true);
    } else if is_str_initial(slice) {
      tok.set_is_initial(true);
    }

    for c in slice.chars() { 
      if c.is_alphabetic() || c == '_' {
        tok.set_is_non_punct(true);
      } else if !c.is_digit(10) {
        has_punct = true;
      }
    }
    
    if first.is_uppercase() {
      tok.set_is_uppercase(true);
    } else if first.is_lowercase() {
      tok.set_is_lowercase(true);
    }

    tok.set_is_alphabetic(!has_punct);
    tok.set_is_ellipsis(is_el);
    tok.set_is_paragraph_start(is_pg);
    tok.set_is_newline_start(is_nl);

    tok
  }

  #[inline] pub fn tok(&self) -> &str {
    if self.is_numeric() {
      "##number##"
    } else {
      self.inner
    }
  }

  #[inline(always)] pub fn is_uppercase(&self) -> bool { 
    self.flags & IS_UPPERCASE != 0 
  }

  #[inline(always)] pub fn is_lowercase(&self) -> bool {
    self.flags & IS_LOWERCASE != 0
  }

  #[inline(always)] pub fn is_ellipsis(&self) -> bool {
    self.flags & IS_ELLIPSIS != 0
  }

  #[inline(always)] pub fn is_abbrev(&self) -> bool {
    self.flags & IS_ABBREV != 0
  }

  #[inline(always)] pub fn is_sentence_break(&self) -> bool {
    self.flags & IS_SENTENCE_BREAK != 0
  }

  #[inline(always)] pub fn has_final_period(&self) -> bool {
    self.flags & HAS_FINAL_PERIOD != 0
  }

  #[inline(always)] pub fn is_paragraph_start(&self) -> bool {
    self.flags & IS_PARAGRAPH_START != 0
  }

  #[inline(always)] pub fn is_newline_start(&self) -> bool {
    self.flags & IS_NEWLINE_START != 0
  }

  #[inline(always)] pub fn is_numeric(&self) -> bool {
    self.flags & IS_NUMERIC != 0
  }

  #[inline(always)] pub fn is_initial(&self) -> bool {
    self.flags & IS_INITIAL != 0
  }

  // The NLTK docs note that all numeric tokens are considered be not only
  // punctuation, because they are converted to `##number##`, which clearly
  // has alphabetic characters.
  #[inline(always)] pub fn is_non_punct(&self) -> bool {
    (self.flags & IS_NON_PUNCT != 0) || self.is_numeric()
  }

  #[inline(always)] pub fn is_alphabetic(&self) -> bool {
    self.flags & IS_ALPHABETIC != 0
  }

  #[inline(always)] pub fn set_is_ellipsis(&mut self, b: bool) {
    if b {
      self.flags |= IS_ELLIPSIS;
    } else if self.is_ellipsis() {
      self.flags ^= IS_ELLIPSIS;
    }
  }

  #[inline(always)] pub fn set_is_abbrev(&mut self, b: bool) { 
    if b {
      self.flags |= IS_ABBREV;
    } else if self.is_abbrev() {
      self.flags ^= IS_ABBREV;
    }
  }

  #[inline(always)] pub fn set_is_sentence_break(&mut self, b: bool) {
    if b {
      self.flags |= IS_SENTENCE_BREAK;
    } else if self.is_sentence_break() {
      self.flags ^= IS_SENTENCE_BREAK;
    }
  }

  #[inline(always)] pub fn set_has_final_period(&mut self, b: bool) {
    if b {
      self.flags |= HAS_FINAL_PERIOD;
    } else if self.has_final_period() {
      self.flags ^= HAS_FINAL_PERIOD;
    }
  }

  #[inline(always)] pub fn set_is_paragraph_start(&mut self, b: bool) {
    if b {
      self.flags |= IS_PARAGRAPH_START;
    } else if self.is_paragraph_start() {
      self.flags ^= IS_PARAGRAPH_START;
    }
  }

  #[inline(always)] pub fn set_is_newline_start(&mut self, b: bool) {
    if b {
      self.flags |= IS_NEWLINE_START;
    } else if self.is_newline_start() {
      self.flags ^= IS_NEWLINE_START;
    }
  }

  #[inline(always)] pub fn set_is_uppercase(&mut self, b: bool) {
    if b {
      self.flags |= IS_UPPERCASE;
    } else if self.is_uppercase() {
      self.flags ^= IS_UPPERCASE;
    }
  }

  #[inline(always)] pub fn set_is_lowercase(&mut self, b: bool) {
    if b {
      self.flags |= IS_LOWERCASE;
    } else if self.is_lowercase() {
      self.flags ^= IS_LOWERCASE;
    }
  }

  #[inline(always)] pub fn set_is_numeric(&mut self, b: bool) {
    if b {
      self.flags |= IS_NUMERIC;
    } else if self.is_numeric() {
      self.flags ^= IS_NUMERIC;
    }
  }

  #[inline(always)] pub fn set_is_initial(&mut self, b: bool) {
    if b {
      self.flags |= IS_INITIAL;
    } else if self.is_initial() {
      self.flags ^= IS_INITIAL;
    }
  }

  #[inline(always)] pub fn set_is_non_punct(&mut self, b: bool) {
    if b {
      self.flags |= IS_NON_PUNCT;
    } else if self.is_non_punct() {
      self.flags ^= IS_NON_PUNCT;
    }
  }

  #[inline(always)] pub fn set_is_alphabetic(&mut self, b: bool) {
    if b {
      self.flags |= IS_ALPHABETIC;
    } else if self.is_alphabetic() {
      self.flags ^= IS_ALPHABETIC;
    }
  }
}

impl<'a> Deref for Token<'a> {
  type Target = str;

  #[inline(always)] fn deref(&self) -> &str { self.inner }
}

impl<'a> PartialEq for Token<'a> {
  #[inline(always)] fn eq(&self, other: &Token) -> bool {
    self.case_insensitive_eq(other)
  }
}

impl<'a> Hash for Token<'a> {
  fn hash<H>(&self, state: &mut H) where H : Hasher {
    if self.is_numeric() {
      "##number##".hash(state);
    } else {
      if self.has_final_period() {
        for c in self[..self.len() - 1].chars().flat_map(|c| c.to_lowercase()) {
          state.write_u32(c as u32);
        }
      } else {
        for c in self.chars().flat_map(|c| c.to_lowercase()) {
          state.write_u32(c as u32);
        }
      };
    }
  }
}


/// A number can start with a negative sign ('-'), and be followed by digits
/// or isolated periods, commas, or dashes. 
/// Note: It's assumed that multi-chars are taken out of the input when creating word 
/// tokens, so a numeric word token SHOULD not have a multi-char within it as
/// its received. This assumption should be fulfilled by the parser generating 
/// these word tokens. If it isn't some weird outputs are possible (such as "5.4--5").
#[inline] fn is_str_numeric(tok: &str) -> bool {
  let mut digit_found = false;
  let mut pos = 0;

  for c in tok.chars() {
    match c {
      // A digit was found. Note this to confirm punctuation
      // within the number is valid or not.
      _ if c.is_digit(10) => {
        digit_found = true
      }
      // A delimeter was found. This is valid as long as 
      // a digit was also found prior.
      ',' | '.' | '-' if digit_found => (),
      // A comma or period was found as the first character, or 
      // after a negative sign. This is a valid token.
      ',' | '.' if pos == 0 || pos == 1 => (),
      // A negative sign is found. 
      '-' if pos == 0 => (),
      // A non numeric token was encountered in the string that 
      // isn't a valid one. Return false.
      _ => return false
    }

    pos += c.len_utf8(); 
  }

  digit_found
}


/// Tests if the token is an initial. An initial is a 2 character grouping
/// where the first character is a letter (non-digit, non-symbol), and the 
/// next is a period.
#[inline] fn is_str_initial(tok: &str) -> bool {
  let mut iter = tok.chars();

  match (iter.next(), iter.next()) {
    (Some(c), Some('.')) if c.is_alphabetic() => iter.next().is_none(),
    _ => false
  }
}


#[test] fn test_token_flags() {
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

  let mut tok = Token::new("test", false, false, false);

  tok.set_is_non_punct(false);
  tok.set_is_lowercase(false);
  tok.set_is_alphabetic(false);
  
  assert_eq!(tok.flags, 0);

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


#[test] fn test_token_hash() {
  macro_rules! perform_hash_test_eq(
    ($tok:expr, $cmp:expr) => (
        {
          let _t0 = Token::new($tok, false, false, false);
          let _t1 = Token::new($cmp, false, false, false);
          let mut _h_t0 = ::std::hash::SipHasher::new();
          let mut _h_t1 = ::std::hash::SipHasher::new();

          _t1.hash(&mut _h_t0);
          _t0.hash(&mut _h_t1);

          assert_eq!(_h_t0.finish(), _h_t1.finish());
        }
    )
  );

  perform_hash_test_eq!("ABC", "abc.");
  perform_hash_test_eq!("1.35", "-100");
  perform_hash_test_eq!("ÀÁÂÃÄ", "àáâãä");
}