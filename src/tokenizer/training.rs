use std::default::Default;

use tokenizer::token::PunktToken;

use phf::Set;

static DEFAULT: PunktWordTokenizer = PunktWordTokenizer;

/// Excluded characters from starting word tokens.
static EXCLUDED_WORD_PREFIX: Set<char> = phf_set! { '(', '"', '`', '{', '[', ':',
  ';', '&', '#', '*', '@', ')', '}', ']', '-', ',' };

/// Non-word Characters 
static EXCLUDED_WORD_CHARS: Set<char> = phf_set! { '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '[' };

/// Takes a document and splits it into tokens. A token is the same token as defined 
/// in NLTK. This tokenizer is heavily inspired by the one in NLTK, but tries to improve
/// upon the performance by avoiding the use of regular expressions.
///
/// A token does NOT contain the original slice of the document because a trainer must 
/// hold onto it (and can hold onto a token without the original document being in scope,
/// as it is copied). You can easily obtain the slice with the original document, and 
/// `start` and `len`.
///
/// # Example
///
/// ```rust
/// use std::default::Default;
/// use punkt::tokenizer::WordTokenizer;
/// use punkt::tokenizer::prelude::{Tokenizer, DocumentSlice};
///
/// let doc = "This is a (short) document.";
/// let tokenizer: WordTokenizer = Default::default();
///
/// let tokens = tokenizer.tokenize_document(doc);
/// let tokstrings: Vec<String> = tokens
///   .iter()
///   .map(|t| t.as_doc_slice(doc).to_string())
///   .collect();
///
/// assert_eq!(
///   vec!("This", "is", "a", "(", "short", ")", "document."),
///   tokstrings);
/// ```
#[derive(Copy)]
pub struct PunktWordTokenizer;

impl PunktWordTokenizer {
  /// Creates a new instance of a word Tokenizer.
  #[inline]
  pub fn new() -> PunktWordTokenizer { 
    PunktWordTokenizer 
  }
}

impl Default for PunktWordTokenizer {
  #[inline]
  fn default() -> PunktWordTokenizer { 
    PunktWordTokenizer::new() 
  }
}

impl Default for &'static PunktWordTokenizer {
  #[inline]
  fn default() -> &'static PunktWordTokenizer {
    &DEFAULT
  }
}

pub struct PunktTokenIterator<'a> {
  pos: uint,
  doc: &'a str 
}

/*
impl<'a> Tokenizer<'a, PunktToken> for PunktWordTokenizer {
  /// Tokenize words in a document. Iterates through the document and finds 
  /// tokens. Must do some look aheads to determine if a character is part of an elipses
  /// or dash. This should be faster than a regex for MOST inputs, 
  /// and it's almost guaranteed to be faster if the input is not some weird edge case.
  fn tokenize_document(&self, doc: &str) -> Vec<PunktToken> {
    // 5 is a general estimate at avg. word length.
    let mut tokens = Vec::with_capacity(doc.len() / 5); 
    let mut pos = 0u;
    let mut cap_start = 0u;
    let mut cap_active = false;
    let mut is_paragraph_start = false;
    let mut is_newline_start = true;
    let mut is_ellipsis = false;
    let end = doc.len();

    while pos < end { 
      let cur = doc.char_at(pos);

      // Macro to end a current capture. Uses some local vars, so shouldn't
      // be used outside the loop.
      macro_rules! end_capture(
        () => (
          { 
            let tok = doc.slice(cap_start, pos);

            tokens.push(
              PunktToken::new(
                tok,
                cap_start,
                is_ellipsis, 
                is_paragraph_start,
                is_newline_start));

            debug!(
              "terminated at `{}` ({}): {} -> {}", 
              String::from_utf8(
                cur.escape_default().map(|c| c as u8).collect::<Vec<u8>>()).unwrap(), 
              tok, 
              cap_start, 
              pos);
          }
        );
        ($cond:ident) => (
          if $cond {
            end_capture!()
          }
        );
      );

      macro_rules! is_multi_char(
        ($c:expr, $pos:expr) => (
          match $c {
            '-' | '.' => PunktWordTokenizer::is_multi_char(doc, $pos),
                    _ => None
          }
        )
      );

      // Compute if the current character is the start of a multi character.
      // Done here because it is used my multiple checks in the below pattern 
      // matching block.
      let multi_char = is_multi_char!(cur, pos); 

      match cur {
        // Hit a multi char. If a capture is active, then end it. 
        _ if multi_char.is_some() => {
          end_capture!(cap_active);

          cap_start = pos;

          let word = multi_char.unwrap();

          pos += word.len();

          // If the multi_char succeeded, then the word has to be greater than 
          // 0 length.
          is_ellipsis = word.ends_with(".");

          end_capture!();

          cap_active = false;
          
          // The position has to be advanced to the last character in the 
          // word to capture. This continue will skip over the increment of 
          // the position at the end, to prevent a character from being skipped.
          continue;
        }
        // Hit a word start character, and the buff is empty. Start 
        // capturing. Excludes the last character because if it is 
        // reached it will not be added to the vector.
        c if !cap_active && 
          !EXCLUDED_WORD_PREFIX.contains(&c) && 
          !c.is_whitespace() &&
          pos + c.len_utf8() != end => 
        {
          cap_start = pos;
          cap_active = true;
          is_ellipsis = false;
          is_newline_start = false;
          is_paragraph_start = false;
        }
        // Hit a possible ending to a string
        c if cap_active && c.is_whitespace() => { 
          end_capture!();

          cap_active = false;
        }
        // Hit a end of word marker that isn't a space. Roll back the 
        // cursor, and treat the EXCLUDED_WORD_CHAR as a potential token.
        c if cap_active && EXCLUDED_WORD_CHARS.contains(&c) => {
          end_capture!();

          cap_active = false;

          continue;
        }
        // Hit the end of the document. Implicit word end.
        c if cap_active && pos + c.len_utf8() == end => {
          pos += c.len_utf8();

          end_capture!();

          break;
        }
        // A single character non-whitespace token.
        c if !cap_active && !c.is_whitespace() => {
          cap_start = pos;

          pos += c.len_utf8();

          // Since cap is NOT set to active, a reset has not occured.
          // Manually reset here!
          is_newline_start = false;
          is_paragraph_start = false;
          is_ellipsis = false;

          end_capture!();

          cap_active = false;

          continue;
        }
        // Hit a comma, check if a word ending exists after. If one does, 
        // it is the end of a word. The comma is considered to be its 
        // own token, so rollback.
        ',' if cap_active && 
          (pos + 1 == end || 
           doc.char_at(pos + 1).is_whitespace() || 
           EXCLUDED_WORD_CHARS.contains(&doc.char_at(pos + 1)) || 
           is_multi_char!(doc.char_at(pos + 1), pos + 1).is_some()) =>
        {
          end_capture!();

          cap_active = false;

          continue;
        }
        _ => () // Skip
      }

      // Determine if a newline has been hit, and whether or not 
      // to treat it as a new paragraph.
      match cur {
        '\n' if !is_newline_start => is_newline_start = true,
        '\n'                      => is_paragraph_start = true,
           _                      => () // Skip
      }

      pos += cur.len_utf8(); 
    }

    tokens
  }
}
*/

impl PunktWordTokenizer {
  /// Checks if the a slice of the document starting at pos 
  /// is a multi char (ex. "...", ". . .", "--").
  /// These are all one-width chars, so iterating by 1 is OK.
  #[inline]
  fn is_multi_char(doc: &str, _pos: uint) -> Option<&str> {
    let start = _pos;
    let mut pos = _pos;
    let mut cur = doc.as_bytes()[pos];
    let mut is_multi = false;
    let end = doc.len();

    // This method should only be triggered on '.' or '-'.
    pos += 1;

    while pos < end {
      let c = doc.as_bytes()[pos];

      match c {
        b'-' if cur == b'-'                => is_multi = true, 
        b'.' if cur == b'.' || cur == b' ' => cur = c,
        b' ' if cur == b'.'                => cur = c,
           _ if cur == b' '                => { pos -= 1; break }
           _                               => break
      }

      pos += 1;
    }

    if is_multi || (pos - start > 1) {
      Some(doc.slice(start, pos))
    } else {
      None
    }
  }
}

#[cfg(test)]
mod punkt_word_tokenizer_tests {
  use super::PunktWordTokenizer;
  use tokenizer::prelude::{Tokenizer, DocumentSlice};

  fn read_expected(s: &str) -> Vec<String> {
    s.split('\n').filter(|s| s.len() > 0).map(|s| s.trim_matches('\r').to_string()).collect()
  }

  fn tokenize(t: PunktWordTokenizer, s: &str) -> Vec<String> {
    t.tokenize_document(s).iter().map(|x| x.typ().to_string()).collect()
  }

  #[test]
  fn smoke_test_tokenize() {
    let tzer = PunktWordTokenizer::new();
    let tests = vec![
      ("This is a sentence,Δ to test comma terminators.",
       vec!["This", "is", "a", "sentence,Δ", "to", "test", "comma", "terminators."]),
      ("Bob spent $5.56 on a Nestle Co. iced tea at C.V.S",
       vec!["Bob", "spent", "$5.56", "on", "a", "Nestle", "Co.", "iced", "tea", "at", "C.V.S"]),
      ("I bought some peaches (for 20% off at Martin's) to give to my grandmother.",
       vec!["I", "bought", "some", "peaches", "(", "for", "20%", "off", "at", "Martin", "'s", ")",
            "to", "give", "to", "my", "grandmother."])
    ];

    for &(doc, ref expect) in tests.iter() {
      let toks = tzer.tokenize_document(doc);
      let sliced: Vec<&str> = toks.iter().map(|t| t.as_doc_slice(doc)).collect();

      assert_eq!(&sliced, expect);
    }
  }

  #[test]
  fn smoke_test_is_multi_char_pass() {
    let docs = vec!(". . .", "..", "--", "---", ". . . . .", ".. .."); 

    for d in docs.iter() { 
      assert!(PunktWordTokenizer::is_multi_char(*d, 0).is_some(), "failed {}", *d); 
    }
  }

  #[test]
  fn smoke_test_is_multi_char_fail() {
    let docs = vec!("- -", ".", ".abc", "abc", "-abc", "-.", "-.-", " ", "-  ");

    for d in docs.iter() { 
      assert!(PunktWordTokenizer::is_multi_char(*d, 0).is_none(), "failed {}", *d); 
    }
  }

  #[test]
  fn test_tokenize_compare_nltk() {
    let tokenizer = PunktWordTokenizer::new();

    assert_eq!(
      tokenize(tokenizer, include_str!("../../test/npr-article-01-raw.txt")),
      read_expected(include_str!("../../test/npr-article-01-expected-words.txt")));
    assert_eq!(
      tokenize(tokenizer, include_str!("../../test/ny-times-article-01-raw.txt")),
      read_expected(include_str!("../../test/ny-times-article-01-expected-words.txt")));
    assert_eq!(
      tokenize(tokenizer, include_str!("../../test/sigma-wiki-raw.txt")),
      read_expected(include_str!("../../test/sigma-wiki-expected.txt")));
    assert_eq!(
      tokenize(tokenizer, include_str!("../../test/the-sayings-of-confucius-raw.txt")),
      read_expected(include_str!("../../test/the-sayings-of-confucius-expected.txt")));
    assert_eq!(
      tokenize(tokenizer, include_str!("../../test/pride-and-prejudice-raw.txt")),
      read_expected(include_str!("../../test/pride-and-prejudice-expected.txt")));
    assert_eq!(
      tokenize(tokenizer, include_str!("../../test/history-of-china-raw.txt")),
      read_expected(include_str!("../../test/history-of-china-expected.txt")));
  }
}

#[cfg(test)]
mod punkt_word_tokenizer_bench {
  use test::Bencher;
  use super::PunktWordTokenizer;
  use tokenizer::prelude::Tokenizer;

  #[bench]
  fn punkt_word_tokenizer_bench_short(b: &mut Bencher) {
    let tokenizer = PunktWordTokenizer::new();

    b.iter(|| {
      tokenizer.tokenize_document(include_str!("../../test/sigma-wiki-raw.txt"));
    });
  }

  #[bench]
  fn punkt_word_tokenizer_bench_long(b: &mut Bencher) {
    let tokenizer = PunktWordTokenizer::new();

    b.iter(|| {
      tokenizer.tokenize_document(
        include_str!("../../test/the-sayings-of-confucius-raw.txt"));
    });
  }

  #[bench]
  fn punkt_word_tokenizer_bench_very_long(b: &mut Bencher) {
    let tokenizer = PunktWordTokenizer::new();

    b.iter(|| {
      tokenizer.tokenize_document(include_str!("../../test/pride-and-prejudice-raw.txt"));
    });
  }
}
