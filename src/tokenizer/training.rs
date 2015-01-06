#[cfg(test)] use test::Bencher;
#[cfg(test)] use std::io::fs;
#[cfg(test)] use std::io::fs::PathExtensions;
#[cfg(test)] use token::prelude::WordTypeToken;

use std::default::Default;

use token::TrainingToken;
use token::prelude::WordToken;

use phf::Set;

static DEFAULT: TrainingWordTokenizerParameters = TrainingWordTokenizerParameters {
  non_pref: &phf_set![
    '(', '"', '`', '{', '[', ':', ';', '&', '#', '*', '@', ')', '}', ']', '-', ','
  ],
  non_word: &phf_set![
    '?', '!', ')', '"', ';', '}', ']', '*', ':', '@', '\'', '(', '{', '['
  ]
};

#[derive(Copy)]
pub struct TrainingWordTokenizerParameters {
  non_pref: &'static Set<char>,
  non_word: &'static Set<char>
}

impl Default for &'static TrainingWordTokenizerParameters {
  fn default() -> &'static TrainingWordTokenizerParameters {
    &DEFAULT
  }
}

pub struct TrainingWordTokenizer<'a> {
  pos: uint,
  doc: &'a str,
  params: &'a TrainingWordTokenizerParameters
}

impl<'a> TrainingWordTokenizer<'a> {
  pub fn new(doc: &'a str) -> TrainingWordTokenizer<'a> {
    TrainingWordTokenizer {
      pos: 0,
      doc: doc,
      params: Default::default()
    }
  }

  pub fn with_parameters(
    doc: &'a str, 
    params: &'a TrainingWordTokenizerParameters
  ) -> TrainingWordTokenizer<'a> {
    TrainingWordTokenizer {
      pos: 0,
      doc: doc,
      params: params
    }
  }
}

const NEWLINE_START: u8 = 0b00000001;
const PARAGPH_START: u8 = 0b00000010;
const CAPTURE_START: u8 = 0b00000100;
const CAPTURE_COMMA: u8 = 0b00001000;

impl<'a> Iterator for TrainingWordTokenizer<'a> {
  type Item = TrainingToken;

  fn next(&mut self) -> Option<TrainingToken> {
    let mut state = if self.pos == 0 {
      NEWLINE_START
    } else {
      0u8
    };
    let mut start = self.pos;
    let mut is_ellipsis = false;

    while self.pos < self.doc.len() {
      let cur = self.doc.char_at(self.pos);

      // Slices the document, and returns the current token.
      macro_rules! return_token(
        () => (
          {
            // Rollback if the reason the capture was ended was because 
            // of a comma.
            if state & CAPTURE_COMMA != 0 {
              self.pos -= 1;
            }

            return Some(TrainingToken::new(
              self.doc.slice(start, self.pos),
              is_ellipsis, 
              state & PARAGPH_START != 0,
              state & NEWLINE_START != 0));
          }
        )
      );

      // Periods or dashes are the start of multi-chars. A multi-char
      // is defined as an ellipsis or hyphen (multiple-dashes). If there 
      // is a multi-character starting from the current character, return.
      // Otherwise, continue.
      match cur {
        // Both potential multi-char starts. Check for a multi-char. If 
        // one exists return it, and modify `self.pos`. Otherwise, continue.
        // If a capture has begin, or a comma was encountered, return the token
        // before this multi-char. 
        '.' | '-' => {
          match is_multi_char(self.doc, self.pos) {
            Some(s) => {
              if state & CAPTURE_START != 0 || 
                 state & CAPTURE_COMMA != 0 
              {
                return_token!()
              }
              start = self.pos;
              is_ellipsis = s.ends_with(".");
              
              self.pos += s.len();

              return_token!()
            },
            None => ()
          }
        }
        // Not a potential multi-char start, continue...
        _ => ()
      }

      match cur {
        // A capture has already started (meaning a valid character was encountered).
        // This block handles the cases with characters during a capture.
        c if state & CAPTURE_START != 0 => {
          match c {
            // Valid tokens. This isn't the only definition, but matching here
            // allows us to skip the subsequent checks, which can be costly. 
            // If a comma was encountered, reset `CAPTURE_COMMA`, as the comma 
            // does not signify the ending of the token.
            _ if c.is_alphanumeric() => {
              if state & CAPTURE_COMMA != 0 {
                state ^= CAPTURE_COMMA;
              }
            }
            // Found some whitespace, a non-word, or at the end of the document.
            // Return the token.
            _ if c.is_whitespace() ||
                 self.params.non_word.contains(&c) || 
                 self.pos + c.len_utf8() >= self.doc.len() => 
            {
              return_token!()
            }
            // A comma was found. Set the flag noting that a comma was found. 
            // Do NOT capture past the comma. Simply skip. 
            ',' => {
              state |= CAPTURE_COMMA;
            }
            // A valid token was encountered. Reset `CAPTURE_COMMA` to false, 
            // as the comma does not signal the end of the token.
            _ => {
              if state & CAPTURE_COMMA != 0 {
                state ^= CAPTURE_COMMA;
              }
            }
          }
        }
        // A valid prefix was found, and capturing has not yet begun. 
        // Capturing can begin!
        c if state & CAPTURE_START == 0 &&
             !c.is_whitespace() &&
             !self.params.non_pref.contains(&c) => 
        {
          start = self.pos;
          state |= CAPTURE_START
        }
        // A non-whitespace was encountered. End with just the character.
        c if !c.is_whitespace() => {
          start = self.pos;
          self.pos += c.len_utf8();
          return_token!()
        }
        // A newline was encountered, and no newline was found before. This
        // signifies a newline, but not a new paragraph.
        '\n' if state & NEWLINE_START == 0 => {
          state |= NEWLINE_START
        }
        // A newline was encountered, and the above pattern was not matched. This
        // signifies a string of newlines (a new paragraph).
        '\n' => {
          state |= PARAGPH_START
        }
        _ => ()
      }

      self.pos += cur.len_utf8();
    }

    None
  }

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    (self.doc.len() / 5, Some(self.doc.len() / 3))
  }
}

/// Checks if the a slice of the document starting at pos 
/// is a multi char (ex. "...", ". . .", "--").
/// These are all one-width chars, so iterating by 1 is OK.
fn is_multi_char(doc: &str, start: uint) -> Option<&str> {
  let mut end = start;
  let mut prv = doc.as_bytes()[end];

  // This method should only be triggered on '.' or '-'.
  end += 1;

  while end < doc.len() {
    let c = doc.as_bytes()[end];

    match c {
      // Hit a dash, and our previous was a dash -- 
      // continue matching dashes.
      b'-' if prv == b'-' => (),
      // Hit a period, and our previous was a period or
      // space. This is valid, skip. 
      b'.' if prv == b'.' || prv == b' ' => (),
      // Hit a space, and our previous was a period. 
      // Could be a ellipsis -- continue.
      b' ' if prv == b'.' => (),
      // Hit a non-multi-char character. If the previous 
      // was a space, truncate it. Then break, and check 
      // if our word was long enough.
      _ => { 
        if prv == b' ' {
          end -= 1;
        }

        break 
      }
    }

    prv = c;
    end += 1;
  }

  if end - start > 1 {
    Some(doc.slice(start, end))
  } else {
    None
  }
}

#[test]
fn smoke_test_is_multi_char_pass() {
  let docs = vec![". . .", "..", "--", "---", ". . . . .", ".. .."];
  
  for d in docs.iter() {
    assert!(is_multi_char(*d, 0).is_some(), "failed {}", *d);
  }
}

#[test]
fn word_tokenizer_compare_nltk() {
  for path in fs::walk_dir(&Path::new("test/word-training/")).unwrap() {
    if path.is_file() {
      let rawp = Path::new("test/raw/").join(path.filename_str().unwrap());
      let expf = fs::File::open(&path).read_to_string().unwrap();
      let rawf = fs::File::open(&rawp).read_to_string().unwrap();
      let exps = expf.split('\n');
      let tokr = TrainingWordTokenizer::new(rawf.as_slice());

      for (t, e) in tokr.zip(exps) {
        assert!(
          t.typ() == e.trim(), 
          "{} - you: [{}] != exp: [{}]", 
          path.filename_str().unwrap(),
          t.typ(),
          e.trim());
      }
    }
  }
}

#[bench]
fn word_tokenizer_bench_short(b: &mut Bencher) {
  b.iter(|| {
    let _: Vec<TrainingToken> = TrainingWordTokenizer::new(
      include_str!("../../test/raw/sigma-wiki.txt")).collect();
  })
}

#[bench]
fn word_tokenizer_bench_long(b: &mut Bencher) {
  b.iter(|| {
    let _: Vec<TrainingToken> = TrainingWordTokenizer::new(
      include_str!("../../test/raw/the-sayings-of-confucius.txt")).collect();
  })
}

#[bench]
fn word_tokenizer_bench_very_long(b: &mut Bencher) {
  b.iter(|| {
    let _: Vec<TrainingToken> = TrainingWordTokenizer::new(
      include_str!("../../test/raw/pride-and-prejudice.txt")).collect();
  })
}
