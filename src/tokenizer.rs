// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::PhantomData;

use token::Token;
use trainer::TrainingData;
use prelude::{DefinesNonPrefixCharacters, DefinesNonWordCharacters, DefinesPunctuation,
              DefinesSentenceEndings};

const STATE_SENT_END: u8 = 0b00000001; // Hit a sentence end state.
const STATE_TOKN_BEG: u8 = 0b00000010; // Token began state.
const STATE_CAPT_TOK: u8 = 0b00000100; // Start capturing token state.
const STATE_UPDT_STT: u8 = 0b10000000; // Update the start token flag.
const STATE_UPDT_RET: u8 = 0b01000000; // Update the position at end flag.

struct PeriodContextTokenizer<'a, P> {
  doc: &'a str,
  pos: usize,
  params: PhantomData<P>,
}

impl<'a, P> PeriodContextTokenizer<'a, P>
where
  P: DefinesNonWordCharacters + DefinesSentenceEndings,
{
  #[inline(always)]
  pub fn new(doc: &'a str) -> PeriodContextTokenizer<'a, P> {
    PeriodContextTokenizer {
      doc: doc,
      pos: 0,
      params: PhantomData,
    }
  }

  /// Performs a lookahead to see if a sentence ending character is actually
  /// the end of the token. If it is the end, `None` is returned. Otherwise,
  /// return `Some(x)` where `x` is the new position to iterate to.
  fn lookahead_is_token(&self) -> Option<usize> {
    let mut pos = self.pos;

    while pos < self.doc.len() {
      let mut iter = self.doc[pos..].chars();
      let cur = iter.nth(0).unwrap();

      match cur {
        // A whitespace is reached before a sentence ending character
        // that could signal the continuation of a token.
        c if c.is_whitespace() => return None,
        // A sentence ending is reached. Check if it could be the beginning
        // of a new token (if there is a space after it, or if the next
        // character is puntuation).
        c if P::is_sentence_ending(&c) => {
          if let Some(nxt) = iter.next() {
            if nxt.is_whitespace() || P::is_nonword_char(&nxt) {
              break;
            }
          } else {
            break;
          }
        }
        _ => (),
      }

      pos += cur.len_utf8();
    }

    Some(pos)
  }
}

impl<'a, P> Iterator for PeriodContextTokenizer<'a, P>
where
  P: DefinesNonWordCharacters + DefinesSentenceEndings,
{
  // (Entire slice of section, beginning of next break (if there is one),
  // start of whitespace before next token, end of entire slice,
  // number of bytes of the last character)
  type Item = (&'a str, usize, usize, usize, usize);

  fn next(&mut self) -> Option<(&'a str, usize, usize, usize, usize)> {
    let mut astart = self.pos;
    let mut wstart = self.pos;
    let mut nstart = self.pos;
    let mut state: u8 = 0;

    while self.pos < self.doc.len() {
      let cur = self.doc[self.pos..].chars().next().unwrap();

      macro_rules! return_token(
        () => (
          {
            let end = self.pos;

            // Return to the start of a any next token that occured
            // with a sentence ending.
            if state & STATE_UPDT_RET != 0 { self.pos = nstart; }

            return Some((
              &self.doc[astart..end],
              nstart,
              wstart,
              end,
              cur.len_utf8()));
          }
        )
      );

      match cur {
        // A sentence ending was encountered. Set the appropriate state.
        // This is done anytime a sentence ender is encountered. It should not
        // affect capturing.
        c if P::is_sentence_ending(&c) => {
          state |= STATE_SENT_END;

          // If an update is needed on the starting position of the entire token
          // update it, and toggle the flag.
          if state & STATE_UPDT_STT != 0 {
            astart = self.pos;
            state ^= STATE_UPDT_STT;
          }

          // Capturing a token, and a sentence ending token is encountered.
          // Flag this token to be revisited.
          if state & STATE_CAPT_TOK != 0 {
            state |= STATE_UPDT_RET;
          }
        }
        // A sentence ending has not yet been countered. If a whitespace is
        // encountered, the start of the token needs to be updated. Set a flag
        // to state this fact. If a non-whitespace is encountered, and the start
        // needs to be updated, then actually update the start position.
        c if state & STATE_SENT_END == 0 => {
          if c.is_whitespace() {
            state |= STATE_UPDT_STT;
          } else if state & STATE_UPDT_STT != 0 {
            astart = self.pos;
            state ^= STATE_UPDT_STT;
          }
        }
        // Hit a sentence end character already, but not yet at token begin.
        // If whitespace is hit, then capturing of token can begin.
        // If a non-word token is hit, then return.
        // Otherwise, no match was made, continue.
        c if state & STATE_SENT_END != 0 && state & STATE_TOKN_BEG == 0 => {
          if c.is_whitespace() {
            state |= STATE_TOKN_BEG;
            wstart = self.pos;
          } else if P::is_nonword_char(&c) {
            self.pos += c.len_utf8();
            nstart = self.pos;

            match self.lookahead_is_token() {
              Some(x) => self.pos = x,
              None => return_token!(),
            }
          } else if !P::is_sentence_ending(&c) {
            state ^= STATE_SENT_END;
          }
        }
        // Capturing the whitespace before a token, and a non-whitespace
        // is encountered. Start capturing that token.
        c if state & STATE_SENT_END != 0 && state & STATE_TOKN_BEG != 0
          && state & STATE_CAPT_TOK == 0 =>
        {
          if !c.is_whitespace() {
            nstart = self.pos;
            state |= STATE_CAPT_TOK;
          }
        }
        // Whitespace after a token has been encountered. Final state -- return.
        c if state & STATE_CAPT_TOK != 0 && c.is_whitespace() => return_token!(),
        // Skip if not in a state at all.
        _ => (),
      }

      self.pos += cur.len_utf8();
    }

    None
  }
}

const NEWLINE_START: u8 = 0b00000001;
const PARAGPH_START: u8 = 0b00000010;
const CAPTURE_START: u8 = 0b00000100;
const CAPTURE_COMMA: u8 = 0b00001000;

pub struct WordTokenizer<'a, P> {
  pos: usize,
  doc: &'a str,
  params: PhantomData<P>,
}

impl<'a, P> WordTokenizer<'a, P>
where
  P: DefinesNonPrefixCharacters + DefinesNonWordCharacters,
{
  #[inline(always)]
  pub fn new(doc: &'a str) -> WordTokenizer<'a, P> {
    WordTokenizer {
      pos: 0,
      doc: doc,
      params: PhantomData,
    }
  }
}

impl<'a, P> Iterator for WordTokenizer<'a, P>
where
  P: DefinesNonPrefixCharacters + DefinesNonWordCharacters,
{
  type Item = Token;

  fn next(&mut self) -> Option<Token> {
    let mut state = if self.pos == 0 { NEWLINE_START } else { 0u8 };
    let mut start = self.pos;
    let mut is_ellipsis = false;

    // Slices the document, and returns the current token.
    macro_rules! return_token(
      () => (
        {
          // Rollback if the reason the capture was ended was because
          // of a comma.
          if state & CAPTURE_COMMA != 0 {
            self.pos -= 1;
          }

          return Some(Token::new(
            &self.doc[start..self.pos],
            is_ellipsis,
            state & PARAGPH_START != 0,
            state & NEWLINE_START != 0));
        }
      )
    );

    while self.pos < self.doc.len() {
      let cur = self.doc[self.pos..].chars().next().unwrap();

      // Periods or dashes are the start of multi-chars. A multi-char
      // is defined as an ellipsis or hyphen (multiple-dashes). If there
      // is a multi-character starting from the current character, return.
      // Otherwise, continue.
      match cur {
        // Both potential multi-char starts. Check for a multi-char. If
        // one exists return it, and modify `self.pos`. Otherwise, continue.
        // If a capture has begin, or a comma was encountered, return the token
        // before this multi-char.
        '.' | '-' => match is_multi_char(self.doc, self.pos) {
          Some(s) => {
            if state & CAPTURE_START != 0 || state & CAPTURE_COMMA != 0 {
              return_token!()
            }

            start = self.pos;
            is_ellipsis = s.ends_with(".");

            self.pos += s.len();

            return_token!()
          }
          None => (),
        },
        // Not a potential multi-char start, continue...
        _ => (),
      }

      match cur {
        // A capture has already started (meaning a valid character was encountered).
        // This block handles the cases with characters during a capture.
        c if state & CAPTURE_START != 0 => {
          match c {
            // Found some whitespace, a non-word. Return the token.
            _ if c.is_whitespace() || P::is_nonword_char(&c) => return_token!(),
            // Valid tokens. If a comma was encountered, reset `CAPTURE_COMMA`, as the comma
            // does not signify the ending of the token.
            _ if c.is_alphanumeric() => {
              if state & CAPTURE_COMMA != 0 {
                state ^= CAPTURE_COMMA;
              }
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
        c if state & CAPTURE_START == 0 && !c.is_whitespace() && !P::is_nonprefix_char(&c) => {
          start = self.pos;
          state |= CAPTURE_START;
        }
        // A non-whitespace was encountered. End with just the character.
        c if !c.is_whitespace() => {
          start = self.pos;
          self.pos += c.len_utf8();
          return_token!()
        }
        // A newline was encountered, and no newline was found before. This
        // signifies a newline, but not a new paragraph.
        '\n' if state & NEWLINE_START == 0 => state |= NEWLINE_START,
        // A newline was encountered, and the above pattern was not matched. This
        // signifies a string of newlines (a new paragraph).
        '\n' => state |= PARAGPH_START,
        _ => (),
      }

      self.pos += cur.len_utf8();
    }

    if state & CAPTURE_START != 0 {
      return_token!()
    }

    None
  }
}

/// Iterator over the byte offsets of a document.
///
/// # Examples
///
/// ```
/// # use punkt::{SentenceByteOffsetTokenizer, TrainingData};
/// # use punkt::params::Standard;
/// #
/// let doc = "this is a great sentence! this is a sad sentence.";
/// let data = TrainingData::english();
///
/// for (start, end) in SentenceByteOffsetTokenizer::<Standard>::new(doc, &data) {
///   println!("{:?}", &doc[start..end]);
/// }
/// ```
pub struct SentenceByteOffsetTokenizer<'a, P> {
  doc: &'a str,
  data: &'a TrainingData,
  iter: PeriodContextTokenizer<'a, P>,
  last: usize,
  params: PhantomData<P>,
}

impl<'a, P> SentenceByteOffsetTokenizer<'a, P>
where
  P: DefinesNonPrefixCharacters
    + DefinesNonWordCharacters
    + DefinesPunctuation
    + DefinesSentenceEndings,
{
  /// Creates a new `SentenceByteOffsetTokenizer`.
  #[inline(always)]
  pub fn new(doc: &'a str, data: &'a TrainingData) -> SentenceByteOffsetTokenizer<'a, P> {
    SentenceByteOffsetTokenizer {
      doc: doc,
      iter: PeriodContextTokenizer::new(doc),
      data: data,
      last: 0,
      params: PhantomData,
    }
  }
}

impl<'a, P> Iterator for SentenceByteOffsetTokenizer<'a, P>
where
  P: DefinesNonPrefixCharacters
    + DefinesNonWordCharacters
    + DefinesPunctuation
    + DefinesSentenceEndings,
{
  type Item = (usize, usize);

  fn next(&mut self) -> Option<(usize, usize)> {
    while let Some((slice, tok_start, ws_start, slice_end, len)) = self.iter.next() {
      let mut prv = None;
      let mut has_sentence_break = false;

      // Get word tokens in the slice. If any of them has a sentence break,
      // then set the flag `has_sentence_break`.
      for mut t in WordTokenizer::<P>::new(slice) {
        // First pass annotation can occur for each token...
        ::util::annotate_first_pass::<P>(&mut t, self.data);

        // Second pass annotation is a bit more finicky...It depends on the
        // previous token that was found.
        match prv {
          Some(mut p) => {
            annotate_second_pass::<P>(&mut t, &mut p, self.data);

            if p.is_sentence_break() {
              has_sentence_break = true;
              break;
            }
          }
          None => (),
        }

        prv = Some(t);
      }

      // If there is a token with a sentence break, it is the end of
      // a sentence. Set the beginning of the next sentence to the start
      // of the start of the token, or the end of the slice if the token is
      // punctuation. Then return the sentence.
      if has_sentence_break {
        let start = self.last;

        return if tok_start == slice_end {
          self.last = slice_end - len;
          Some((start, self.last))
        } else {
          self.last = tok_start;
          Some((start, ws_start))
        };
      }
    }

    // TODO: NLTK gives you back the remaining text as a sentence, including
    // trailing whitespace. Ideally, this wouldn't return trailing whitespace.
    if self.iter.pos == self.doc.len() {
      self.iter.pos += 1;
      Some((self.last, self.doc.len()))
    } else {
      None
    }
  }
}

/// Iterator over the sentence slices of a document.
///
/// # Examples
///
/// ```
/// # use punkt::{SentenceTokenizer, TrainingData};
/// # use punkt::params::Standard;
/// #
/// let doc = "this is a great sentence! this is a sad sentence.";
/// let data = TrainingData::english();
///
/// for sent in SentenceTokenizer::<Standard>::new(doc, &data) {
///   println!("{:?}", sent);
/// }
/// ```
pub struct SentenceTokenizer<'a, P> {
  doc: &'a str,
  iter: SentenceByteOffsetTokenizer<'a, P>,
  params: PhantomData<P>,
}

impl<'a, P> SentenceTokenizer<'a, P>
where
  P: DefinesNonPrefixCharacters
    + DefinesNonWordCharacters
    + DefinesPunctuation
    + DefinesSentenceEndings,
{
  /// Creates a new `SentenceTokenizer`.
  #[inline(always)]
  pub fn new(doc: &'a str, data: &'a TrainingData) -> SentenceTokenizer<'a, P> {
    SentenceTokenizer {
      doc: doc,
      iter: SentenceByteOffsetTokenizer::new(doc, data),
      params: PhantomData,
    }
  }
}

impl<'a, P> Iterator for SentenceTokenizer<'a, P>
where
  P: DefinesNonPrefixCharacters
    + DefinesNonWordCharacters
    + DefinesPunctuation
    + DefinesSentenceEndings,
{
  type Item = &'a str;

  #[inline]
  fn next(&mut self) -> Option<&'a str> {
    self.iter.next().map(|(start, end)| &self.doc[start..end])
  }
}

/// Orthographic heuristic uses structural properties of the token to
/// decide whether a token is the first in a sentence or not. If no
/// decision can be made, None is returned.
fn orthographic_heuristic<P>(tok: &Token, data: &TrainingData) -> Option<bool>
where
  P: DefinesPunctuation,
{
  use prelude::{BEG_LC, MID_UC, ORT_LC, ORT_UC};

  if P::is_punctuation(&tok.tok().chars().nth(0).unwrap()) {
    Some(false)
  } else {
    let ctxt = data.get_orthographic_context(tok.typ_without_break_or_period());

    if tok.is_uppercase() && (ctxt & ORT_LC != 0) && (ctxt & MID_UC == 0) {
      Some(true)
    } else if tok.is_lowercase() && ((ctxt & ORT_UC != 0) || (ctxt & BEG_LC == 0)) {
      Some(false)
    } else {
      None
    }
  }
}

/// Performs a second pass annotation on the tokens revising any previously
/// made decisions if new, relevant data is known.
fn annotate_second_pass<P>(cur: &mut Token, prv: &mut Token, data: &TrainingData)
where
  P: DefinesPunctuation,
{
  use prelude::ORT_LC;

  if data.contains_collocation(prv.typ_without_period(), cur.typ_without_break_or_period()) {
    prv.set_is_abbrev(true);
    prv.set_is_sentence_break(false);
    return;
  }

  if (prv.is_abbrev() || prv.is_ellipsis()) && !prv.is_initial() {
    // Abbreviation with orthographic heuristic
    if orthographic_heuristic::<P>(cur, data).unwrap_or(false) {
      prv.set_is_sentence_break(true);
      return;
    }

    // Abbreviation with sentence starter
    if cur.is_uppercase() && data.contains_sentence_starter(cur.typ_without_break_or_period()) {
      prv.set_is_sentence_break(true);
      return;
    }
  }

  if prv.is_initial() || prv.is_numeric() {
    let ortho_dec = orthographic_heuristic::<P>(cur, data);

    // Initial or Number with orthographic heuristic
    if !ortho_dec.unwrap_or(true) {
      prv.set_is_sentence_break(false);
      prv.set_is_abbrev(true);
      return;
    }

    let ctxt = data.get_orthographic_context(cur.typ_without_break_or_period());

    // Initial with special orthographic heuristic
    if ortho_dec.is_none() && prv.is_initial() && cur.is_uppercase() && ctxt & ORT_LC == 0 {
      prv.set_is_sentence_break(false);
      prv.set_is_abbrev(true);
    }
  }
}

/// Checks if the a slice of the document starting at pos
/// is a multi char (ex. "...", ". . .", "--").
/// These are all one-width chars, so iterating by 1 is OK.
fn is_multi_char(doc: &str, start: usize) -> Option<&str> {
  let mut end = start;
  let mut prv = doc.as_bytes()[start];

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
        break;
      }
    }

    prv = c;
    end += 1;
  }

  if end - start > 1 {
    Some(&doc[start..end])
  } else {
    None
  }
}

#[test]
fn periodctxt_tokenizer_compare_nltk() {
  use std::iter::Iterator;
  use prelude::Standard;

  for (expected, raw, file) in super::get_test_scenarios("test/word-periodctxt/", "test/raw/") {
    let iter: PeriodContextTokenizer<Standard> = PeriodContextTokenizer::new(&raw[..]);

    println!("  running periodctxt tests for '{:?}'", file);

    for ((t, _, _, _, _), e) in iter.zip(expected) {
      let t = t.replace("\n", r"\n").replace("\r", "");
      let e = e.replace("\r", "");

      assert!(t == e, "{} - you: [{}] != exp: [{}]", file, t, e);
    }
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
  use prelude::Standard;

  for (expected, raw, file) in super::get_test_scenarios("test/word-training", "test/raw/") {
    let iter: WordTokenizer<Standard> = WordTokenizer::new(&raw[..]);

    println!("  running wordtok tests for {:?}", file);

    for (t, e) in iter.zip(expected) {
      assert!(
        t.typ().to_lowercase() == e.trim(),
        "{} - you: [{}] != exp: [{}]",
        file,
        t.typ().to_lowercase(),
        e.trim()
      );
    }
  }
}

#[cfg(test)]
fn train_on_document(data: &mut TrainingData, doc: &str) {
  use trainer::Trainer;

  let trainer: Trainer<::prelude::Standard> = Trainer::new();
  trainer.train(&doc, data);
}

#[test]
fn sentence_tokenizer_compare_nltk_train_on_document() {
  let cases = super::get_test_scenarios("test/sentence/", "test/raw/");

  for (expected, raw, file) in cases {
    println!("  running sentencetok tests for {:?}", file);

    let mut data = TrainingData::new();

    train_on_document(&mut data, &raw[..]);

    let iter: SentenceTokenizer<::prelude::Standard> = SentenceTokenizer::new(&raw[..], &data);

    for (t, e) in iter.zip(expected.iter()) {
      let s = format!("[{}]", t)
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "");

      assert!(
        s == e.trim(),
        "{} - you: [{}] != exp: [{}]",
        file,
        s,
        e.trim()
      );
    }
  }
}

// https://github.com/ferristseng/rust-punkt/issues/5
#[test]
fn sentence_tokenizer_issue_5_test() {
  let data = TrainingData::english();
  let doc = "this is a great sentence! this is a sad sentence.";
  let mut iter = SentenceTokenizer::<::params::Standard>::new(doc, &data);

  assert_eq!(iter.next().unwrap(), "this is a great sentence!");
  assert_eq!(iter.next().unwrap(), "this is a sad sentence.");
}

// https://github.com/ferristseng/rust-punkt/issues/8
#[test]
fn sentence_tokenizer_issue_8_test() {
  let data = TrainingData::english();
  let doc = "this is a great sentence! this is a sad sentence.)...";
  let _: Vec<_> = SentenceTokenizer::<::params::Standard>::new(doc, &data).collect();
}

macro_rules! bench_word_tokenizer(
  ($name:ident, $doc:expr) => (
    #[bench] fn $name(b: &mut ::test::Bencher) {
      b.iter(|| {
        let t: WordTokenizer<::prelude::Standard> = WordTokenizer::new($doc);
        let _: Vec<Token> = t.collect();
      })
    }
  )
);

bench_word_tokenizer!(
  word_tokenizer_bench_short,
  include_str!("../test/raw/sigma-wiki.txt")
);

bench_word_tokenizer!(
  word_tokenizer_bench_medium,
  include_str!("../test/raw/npr-article-01.txt")
);

bench_word_tokenizer!(
  word_tokenizer_bench_long,
  include_str!("../test/raw/the-sayings-of-confucius.txt")
);

bench_word_tokenizer!(
  word_tokenizer_bench_very_long,
  include_str!("../test/raw/pride-and-prejudice.txt")
);

macro_rules! bench_sentence_tokenizer(
  ($name:ident, $doc:expr) => (
    #[bench] fn $name(b: &mut ::test::Bencher) {
      let doc = $doc;

      b.iter(|| {
        let mut data = TrainingData::new();

        train_on_document(&mut data, doc);

        let iter: SentenceTokenizer<::prelude::Standard> =
          SentenceTokenizer::new(doc, &mut data);
        let _: Vec<&str> = iter.collect();
      })
    }
  )
);

bench_sentence_tokenizer!(
  bench_sentence_tokenizer_train_on_document_short,
  include_str!("../test/raw/sigma-wiki.txt")
);

bench_sentence_tokenizer!(
  bench_sentence_tokenizer_train_on_document_medium,
  include_str!("../test/raw/npr-article-01.txt")
);

bench_sentence_tokenizer!(
  bench_sentence_tokenizer_train_on_document_long,
  include_str!("../test/raw/pride-and-prejudice.txt")
);
