#[cfg(test)] use test::Bencher;
#[cfg(test)] use trainer::Trainer;

use std::default::Default;

use phf::Set;

use util;
use trainer::TrainingData;
use ortho::{BEG_LC, MID_UC, ORT_LC, ORT_UC};
use tokenizer::{WordTokenizer, WordTokenizerParameters};
use tokenizer::word::DEFAULT as WORD_DEFAULTS;
use tokenizer::periodctxt::PeriodContextTokenizer;
use token::prelude::{WordTokenWithFlagsOps, WordTypeToken, WordTokenWithFlagsOpsExt};

static DEFAULT: SentenceTokenizerParameters<'static> = SentenceTokenizerParameters {
  wtokp: &WORD_DEFAULTS,
  punct: &phf_set![';', ':', ',', '.', '!', '?']
};

/// Parameters for a sentence tokenizer.
pub struct SentenceTokenizerParameters<'a> {
  /// Parameters to use when creating a word tokenizer internally.
  pub wtokp: &'a WordTokenizerParameters,

  /// Characters considered as punctuation.
  pub punct: &'a Set<char>
}

impl<'a> Default for &'static SentenceTokenizerParameters<'a> {
  fn default() -> &'static SentenceTokenizerParameters<'a> {
    &DEFAULT
  }
}

/// Tokenizes a document into sentences using the Punkt algorithm (iterable).
pub struct SentenceTokenizer<'a> {
  doc: &'a str,
  iter: PeriodContextTokenizer<'a>,
  data: &'a TrainingData,
  last: usize,
  #[allow(missing_docs)] pub params: &'a SentenceTokenizerParameters<'a>
}

impl<'a> SentenceTokenizer<'a> {
  /// Creates a new sentence tokenizer across a document, using the provided data 
  /// to perform tokenization.
  #[inline]
  pub fn new(
    doc: &'a str, 
    data: &'a TrainingData
  ) -> SentenceTokenizer<'a> {
    SentenceTokenizer::with_parameters(doc, data, Default::default())
  }

  /// Creates a new sentence tokenizer with custom parameters
  #[inline]
  pub fn with_parameters(
    doc: &'a str,
    data: &'a TrainingData,
    params: &'a SentenceTokenizerParameters
  ) -> SentenceTokenizer<'a> {
    SentenceTokenizer {
      doc: doc,
      iter: PeriodContextTokenizer::new(doc),
      data: data,
      last: 0,
      params: params
    }
  }
}

impl<'a> Iterator for SentenceTokenizer<'a> {
  type Item = &'a str; 

  fn next(&mut self) -> Option<&'a str> {
    loop {
      match self.iter.next() {
        Some((slice, tok_start, ws_start, slice_end)) => {
          let mut prv = None;
          let mut has_sentence_break = false;

          // Get word tokens in the slice. If any of them has a sentence break,
          // then set the flag `has_sentence_break`.
          for mut t in WordTokenizer::with_parameters(slice, self.params.wtokp) {
            // First pass annotation can occur for each token.
            util::annotate_first_pass(
              &mut t, 
              self.data, 
              self.iter.params.sent_end);

            // Second pass annotation is a bit more finicky...It depends on the previous 
            // token that was found.
            match prv {
              Some(mut p) => {
                annotate_second_pass(
                  &mut t, 
                  &mut p,
                  self.data, 
                  self.params.punct);

                if p.is_sentence_break() {
                  has_sentence_break = true;
                  break;
                }
              }
              None => ()
            }

            if t.is_sentence_break() { 
              has_sentence_break = true; 
              break; 
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
              self.last = slice_end - 1;
              Some(&self.doc[start..slice_end - 1])
            } else {
              self.last = tok_start;
              Some(&self.doc[start..ws_start])
            }
          }
        }
        None => break
      }
    }

    None
  }

  #[inline]
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.doc.len() / 10, None)
  }
}

/// Orthographic heuristic uses 'physical' properties of the token to 
/// decide whether a token is the first in a sentence or not. If no 
/// decision can be made, None is returned.
#[inline]
fn orthographic_heuristic<F, T>(
  tok: &T,
  data: &TrainingData,
  punc: &Set<char>
) -> Option<bool>
  where T: WordTokenWithFlagsOps<F> + WordTypeToken
{
  if punc.contains(&tok.token().char_at(0)) {
    Some(false)
  } else {
    let ctxt = *data
      .get_orthographic_context(tok.typ_without_break_or_period())
      .unwrap_or(&0); 

    if tok.is_uppercase() && (ctxt & ORT_LC != 0) && (ctxt & MID_UC == 0) 
    {
      Some(true)
    } else if tok.is_lowercase() && (ctxt & ORT_UC != 0) || (ctxt & BEG_LC == 0)
    {
      Some(false)
    } else {
      None
    }
  }
}

/// Performs a second pass annotation on the tokens revising any 
fn annotate_second_pass<F, T>(
  cur: &mut T,
  prv: &mut T,
  data: &TrainingData,
  punc: &Set<char>
)
  where T: WordTokenWithFlagsOps<F> + WordTokenWithFlagsOpsExt<F> + WordTypeToken
{
  // Known Collocation
  if data.contains_collocation(
    prv.typ_without_period(), cur.typ_without_break_or_period())
  {
    prv.set_is_abbrev(true);
    prv.set_is_sentence_break(false);
    return; 
  }

  if (prv.is_abbrev() || prv.is_ellipsis()) && !prv.is_initial() {
    // Abbreviation with orthographic heuristic
    if orthographic_heuristic(cur, data, punc).unwrap_or(false) {
      prv.set_is_sentence_break(true);
      return; 
    }

    // Abbreviation with sentence starter
    if cur.is_uppercase() && 
       data.contains_sentence_starter(cur.typ_without_break_or_period())
    {
      prv.set_is_sentence_break(true); 
      return;
    }
  }

  if prv.is_initial() || prv.is_numeric() {
    let ortho_dec = orthographic_heuristic(cur, data, punc);

    // Initial or Number with orthographic heuristic
    if !ortho_dec.unwrap_or(true) {
      prv.set_is_sentence_break(false);
      prv.set_is_abbrev(true); 
      return;
    }

    let ctxt = *data
      .get_orthographic_context(cur.typ_without_break_or_period())
      .unwrap_or(&0);

    // Initial with special orthographic heuristic
    if ortho_dec.is_none() && 
       prv.is_initial() && 
       cur.is_uppercase() &&
       ctxt & ORT_LC == 0
    {
      prv.set_is_sentence_break(false);
      prv.set_is_abbrev(true);
    }
  }
}

#[inline] #[cfg(test)]
fn train_on_document(data: &mut TrainingData, doc: &str) {
  let mut trainer = Trainer::new(data);

  trainer.train(&doc);
  trainer.finalize();
}

#[test]
fn sentence_tokenizer_compare_nltk_train_on_document() {
  let cases = super::get_test_scenarios("test/sentence/", "test/raw/");

  for (expected, raw, file) in cases {
    let mut data = Default::default();

    train_on_document(&mut data, &raw[..]);

    for (t, e) in SentenceTokenizer::new(&raw[..], &data).zip(expected.iter()) {
      let s = format!("[{}]", t)
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "");

      assert!(
        s == e.trim(),
        "{} - you: [{}] != exp: [{}]", 
        file,
        s,
        e.trim());
    }
  }
}

macro_rules! bench_sentence_tokenizer(
  ($name:ident, $doc:expr) => (
    #[bench]
    fn $name(b: &mut Bencher) {
      let doc = $doc;

      b.iter(|| {
        let mut data = Default::default();

        train_on_document(&mut data, doc);

        let _: Vec<&str> = SentenceTokenizer::new(doc, &mut data).collect();
      })
    }
  )
);

bench_sentence_tokenizer!(
  bench_sentence_tokenizer_train_on_document_short, 
  include_str!("../../test/raw/sigma-wiki.txt"));

bench_sentence_tokenizer!(
  bench_sentence_tokenizer_train_on_document_medium,
  include_str!("../../test/raw/npr-article-01.txt"));

bench_sentence_tokenizer!(
  bench_sentence_tokenizer_train_on_document_long,
  include_str!("../../test/raw/pride-and-prejudice.txt"));
