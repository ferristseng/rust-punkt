#[cfg(test)] use std::io::fs;
#[cfg(test)] use std::io::fs::PathExtensions;
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

static DEFAULT: SentenceTokenizerParams<'static> = SentenceTokenizerParams {
  wtokp: &WORD_DEFAULTS,
  punct: &phf_set![';', ':', ',', '.', '!', '?']
};

pub struct SentenceTokenizerParams<'a> {
  pub wtokp: &'a WordTokenizerParameters,
  pub punct: &'a Set<char>
}

impl<'a> Default for &'static SentenceTokenizerParams<'a> {
  fn default() -> &'static SentenceTokenizerParams<'a> {
    &DEFAULT
  }
}

pub struct SentenceTokenizer<'a> {
  doc: &'a str,
  iter: PeriodContextTokenizer<'a>,
  data: &'a TrainingData,
  last: usize,
  pub params: &'a SentenceTokenizerParams<'a>
}

impl<'a> SentenceTokenizer<'a> {
  pub fn new(
    doc: &'a str, 
    data: &'a TrainingData
  ) -> SentenceTokenizer<'a> {
    SentenceTokenizer {
      doc: doc,
      iter: PeriodContextTokenizer::new(doc),
      data: data,
      last: 0,
      params: Default::default()
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
            match prv {
              Some(mut p) => {
                util::annotate_first_pass(
                  &mut t, 
                  self.data, 
                  self.iter.params.sent_end);

                annotate_second_pass(
                  &mut p, 
                  &mut t,
                  self.data, 
                  self.params.punct);

                if t.is_sentence_break() { 
                  has_sentence_break = true; 
                  break; 
                }

                prv = Some(t);
              }
              None => {
                prv = Some(t)
              }
            }
          }

          // If there is a token with a sentence break, it is the end of 
          // a sentence. Set the beginning of the next sentence to the start 
          // of the start of the token, or the end of the slice if the token is 
          // punctuation. Then return the sentence.
          if has_sentence_break {
            let start = self.last;

            return if tok_start == slice_end {
              self.last = slice_end - 1;
              Some(self.doc.slice(start, slice_end - 1))
            } else {
              self.last = tok_start;
              Some(self.doc.slice(start, ws_start))
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

    if tok.is_uppercase() && 
       (ctxt & ORT_LC != 0) && 
       (ctxt & MID_UC == 0) 
    {
      Some(true)
    } else if tok.is_lowercase() &&  
      ((ctxt & ORT_UC != 0) || (ctxt & BEG_LC == 0))
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

#[test]
fn sentence_tokenizer_compare_nltk_train_on_document() {
  for path in fs::walk_dir(&Path::new("test/sentence/")).unwrap() {
    if path.is_file() {
      let rawp = Path::new("test/raw/").join(path.filename_str().unwrap());
      let expf = fs::File::open(&path).read_to_string().unwrap();
      let rawf = fs::File::open(&rawp).read_to_string().unwrap();
      let exps = expf.split('\n');
      let mut data = Default::default();

      {
        let mut trainer = Trainer::new(&mut data);

        trainer.train(rawf.as_slice());
        trainer.finalize();
      }

      for (t, e) in SentenceTokenizer::new(rawf.as_slice(), &data).zip(exps) {
        let s = format!("[{}]", t)
          .replace("\"", "\\\"")
          .replace("\n", "\\n")
          .replace("\r", "");

        assert!(
          s == e.trim(),
          "{} - you: [{}] != exp: [{}]", 
          path.filename_str().unwrap(),
          s,
          e.trim())
      }
    }
  }
}