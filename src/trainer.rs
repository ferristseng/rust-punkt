// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::min;
use std::ops::Deref;
use std::str::FromStr;
use std::default::Default;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::collections::{HashSet, HashMap};

use freqdist::FrequencyDistribution;
use rustc_serialize::json::Json;

use util;
use token::Token;
use tokenizer::WordTokenizer;
use prelude::{TrainerParameters, DefinesNonPrefixCharacters, DefinesNonWordCharacters,
              OrthographicContext, OrthographyPosition};


/// A collocation is any pair of words that has a high likelihood of appearing
/// together.
#[derive(Debug, Eq)]
pub struct Collocation<T>
  where T: Deref<Target = Token>
{
  l: T,
  r: T,
}

impl<T> Collocation<T> where T: Deref<Target = Token>
{
  #[inline(always)]
  pub fn new(l: T, r: T) -> Collocation<T> {
    Collocation { l: l, r: r }
  }

  #[inline(always)]
  pub fn left(&self) -> &T {
    &self.l
  }

  #[inline(always)]
  pub fn right(&self) -> &T {
    &self.r
  }
}

impl<T> Hash for Collocation<T> where T: Deref<Target = Token>
{
  #[inline(always)]
  fn hash<H>(&self, state: &mut H)
    where H: Hasher
  {
    (*self.l).typ_without_period().hash(state);
    (*self.r).typ_without_break_or_period().hash(state);
  }
}

impl<T> PartialEq for Collocation<T> where T: Deref<Target = Token>
{
  #[inline(always)]
  fn eq(&self, x: &Collocation<T>) -> bool {
    (*self.l).typ_without_period() == (*x.l).typ_without_period() &&
    (*self.r).typ_without_break_or_period() == (*x.r).typ_without_break_or_period()
  }
}


/// Stores data that was obtained during training.
///
/// # Examples
///
/// Precompiled data can be loaded via a language specific constructor.
///
/// ```
/// # use punkt::TrainingData;
/// #
/// let eng_data = TrainingData::english();
/// let ger_data = TrainingData::german();
///
/// assert!(eng_data.contains_abbrev("va"));
/// assert!(ger_data.contains_abbrev("crz"));
/// ```
#[derive(Debug, Default)]
pub struct TrainingData {
  abbrevs: HashSet<String>,
  collocations: HashMap<String, HashSet<String>>,
  sentence_starters: HashSet<String>,
  orthographic_context: HashMap<String, OrthographicContext>,
}

impl TrainingData {
  /// Creates a new, empty data object.
  #[inline(always)]
  pub fn new() -> TrainingData {
    TrainingData { ..Default::default() }
  }

  /// Check if a token is considered to be an abbreviation.
  #[inline(always)]
  pub fn contains_abbrev(&self, tok: &str) -> bool {
    self.abbrevs.contains(tok)
  }

  /// Insert a newly learned abbreviation.
  #[inline]
  fn insert_abbrev(&mut self, tok: &str) -> bool {
    if !self.contains_abbrev(tok) {
      self.abbrevs.insert(tok.to_lowercase())
    } else {
      false
    }
  }

  /// Removes a learned abbreviation.
  #[inline]
  fn remove_abbrev(&mut self, tok: &str) -> bool {
    self.abbrevs.remove(tok)
  }

  /// Check if a token is considered to be a token that commonly starts a
  /// sentence.
  #[inline(always)]
  pub fn contains_sentence_starter(&self, tok: &str) -> bool {
    self.sentence_starters.contains(tok)
  }

  /// Insert a newly learned word that signifies the start of a sentence.
  #[inline]
  fn insert_sentence_starter(&mut self, tok: &str) -> bool {
    if !self.contains_sentence_starter(tok) {
      self.sentence_starters.insert(tok.to_string())
    } else {
      false
    }
  }

  /// Checks if a pair of words are commonly known to appear together.
  #[inline]
  pub fn contains_collocation(&self, left: &str, right: &str) -> bool {
    self.collocations
        .get(left)
        .map(|s| s.contains(right))
        .unwrap_or(false)
  }

  /// Insert a newly learned pair of words that frequently appear together.
  fn insert_collocation(&mut self, left: &str, right: &str) -> bool {
    if !self.collocations.contains_key(left) {
      self.collocations.insert(left.to_string(), HashSet::new());
    }

    if !self.collocations.get(left).unwrap().contains(right) {
      self.collocations
          .get_mut(left)
          .unwrap()
          .insert(right.to_string());
      true
    } else {
      false
    }
  }

  /// Insert or update the known orthographic context that a word commonly
  /// appears in.
  #[inline]
  fn insert_orthographic_context(&mut self, tok: &str, ctxt: OrthographicContext) -> bool {
    // `get_mut` isn't allowed here, without adding an unnecessary lifetime
    // qualifier to `tok`.
    match self.orthographic_context.get_mut(tok) {
      Some(c) => {
        *c |= ctxt;
        return false;
      }
      None => (),
    }

    self.orthographic_context.insert(tok.to_string(), ctxt);
    true
  }

  /// Gets the orthographic context for a token. Returns 0 if the token
  /// was not yet encountered.
  #[inline(always)]
  pub fn get_orthographic_context(&self, tok: &str) -> u8 {
    *self.orthographic_context.get(tok).unwrap_or(&0)
  }
}

impl FromStr for TrainingData {
  type Err = &'static str;

  /// Deserializes JSON and loads the data into a new TrainingData object.
  fn from_str(s: &str) -> Result<TrainingData, &'static str> {
    match Json::from_str(s) {
      Ok(Json::Object(mut obj)) => {
        let mut data: TrainingData = Default::default();

        // Macro that gets a Json array by a path on the object. Then does a
        // pattern match on a specified pattern, and runs a specified action.
        macro_rules! read_json_array_data(
          ($path:expr, $mtch:pat, $act:expr) => (
            match obj.remove($path) {
              Some(Json::Array(arr)) => {
                for x in arr.into_iter() {
                  match x {
                    $mtch => { $act; }
                        _ => ()
                  }
                }
              }
              _ => return Err("failed to parse expected path")
            }
          );
        );

        read_json_array_data!("abbrev_types",
                              Json::String(st),
                              data.insert_abbrev(&st[..]));

        read_json_array_data!("sentence_starters",
                              Json::String(st),
                              data.insert_sentence_starter(&st[..]));

        // Load collocations, these come as an array with 2 members in them (or they should).
        // Pop them in reverse order, then insert into the proper bucket.
        read_json_array_data!("collocations", Json::Array(mut ar), {
          match (ar.pop(), ar.pop()) {
            (Some(Json::String(r)), Some(Json::String(l))) =>
              data
                .collocations
                .entry(l)
                .or_insert(HashSet::new())
                .insert(r),
            _ => return Err("failed to parse collocations section")
          };
        });

        match obj.remove("ortho_context") {
          Some(Json::Object(obj)) => {
            for (k, ctxt) in obj.into_iter() {
              ctxt.as_u64()
                  .map(|c| data.orthographic_context.insert(k, c as u8));
            }
          }
          _ => return Err("failed to parse orthographic context section"),
        }

        Ok(data)
      }
      _ => Err("no json object found containing training data"),
    }
  }
}


/// A trainer will build data about abbreviations, sentence starters,
/// collocations, and context that tokens appear in. The data is
/// used by the sentence tokenizer to determine if a period is likely
/// part of an abbreviation, or actually marks the termination of a sentence.
pub struct Trainer<P> {
  params: PhantomData<P>,
}

impl<P> Trainer<P>
  where P: TrainerParameters + DefinesNonPrefixCharacters + DefinesNonWordCharacters
{
  /// Creates a new Trainer.
  #[inline(always)]
  pub fn new() -> Trainer<P> {
    Trainer { params: PhantomData }
  }

  /// Train on a document. Does tokenization using a WordTokenizer.
  pub fn train(&self, doc: &str, data: &mut TrainingData) {
    let mut period_token_count: usize = 0;
    let mut sentence_break_count: usize = 0;
    let tokens: Vec<Token> = WordTokenizer::<P>::new(doc).collect();
    let mut type_fdist: FrequencyDistribution<&str> = FrequencyDistribution::new();
    let mut collocation_fdist = FrequencyDistribution::new();
    let mut sentence_starter_fdist = FrequencyDistribution::new();

    for t in tokens.iter() {
      if t.has_final_period() {
        period_token_count += 1
      }
      type_fdist.insert(t.typ());
    }

    // Iterate through to see if any tokens need to be reclassified as an
    // abbreviation or removed as an abbreviation.
    {
      let reclassify_iter: ReclassifyIterator<_, P> = ReclassifyIterator {
        iter: tokens.iter(),
        data: data,
        period_token_count: period_token_count,
        type_fdist: &mut type_fdist,
        params: PhantomData,
      };

      for (t, score) in reclassify_iter {
        if score >= P::abbrev_lower_bound() {
          if t.has_final_period() {
            unsafe {
              (&mut *(data as *const TrainingData as *mut TrainingData))
                .insert_abbrev(t.typ_without_period());
            }
          }
        } else {
          if !t.has_final_period() {
            unsafe {
              (&mut *(data as *const TrainingData as *mut TrainingData))
                .remove_abbrev(t.typ_without_period());
            }
          }
        }
      }
    }

    // Annotating the tokens requires an unsafe block, but it won't modify any pointers,
    // just will modify some flags on the tokens.
    for t in tokens.iter() {
      unsafe {
        util::annotate_first_pass::<P>(&mut *(t as *const Token as *mut Token), data);
      }
    }

    // Update or insert the orthographic context of all tokens in the document.
    {
      let token_with_context_iter = TokenWithContextIterator {
        iter: tokens.iter(),
        ctxt: OrthographyPosition::Internal,
      };

      for (t, ctxt) in token_with_context_iter {
        if ctxt != 0 {
          data.insert_orthographic_context(t.typ_without_break_or_period(), ctxt);
        }
      }
    }

    // Order matters! Sentence break checks are dependent on whether or not
    // the token is an abbreviation. Must come after the first pass annotation!
    for t in tokens.iter() {
      if t.is_sentence_break() {
        sentence_break_count += 1;
      }
    }

    // Iterate over tokens, and determine if they're abbreviations or if they
    // are potential sentence starters or potential collocations.
    {
      let consecutive_token_iter = ConsecutiveItemIterator {
        iter: tokens.iter(),
        last: None,
      };

      for (lt, rt) in consecutive_token_iter {
        match rt {
          Some(cur) if lt.has_final_period() => {
            if is_rare_abbrev_type::<P>(&data, &type_fdist, lt, cur) {
              data.insert_abbrev(lt.typ_without_period());
            }

            if is_potential_sentence_starter(cur, lt) {
              sentence_starter_fdist.insert(cur);
            }

            if is_potential_collocation::<P>(lt, cur) {
              collocation_fdist.insert(Collocation::new(lt, cur));
            }
          }
          _ => (),
        }
      }
    }

    {
      let ss_iter: PotentialSentenceStartersIterator<_, P> = PotentialSentenceStartersIterator {
        iter: sentence_starter_fdist.keys(),
        sentence_break_count: sentence_break_count,
        type_fdist: &type_fdist,
        sentence_starter_fdist: &sentence_starter_fdist,
        params: PhantomData,
      };

      for (tok, _) in ss_iter {
        data.insert_sentence_starter(tok.typ());
      }
    }

    {
      let clc_iter: PotentialCollocationsIterator<_, P> = PotentialCollocationsIterator {
        iter: collocation_fdist.keys(),
        data: &data,
        type_fdist: &type_fdist,
        collocation_fdist: &collocation_fdist,
        params: PhantomData,
      };

      for (col, _) in clc_iter {
        unsafe {
          (&mut *(data as *const TrainingData as *mut TrainingData))
            .insert_collocation(col.left().typ_without_period(),
                                col.right().typ_without_break_or_period());
        }
      }
    }
  }
}


fn is_rare_abbrev_type<P>(data: &TrainingData,
                          type_fdist: &FrequencyDistribution<&str>,
                          tok0: &Token,
                          tok1: &Token)
                          -> bool
  where P: TrainerParameters
{
  use prelude::{BEG_UC, MID_UC};

  if tok0.is_abbrev() || !tok0.is_sentence_break() {
    false
  } else {
    let key = tok0.typ_without_break_or_period();
    let count = (type_fdist[key] + type_fdist[&key[..key.len() - 1]]) as f64;

    // Already an abbreviation...
    if data.contains_abbrev(tok0.typ()) || count >= P::abbrev_upper_bound() {
      false
    } else if P::is_internal_punctuation(&tok1.typ().chars().next().unwrap()) {
      true
    } else if tok1.is_lowercase() {
      let ctxt = data.get_orthographic_context(tok1.typ_without_break_or_period());

      if (ctxt & BEG_UC > 0) && !(ctxt & MID_UC > 0) {
        true
      } else {
        false
      }
    } else {
      false
    }
  }
}


#[inline(always)]
fn is_potential_sentence_starter(cur: &Token, prev: &Token) -> bool {
  prev.is_sentence_break() && !(prev.is_numeric() || prev.is_initial()) && cur.is_alphabetic()
}


#[inline(always)]
fn is_potential_collocation<P>(tok0: &Token, tok1: &Token) -> bool
  where P: TrainerParameters
{
  P::include_all_collocations() || (P::include_abbrev_collocations() && tok0.is_abbrev()) ||
  (tok0.is_sentence_break() && (tok0.is_numeric() || tok0.is_initial())) && tok0.is_non_punct() &&
  tok1.is_non_punct()
}


/// Iterates over every token from the supplied iterator. Only returns
/// the ones that are 'not obviously' abbreviations. Also returns the associated
/// score of that token.
struct ReclassifyIterator<'b, I, P> {
  iter: I,
  data: &'b TrainingData,
  period_token_count: usize,
  type_fdist: &'b FrequencyDistribution<&'b str>,
  params: PhantomData<P>,
}

impl<'b, I, P> Iterator for ReclassifyIterator<'b, I, P>
  where I: Iterator<Item = &'b Token>,
        P: TrainerParameters
{
  type Item = (&'b Token, f64);

  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    while let Some(t) = self.iter.next() {
      if !t.is_non_punct() || t.is_numeric() {
        continue;
      }

      if t.has_final_period() {
        if self.data.contains_abbrev(t.typ()) {
          continue;
        }
      } else {
        if !self.data.contains_abbrev(t.typ()) {
          continue;
        }
      }

      let num_periods = t.typ_without_period()
                         .chars()
                         .fold(0, |acc, c| {
                           if c == '.' {
                             acc + 1
                           } else {
                             acc
                           }
                         }) + 1;
      let num_nonperiods = t.typ_without_period().chars().count() - num_periods + 1;

      let count_with_period = self.type_fdist.get(t.typ_with_period());
      let count_without_period = self.type_fdist.get(t.typ_without_period());

      let likelihood =
        util::dunning_log_likelihood((count_with_period + count_without_period) as f64,
                                     self.period_token_count as f64,
                                     count_with_period as f64,
                                     self.type_fdist.sum_counts() as f64);

      let f_length = (-(num_nonperiods as f64)).exp();
      let f_penalty = if P::ignore_abbrev_penalty() {
        0f64
      } else {
        (num_nonperiods as f64).powi(-(count_without_period as i32))
      };

      let score = likelihood * f_length * f_penalty * (num_periods as f64);

      return Some((t, score));
    }

    None
  }
}


struct TokenWithContextIterator<I> {
  iter: I,
  ctxt: OrthographyPosition,
}

impl<'a, I> Iterator for TokenWithContextIterator<I> where I: Iterator<Item = &'a Token>
{
  type Item = (&'a Token, OrthographicContext);

  #[inline]
  fn next(&mut self) -> Option<(&'a Token, OrthographicContext)> {
    match self.iter.next() {
      Some(t) => {
        if t.is_paragraph_start() && self.ctxt != OrthographyPosition::Unknown {
          self.ctxt = OrthographyPosition::Initial;
        }

        if t.is_newline_start() && self.ctxt == OrthographyPosition::Internal {
          self.ctxt = OrthographyPosition::Unknown;
        }

        let flag = *::prelude::ORTHO_MAP.get(&(self.ctxt.as_byte() | t.first_case().as_byte()))
                                        .unwrap_or(&0);

        if t.is_sentence_break() {
          if !(t.is_numeric() || t.is_initial()) {
            self.ctxt = OrthographyPosition::Initial;
          } else {
            self.ctxt = OrthographyPosition::Unknown;
          }
        } else if t.is_ellipsis() || t.is_abbrev() {
          self.ctxt = OrthographyPosition::Unknown;
        } else {
          self.ctxt = OrthographyPosition::Internal;
        }

        Some((t, flag))
      }
      None => None,
    }
  }
}


struct PotentialCollocationsIterator<'b, I, P> {
  iter: I,
  data: &'b TrainingData,
  type_fdist: &'b FrequencyDistribution<&'b str>,
  collocation_fdist: &'b FrequencyDistribution<Collocation<&'b Token>>,
  params: PhantomData<P>,
}

impl<'a, 'b, I, P> Iterator for PotentialCollocationsIterator<'b, I, P>
  where I: Iterator<Item = &'a Collocation<&'a Token>>,
        P: TrainerParameters
{
  type Item = (&'a Collocation<&'a Token>, f64);

  #[inline]
  fn next(&mut self) -> Option<(&'a Collocation<&'a Token>, f64)> {
    while let Some(col) = self.iter.next() {
      if self.data.contains_sentence_starter(col.right().typ_without_break_or_period()) {
        continue;
      }

      let count = self.collocation_fdist.get(col);

      let left_count = self.type_fdist.get(col.left().typ_without_period()) +
                       self.type_fdist.get(col.left().typ_with_period());
      let right_count = self.type_fdist.get(col.right().typ_without_period()) +
                        self.type_fdist.get(col.right().typ_with_period());

      if left_count > 1 && right_count > 1 &&
         P::collocation_frequency_lower_bound() < count as f64 &&
         count <= min(left_count, right_count) {
        let likelihood = util::col_log_likelihood(left_count as f64,
                                                  right_count as f64,
                                                  count as f64,
                                                  self.type_fdist.sum_counts() as f64);

        if likelihood >= P::collocation_lower_bound() &&
           (self.type_fdist.sum_counts() as f64 / left_count as f64) >
           (right_count as f64 / count as f64) {
          return Some((col, likelihood));
        }
      }
    }

    None
  }
}


struct PotentialSentenceStartersIterator<'b, I, P> {
  iter: I,
  sentence_break_count: usize,
  type_fdist: &'b FrequencyDistribution<&'b str>,
  sentence_starter_fdist: &'b FrequencyDistribution<&'b Token>,
  params: PhantomData<P>,
}

impl<'a, 'b, I, P> Iterator for PotentialSentenceStartersIterator<'b, I, P>
  where I: Iterator<Item = &'a &'a Token>,
        P: TrainerParameters
{
  type Item = (&'a Token, f64);

  #[inline]
  fn next(&mut self) -> Option<(&'a Token, f64)> {
    while let Some(tok) = self.iter.next() {
      let ss_count = self.sentence_starter_fdist.get(tok);
      let typ_count = self.type_fdist.get(tok.typ_with_period()) +
                      self.type_fdist.get(tok.typ_without_period());

      if typ_count < ss_count {
        continue;
      }

      let likelihood = util::col_log_likelihood(self.sentence_break_count as f64,
                                                typ_count as f64,
                                                ss_count as f64,
                                                self.type_fdist.sum_counts() as f64);

      let ratio = self.type_fdist.sum_counts() as f64 / self.sentence_break_count as f64;

      if likelihood >= P::sentence_starter_lower_bound() &&
         ratio > (typ_count as f64 / ss_count as f64) {
        return Some((*tok, likelihood));
      }
    }

    None
  }
}


struct ConsecutiveItemIterator<'a, T: 'a, I>
  where I: Iterator<Item = &'a T>
{
  iter: I,
  last: Option<&'a T>,
}

impl<'a, T: 'a, I> Iterator for ConsecutiveItemIterator<'a, T, I> where I: Iterator<Item = &'a T>
{
  type Item = (&'a T, Option<&'a T>);

  #[inline]
  fn next(&mut self) -> Option<(&'a T, Option<&'a T>)> {
    match self.last {
      Some(i) => {
        self.last = self.iter.next();
        Some((i, self.last))
      }
      None => {
        match self.iter.next() {
          Some(i) => {
            self.last = self.iter.next();
            Some((i, self.last))
          }
          None => None,
        }
      }
    }
  }
}


// Macro for generating functions to load precompiled data.
macro_rules! preloaded_data(
  ($lang:ident, $file:expr) => (
    impl TrainingData {
      #[inline] #[allow(missing_docs)] pub fn $lang() -> TrainingData {
        FromStr::from_str(include_str!($file)).unwrap()
      }
    }
  )
);


preloaded_data!(czech, "data/czech.json");
preloaded_data!(danish, "data/danish.json");
preloaded_data!(dutch, "data/dutch.json");
preloaded_data!(english, "data/english.json");
preloaded_data!(estonian, "data/estonian.json");
preloaded_data!(finnish, "data/finnish.json");
preloaded_data!(french, "data/french.json");
preloaded_data!(german, "data/german.json");
preloaded_data!(greek, "data/greek.json");
preloaded_data!(italian, "data/italian.json");
preloaded_data!(norwegian, "data/norwegian.json");
preloaded_data!(polish, "data/polish.json");
preloaded_data!(portuguese, "data/portuguese.json");
preloaded_data!(slovene, "data/slovene.json");
preloaded_data!(spanish, "data/spanish.json");
preloaded_data!(swedish, "data/swedish.json");
preloaded_data!(turkish, "data/turkish.json");


#[test]
fn test_data_load_from_json_test() {
  let data: TrainingData = TrainingData::english();

  assert!(data.orthographic_context.len() > 0);
  assert!(data.abbrevs.len() > 0);
  assert!(data.sentence_starters.len() > 0);
  assert!(data.collocations.len() > 0);
  assert!(data.contains_sentence_starter("among"));
  assert!(data.contains_abbrev("w.va"));
  assert!(data.contains_collocation("##number##", "corrections"));
}


macro_rules! bench_trainer(
  ($name:ident, $doc:expr) => (
    #[bench] fn $name(b: &mut ::test::Bencher) {
      b.iter(|| {
        let mut data = TrainingData::new();
        let trainer: Trainer<::prelude::Standard> = Trainer::new();

        trainer.train($doc, &mut data);
      })
    }
  )
);

bench_trainer!(
  bench_trainer_short,
  include_str!("../test/raw/sigma-wiki.txt"));

bench_trainer!(
  bench_trainer_medium,
  include_str!("../test/raw/npr-article-01.txt"));

bench_trainer!(
  bench_trainer_long,
  include_str!("../test/raw/the-sayings-of-confucius.txt"));

bench_trainer!(
  bench_trainer_very_long,
  include_str!("../test/raw/pride-and-prejudice.txt"));
