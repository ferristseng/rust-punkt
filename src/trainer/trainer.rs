use std::cmp::min;
use std::ops::Deref;
use std::default::Default;
use collections::borrow::Borrow;

use phf::Set;
use num::Float;


use util;
use trainer::math;
use trainer::TrainingTokenKey;
use trainer::col::Collocation;
use trainer::data::TrainingData;
use tokenizer::{WordTokenizer, WordTokenizerParameters};
use ortho::{BEG_UC, MID_UC, OrthographyPosition, OrthographicContext, ORTHO_MAP};


/// Trainer to compile data about frequent sentence staters, collocations, 
/// and potential abbreviations.
///
/// After you've trained on any number of documents, you can call `finalize` 
/// to compile the trained data. You can use and modify preexisting data, if 
/// you instantiate with the `with_data` constructor.
pub struct Trainer<'a> {
  period_token_count: usize,
  sentence_break_count: usize,
  data: &'a mut TrainingData,
  tokens: Vec<TrainingTokenKey>,
  type_fdist: FrequencyDistribution<TrainingTokenKey>,
  collocation_fdist: FrequencyDistribution<Collocation<TrainingTokenKey>>,
  sentence_starter_fdist: FrequencyDistribution<TrainingTokenKey>
  params: PhantomData
}

impl<'a> Trainer<'a> {
  /// Creates a trainer from borrowed `TrainingData`. The trainer mutably borrows 
  /// the training data for its lifetime, and inorder to reacquire the borrow, the 
  /// Training step must be wrapped in a block.
  #[inline(always)] pub fn new(data: &'a mut TrainingData) -> Trainer<'a> {
    Trainer {
      period_token_count: 0,
      sentence_break_count: 0,
      data: data,
      tokens: Vec::new(),
      type_fdist: FrequencyDistribution::new(),
      collocation_fdist: FrequencyDistribution::new(),
      sentence_starter_fdist: FrequencyDistribution::new(),
      params: PhantomData
    }
  }


  /// This isn't entirely safe, so should be used with extreme caution. It
  /// returns a mutable reference to data on the Trainer. Mostly this is 
  /// for the scenario when `data` needs to be modified while iterating, and 
  /// self needs to be borrowed. It can be reasoned generally that iterating over
  /// a part of `self` that isn't `data` makes it safe to modify `data` in most cases.
  #[inline]
  unsafe fn borrow_data_mut_unsafe(&self) -> &mut TrainingData {
    &mut *(&*self.data as *const TrainingData as *mut TrainingData)
  }

  /// Train on a document. Does tokenization using a WordTokenizer.
  pub fn train(&mut self, doc: &str) {
    // `self.tokens` hold all the tokens that were encountered during 
    // training. In order to get only the ones for the inputted document, 
    // the current length needs to be saved.
    let start = self.tokens.len();

    // Push new tokens that the tokenizer finds from doc into `self.tokens`.
    for t in WordTokenizer::with_parameters(doc, self.tparams) { 
      self.tokens.push(TrainingTokenKey::new(t));
    }

    // Acquire the slice from `self.tokens` of tokens only found from this 
    // document.
    let slice = &self.tokens[start..];

    // Keep counts of each type for each token in `self.type_fdist`.
    for t in slice.iter() {
      self.type_fdist.insert(t.clone());

      if t.has_final_period() { self.period_token_count += 1 }
    }

    // Iterate through to see if any tokens need to be reclassified as an abbreviation
    // or removed as an abbreviation.
    for (t, score) in reclassify_iter(self, self.type_fdist.keys()) {
      if score >= self.params.abbrev_lower_bound { 
        if t.has_final_period() {
          unsafe {
            self.borrow_data_mut_unsafe().insert_abbrev(t.typ_without_period());
          }
        }
      } else {
        if !t.has_final_period() {
          unsafe {
            self.borrow_data_mut_unsafe().remove_abbrev(t.typ_without_period());
          }
        }
      }
    }

    // Mark abbreviation types if any exist with the first pass annotation function.
    // Note, this also sets `is_sentence_break` flag.
    for t in slice.iter() {
      // TrainingTokenKey doesn't provide a mutable interface into a Token by default. 
      // We have to coerce the Token into being mutable. This is safe, since 
      // `annotate_first_pass` only modifies the flags. Nothing is being added or removed
      // from the list of tokens.
      unsafe {
        util::annotate_first_pass(
          &mut *(t.deref() as *const TrainingToken as *mut TrainingToken),
          self.data,
          self.params.sent_end);
      }
    }

    for (t, ctxt) in orthography_iter(slice.iter()) {
      if ctxt != 0 {
        self.data.insert_orthographic_context(t.typ_without_break_or_period(), ctxt);
      }
    }

    // Order matters! Sentence break checks are dependent on whether or not 
    // the token is an abbreviation. Must come after the first pass annotation!
    for t in slice.iter() {
      if t.is_sentence_break() { self.sentence_break_count += 1; }
    }

    for (lt, rt) in consecutive_token_iter(slice.iter()) {
      match rt {
        Some(cur) if lt.has_final_period() => {
          if is_rare_abbrev_type(self, lt.deref(), cur.deref()) {
            self.data.insert_abbrev(lt.typ_without_period());
          }

          if is_potential_sentence_starter(cur.deref(), lt.deref()) {
            self.sentence_starter_fdist.insert(cur.clone());
          }

          if is_potential_collocation(self, lt.deref(), cur.deref()) {
            self.collocation_fdist.insert(Collocation::new(lt.clone(), cur.clone()));
          }
        }
        _ => ()
      }
    }
  }

  /// Empties the trained data, and compiles it with mutably borrow training data. 
  /// Afterwards, the trainer should be dropped (suggested), although finalizing 
  /// could theoretically could occur between each training stage. 
  pub fn finalize(&mut self) {
    // This method does a lot of `unsafe` things that are actually safe. The issue 
    // is it requires borrow `self` mutably to create the iterators, and to 
    // modify the internal data object. Since these two things are completely 
    // separate, these `unsafe` blocks should be safe.

    self.data.clear_sentence_starters();

    for (tok, _) 
    in potential_sentence_starter_iter(
      self, 
      self.sentence_starter_fdist.keys())
    {
      unsafe {
        self.borrow_data_mut_unsafe().insert_sentence_starter(tok.typ());
      }
    }

    self.data.clear_collocations();

    for (col, _)
    in potential_collocation_iter(
      self,
      self.collocation_fdist.keys())
    {
      unsafe {
        self.borrow_data_mut_unsafe().insert_collocation(
          col.left().deref().typ_without_period(), 
          col.right().deref().typ_without_break_or_period());
      }
    }

    self.period_token_count = 0;
    self.sentence_break_count = 0;
    self.type_fdist.clear();
    self.sentence_starter_fdist.clear();
    self.collocation_fdist.clear();
    self.tokens.clear();
  }
}

fn is_rare_abbrev_type(
  trainer: &Trainer,
  tok0: &TrainingToken, 
  tok1: &TrainingToken
) -> bool {
  if tok0.is_abbrev() || !tok0.is_sentence_break() {
    // Check the first condition, and return if it matches
    false
  } else {
    let key: &str = tok0.typ_without_break_or_period().borrow();

    // Count all variations of the token
    let count = trainer.type_fdist.get(key) + trainer.type_fdist.get(&key[..key.len() - 1]);

    if trainer.data.contains_abbrev(tok0.typ()) || 
       (count as f64) >= trainer.params.abbrev_upper_bound 
    {
      // Check the second condition. Return if it's true...the token is 
      // already an abbreviation!
      false
    } else if trainer.params.internal_punctuation.contains(&tok1.typ().char_at(0)) {
      // Check the first case of the final condition
      true
    } else if tok1.is_lowercase() {
      let ctxt = *trainer
        .data
        .get_orthographic_context(tok1.typ_without_break_or_period())
        .unwrap_or(&0);

      // Check the final condition
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

#[inline]
fn is_potential_sentence_starter(
  cur: &TrainingToken, 
  prev: &TrainingToken
) -> bool {
  prev.is_sentence_break() && 
  !(prev.is_numeric() || prev.is_initial()) && 
  cur.is_alphabetic()
}

#[inline]
fn is_potential_collocation(
  trainer: &Trainer,
  tok0: &TrainingToken,
  tok1: &TrainingToken
) -> bool {
  (trainer.params.include_all_collocations ||
  (trainer.params.include_abbrev_collocations && tok0.is_abbrev()) ||
  (tok0.is_sentence_break() && 
    (tok0.is_numeric() || tok0.is_initial())) &&
    tok0.is_non_punct() && 
    tok1.is_non_punct())
}

#[inline]
fn reclassify_iter<'a, 'b, I>(
  trainer: &'b Trainer<'b>,
  iter: I
) -> PunktReclassifyIterator< 'b, I> 
  where I: Iterator<Item = &'a TrainingTokenKey> 
{
  PunktReclassifyIterator { iter: iter, trainer: trainer }
}

#[inline]
fn orthography_iter<'a, I>(
  iter: I
) -> TokenWithContextIterator<I> 
  where I: Iterator<Item = &'a TrainingTokenKey>
{
  TokenWithContextIterator { iter: iter, ctxt: OrthographyPosition::Internal }
}

#[inline]
fn potential_sentence_starter_iter<'a, 'b, I>(
  trainer: &'b Trainer,
  iter: I
) -> PotentialSentenceStartersIterator<'b, I> 
  where I: Iterator<Item = &'a TrainingTokenKey>
{
  PotentialSentenceStartersIterator { iter: iter, trainer: trainer }
}

#[inline]
fn potential_collocation_iter<'a, 'b, I>(
  trainer: &'b Trainer,
  iter: I
) -> PotentialCollocationsIterator<'b, I> 
  where I: Iterator<Item = &'a Collocation<TrainingTokenKey>>
{
  PotentialCollocationsIterator { iter: iter, trainer: trainer }
}

/// A token and its associated score (likelihood of it being a abbreviation).
type ScoredToken<'a> = (&'a TrainingToken, f64);

/// Iterates over every token from the supplied iterator. Only returns 
/// the ones that are 'not obviously' abbreviations. Also returns the associated 
/// score of that token.
struct PunktReclassifyIterator<'b, I>
{
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'b, I> Iterator for PunktReclassifyIterator<'b, I> 
  where I: Iterator<Item = &'b TrainingTokenKey>
{
  type Item = ScoredToken<'b>;

  #[inline]
  fn next(&mut self) -> Option<ScoredToken<'b>> {
    loop {
      match self.iter.next() {
        Some(t) => {
          let t = t.deref();

          // Numeric tokens or ones that are entirely NOT alphabetic are 
          // 'obviously not' abbreviations.
          if !t.is_non_punct() || t.is_numeric()
          {
            continue;
          }

          if t.has_final_period() {
            if self.trainer.data.contains_abbrev(t.typ()) {
              continue;
            }
          } else {
            if !self.trainer.data.contains_abbrev(t.typ()) {
              continue;
            }
          }

          let num_periods = t
            .typ_without_period()
            .chars()
            .fold(0, |acc, c| if c == '.' { acc + 1 } else { acc }) + 1;
          let num_nonperiods = t.typ_without_period().chars().count() - num_periods + 1;

          let count_with_period = self.trainer.type_fdist.get(t.typ_with_period());
          let count_without_period = self.trainer.type_fdist.get(t.typ_without_period());

          let likelihood = math::dunning_log_likelihood(
            (count_with_period + count_without_period) as f64,
            self.trainer.period_token_count as f64,
            count_with_period as f64,
            self.trainer.type_fdist.sum_counts() as f64);

          let f_length = (-(num_nonperiods as f64)).exp();
          let f_penalty = if self.trainer.params.ignore_abbrev_penalty {
            0f64
          } else {
            (num_nonperiods as f64).powi(-(count_without_period as i32))
          };

          let score = likelihood * f_length * f_penalty * (num_periods as f64);

          return Some((t, score))
        }
        None => return None
      }
    }
  }
}

/// Token annotated with its orthographic context.
type TokenWithContext<'a> = (&'a TrainingToken, OrthographicContext);

/// Iterates over every token from the supplied iterator and returns its
/// decided orthography within the given text. 
struct TokenWithContextIterator<I> 
{
  iter: I,
  ctxt: OrthographyPosition
}

impl<'a, I> Iterator for TokenWithContextIterator<I>
  where I: Iterator<Item = &'a TrainingTokenKey>
{
  type Item = TokenWithContext<'a>;

  /// Returns tokens annotated with their OrthographicContext. Must keep track 
  /// and modify internal position of where previous tokens were.
  #[inline]
  fn next(&mut self) -> Option<TokenWithContext<'a>> {
    match self.iter.next() {
      Some(t) => {
        let t = t.deref();

        if t.is_paragraph_start() && self.ctxt != OrthographyPosition::Unknown {
          self.ctxt = OrthographyPosition::Initial;
        }

        if t.is_newline_start() && self.ctxt == OrthographyPosition::Internal {
          self.ctxt = OrthographyPosition::Unknown;
        }

        let flag = *ORTHO_MAP
          .get(&(self.ctxt.as_byte() | t.first_case().as_byte()))
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
      None => None
    }
  }
}

/// Iterates over every potential Collocation (determined by log likelihood).
/// Also returns the likelihood of the potential collocation.
struct PotentialCollocationsIterator<'b, I> 
{
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'a, 'b, I> Iterator for PotentialCollocationsIterator<'b, I> 
  where I: Iterator<Item = &'a Collocation<TrainingTokenKey>>
{
  type Item = (&'a Collocation<TrainingTokenKey>, f64);

  #[inline]
  fn next(&mut self) -> Option<(&'a Collocation<TrainingTokenKey>, f64)> {
    loop {
      match self.iter.next() {
        Some(col) => {
          if self.trainer.data.contains_sentence_starter(
             col.right().typ_without_break_or_period()) 
          {
            continue;    
          }

          let count = self.trainer.collocation_fdist.get(col);

          let left_count = 
            self.trainer.type_fdist.get(col.left().typ_without_period()) +
            self.trainer.type_fdist.get(col.left().typ_with_period());
          let right_count = 
            self.trainer.type_fdist.get(col.right().typ_without_period()) + 
            self.trainer.type_fdist.get(col.right().typ_with_period());

          if left_count > 1 && 
             right_count > 1 &&
             self.trainer.params.collocation_frequency_lower_bound < count as f64 &&
             count <= min(left_count, right_count)
          {
            let likelihood = math::col_log_likelihood(
              left_count as f64,
              right_count as f64,
              count as f64,
              self.trainer.type_fdist.sum_counts() as f64);

            if likelihood >= self.trainer.params.collocation_lower_bound &&
               (self.trainer.type_fdist.sum_counts() as f64 / left_count as f64) >
               (right_count as f64 / count as f64)
            {
              return Some((col, likelihood))
            }
          }
        }
        None => return None
      }
    }
  }
}

struct PotentialSentenceStartersIterator< 'b, I>
{
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'a, 'b, I> Iterator for PotentialSentenceStartersIterator<'b, I> 
  where I: Iterator<Item = &'a TrainingTokenKey>
{
  type Item = ScoredToken<'a>;

  #[inline]
  fn next(&mut self) -> Option<ScoredToken<'a>> {
    loop {
      match self.iter.next() {
        Some(tok) => {
          let ss_count = self.trainer.sentence_starter_fdist.get(tok.typ());
          let typ_count = 
            self.trainer.type_fdist.get(tok.typ_with_period()) + 
            self.trainer.type_fdist.get(tok.typ_without_period());

          if typ_count < ss_count { continue; }

          let likelihood = math::col_log_likelihood(
            self.trainer.sentence_break_count as f64,
            typ_count as f64,
            ss_count as f64,
            self.trainer.type_fdist.sum_counts() as f64);

          if likelihood >= self.trainer.params.sentence_starter_lower_bound &&
             (self.trainer.type_fdist.sum_counts() as f64 / 
             self.trainer.sentence_break_count as f64) > 
             (typ_count as f64 / ss_count as f64)
          {
            return Some((tok.deref(), likelihood));
          }
        }
        None => return None
      }
    }
  }
}

/// Constructor for a consecutive token iterator.
#[inline]
fn consecutive_token_iter<'a, T, I>(
  iter: I
) -> ConsecutiveTokenIterator<'a, T, I>
  where I: Iterator<Item = &'a T> 
{
  ConsecutiveTokenIterator { iter: iter, last: None }
}

/// Iterates over every PuntkToken from the supplied iterator and returns 
/// the immediate following token. Returns None for the following token on the 
/// last token.
struct ConsecutiveTokenIterator<'a, T: 'a, I> 
  where I: Iterator<Item = &'a T>
{
  iter: I,
  last: Option<&'a T>
}

impl<'a, T: 'a, I> Iterator for ConsecutiveTokenIterator<'a, T, I>
  where I: Iterator<Item = &'a T>
{
  type Item = (&'a T, Option<&'a T>); 

  #[inline]
  fn next(&mut self) -> Option<(&'a T, Option<&'a T>)> {
    match self.last {
      Some(tok) => {
        self.last = self.iter.next();

        Some((tok, self.last))
      }
      None => match self.iter.next() {
        Some(tok) => {
          self.last = self.iter.next();

          Some((tok, self.last))
        }
        None => None
      }
    }
  }
}

macro_rules! bench_trainer(
  ($name:ident, $doc:expr) => (
    #[bench]
    fn $name(b: &mut ::test::Bencher) {
      b.iter(|| {
        let mut data = Default::default();
        let mut trainer = Trainer::new(&mut data);

        trainer.train($doc);
        trainer.finalize();
      })
    }
  )
);

bench_trainer!(
  bench_trainer_short, 
  include_str!("../../test/raw/sigma-wiki.txt"));

bench_trainer!(
  bench_trainer_medium,
  include_str!("../../test/raw/npr-article-01.txt"));

bench_trainer!(
  bench_trainer_long,
  include_str!("../../test/raw/the-sayings-of-confucius.txt"));

bench_trainer!(
  bench_trainer_very_long,
  include_str!("../../test/raw/pride-and-prejudice.txt"));
