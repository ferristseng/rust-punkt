use std::rc::Rc;
use std::cmp::min;
use std::num::Float;
use std::ops::Deref;

use freqdist::Distribution;

use trainer::math;
use trainer::Trainer;
use trainer::col::Collocation;
use token::TrainingToken;
use token::prelude::{WordTypeToken, WordTokenWithFlagsOps, WordTokenWithFlagsOpsExt};
use ortho::{
  OrthographyPosition, 
  OrthographicContext,
  BEG_UC, 
  MID_UC, 
  ORTHO_MAP};

#[inline]
pub fn reclassify_iter<'a, 'b, I: Iterator<Item = &'a Rc<TrainingToken>>>(
  trainer: &'b Trainer<'b>,
  iter: I
) -> PunktReclassifyIterator<'a, 'b, I> {
  PunktReclassifyIterator { iter: iter, trainer: trainer }
}

#[inline]
pub fn orthography_iter<'a, I: Iterator<Item = &'a Rc<TrainingToken>>>(
  iter: I
) -> TokenWithContextIterator<'a, I> {
  TokenWithContextIterator { iter: iter, ctxt: OrthographyPosition::Internal }
}

#[inline]
pub fn potential_sentence_starter_iter<
  'a, 
  'b, 
  I: Iterator<Item = &'a Rc<TrainingToken>>>
(
  trainer: &'b Trainer,
  iter: I
) -> PotentialSentenceStartersIterator<'a, 'b, I> {
  PotentialSentenceStartersIterator { iter: iter, trainer: trainer }
}

#[inline]
pub fn potential_collocation_iter<
  'a, 
  'b, 
  I: Iterator<Item = &'a Collocation<Rc<TrainingToken>>>>
(
  trainer: &'b Trainer,
  iter: I
) -> PotentialCollocationsIterator<'a, 'b, I> {
  PotentialCollocationsIterator { iter: iter, trainer: trainer }
}

#[inline]
pub fn consecutive_token_iter_mut<'a, T, I: Iterator<Item = &'a mut T>>(
  iter: I
) -> ConsecutiveTokenMutIterator<'a, T, I> {
  ConsecutiveTokenMutIterator { iter: iter, last: None }
}

#[inline]
pub fn consecutive_token_iter<'a, T, I: Iterator<Item = &'a T>>(
  iter: I
) -> ConsecutiveTokenIterator<'a, T, I> {
  ConsecutiveTokenIterator { iter: iter, last: None }
}

/// A token and its associated score (likelihood of it being a abbreviation).
type ScoredToken<'a> = (&'a TrainingToken, f64);

/// Iterates over every token from the supplied iterator. Only returns 
/// the ones that are 'not obviously' abbreviations. Also returns the associated 
/// score of that token.
struct PunktReclassifyIterator<'a: 'b, 'b, I: Iterator<Item = &'a Rc<TrainingToken>>> {
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'a, 'b, I: Iterator<Item = &'a Rc<TrainingToken>>> Iterator 
for PunktReclassifyIterator<'a, 'b, I> {
  type Item = ScoredToken<'a>;

  #[inline]
  fn next(&mut self) -> Option<ScoredToken<'a>> {
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

          let count_with_period = *self
            .trainer
            .type_fdist
            .get(t.typ_with_period())
            .unwrap_or(&0);
          let count_without_period = *self
            .trainer
            .type_fdist
            .get(t.typ_without_period())
            .unwrap_or(&0);

          let likelihood = math::dunning_log_likelihood(
            (count_with_period + count_without_period) as f64,
            self.trainer.period_token_count as f64,
            count_with_period as f64,
            self.trainer.type_fdist.sum_counts() as f64);

          let f_length = (-(num_nonperiods as f64)).exp();
          let f_penalty = if self.trainer.parameters.ignore_abbrev_penalty {
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

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    self.iter.size_hint()
  }
}

/// Token annotated with its orthographic context.
type TokenWithContext<'a> = (&'a TrainingToken, OrthographicContext);

/// Iterates over every token from the supplied iterator and returns its
/// decided orthography within the given text. 
struct TokenWithContextIterator<'a, I: Iterator<Item = &'a Rc<TrainingToken>>> {
  iter: I,
  ctxt: OrthographyPosition
}

impl<'a, I: Iterator<Item = &'a Rc<TrainingToken>>> Iterator 
for TokenWithContextIterator<'a, I>
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
  
  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    self.iter.size_hint()
  }
}

/// Iterates over every potential Collocation (determined by log likelihood).
/// Also returns the likelihood of the potential collocation.
struct PotentialCollocationsIterator<
  'a, 
  'b, 
  I: Iterator<Item = &'a Collocation<Rc<TrainingToken>>>> 
{
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'a, 'b, I: Iterator<Item = &'a Collocation<Rc<TrainingToken>>>> Iterator
for PotentialCollocationsIterator<'a, 'b, I> {
  type Item = (&'a Collocation<Rc<TrainingToken>>, f64);

  #[inline]
  fn next(&mut self) -> Option<(&'a Collocation<Rc<TrainingToken>>, f64)> {
    loop {
      match self.iter.next() {
        Some(col) => {
          if self.trainer.data.contains_sentence_starter(
            col.right().typ_without_break_or_period()) 
          {
            continue;    
          }

          let count = *self.trainer.collocation_fdist.get(col).unwrap_or(&0);

          let left_count = 
            *self.trainer.type_fdist.get(col.left().typ_without_period()).unwrap_or(&0) +
            *self.trainer.type_fdist.get(col.left().typ_with_period()).unwrap_or(&0);
          let right_count = 
            *self.trainer.type_fdist.get(col.right().typ_without_period()).unwrap_or(&0) + 
            *self.trainer.type_fdist.get(col.right().typ_with_period()).unwrap_or(&0);

          if left_count > 1 && 
            right_count > 1 &&
            self.trainer.parameters.collocation_frequency_lower_bound < count as f64 &&
            count <= min(left_count, right_count)
          {
            let likelihood = math::col_log_likelihood(
              left_count as f64,
              right_count as f64,
              count as f64,
              self.trainer.type_fdist.sum_counts() as f64);

            if likelihood >= self.trainer.parameters.collocation_lower_bound &&
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

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    self.iter.size_hint()
  }
}

struct PotentialSentenceStartersIterator<
  'a, 
  'b, 
  I: Iterator<Item = &'a Rc<TrainingToken>>>
{
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'a, 'b, I: Iterator<Item = &'a Rc<TrainingToken>>> Iterator
for PotentialSentenceStartersIterator<'a, 'b, I> {
  type Item = ScoredToken<'a>;

  #[inline]
  fn next(&mut self) -> Option<ScoredToken<'a>> {
    loop {
      match self.iter.next() {
        Some(tok) => {
          let ss_count = 
            *self.trainer.sentence_starter_fdist.get(tok.typ()).unwrap_or(&0);
          let typ_count = 
            *self.trainer.type_fdist.get(tok.typ_with_period()).unwrap_or(&0) + 
            *self.trainer.type_fdist.get(tok.typ_without_period()).unwrap_or(&0);

          if typ_count < ss_count { continue; }

          let likelihood = math::col_log_likelihood(
            self.trainer.sentence_break_count as f64,
            typ_count as f64,
            ss_count as f64,
            self.trainer.type_fdist.sum_counts() as f64);

          if likelihood >= self.trainer.parameters.sentence_starter_lower_bound &&
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

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    self.iter.size_hint()
  }
}

/// Iterates over every PuntkToken from the supplied iterator and returns 
/// the immediate following token. Returns None for the following token on the 
/// last token.
struct ConsecutiveTokenIterator<'a, T: 'a, I: Iterator<Item = &'a T>> {
  iter: I,
  last: Option<&'a T>
}

impl<'a, T: 'a, I: Iterator<Item = &'a T>> Iterator
for ConsecutiveTokenIterator<'a, T, I>
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

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    self.iter.size_hint()
  }
}

/// Iterates over every Token from the supplied iterator, and returns the 
/// immediate following token. Returns None for the following token on the last 
/// token.
struct ConsecutiveTokenMutIterator<'a, T: 'a, I: Iterator<Item = &'a mut T>> {
  iter: I,
  last: Option<*mut T>
}

impl<'a, T: 'a, I: Iterator<Item = &'a mut T>> Iterator
for ConsecutiveTokenMutIterator<'a, T, I>
{
  type Item = (&'a mut T, Option<&'a mut T>);

  #[inline]
  fn next(&mut self) -> Option<(&'a mut T, Option<&'a mut T>)> {
    match self.last {
      Some(tok) => {
        self.last = self.iter.next().map(|t| t as *mut T);

        unsafe { Some((&mut *tok, self.last.map(|t| &mut *t))) }
      }
      None => match self.iter.next() {
        Some(tok) => {
          self.last = self.iter.next().map(|t| t as *mut T);

          unsafe { Some((&mut *tok, self.last.map(|t| &mut *t))) }
        }
        None => None
      }
    }
  }

  #[inline]
  fn size_hint(&self) -> (uint, Option<uint>) {
    self.iter.size_hint()
  }
}