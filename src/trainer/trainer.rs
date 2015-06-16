use std::cmp::min;
use std::ops::Deref;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use num::Float;
use freqdist::FrequencyDistribution;
use freqdist::Distribution;

use util;
use token::Token;
use tokenizer::WordTokenizer;
use trainer::data::TrainingData;
use prelude::{TrainerParameters, DefinesNonPrefixCharacters, DefinesNonWordCharacters,
  OrthographicContext, OrthographyPosition};


/// A collocation is any pair of words that has a high likelihood of appearing
/// together.
#[derive(Debug, Eq)] pub struct Collocation<T> where T : Deref<Target = Token> { 
  l: T, 
  r: T 
}

impl<T> Collocation<T> where T : Deref<Target = Token> {
  #[inline(always)] pub fn new(l: T, r: T) -> Collocation<T> { 
    Collocation { l: l, r: r } 
  }

  #[inline(always)] pub fn left(&self) -> &T { &self.l }
  #[inline(always)] pub fn right(&self) -> &T { &self.r }
}

impl<T> Hash for Collocation<T> where T : Deref<Target = Token> {
  #[inline(always)] fn hash<H>(&self, state: &mut H) where H : Hasher {
    (*self.l).typ_without_period().hash(state); 
    (*self.r).typ_without_break_or_period().hash(state);
  }
}

impl<T> PartialEq for Collocation<T> where T : Deref<Target = Token>  {
  #[inline(always)] fn eq(&self, x: &Collocation<T>) -> bool {
    (*self.l).typ_without_period() == (*x.l).typ_without_period() &&
    (*self.r).typ_without_break_or_period() == (*x.r).typ_without_break_or_period()
  }
}


/// A trainer will build data about abbreviations, sentence starters, 
/// collocations, and context that tokens appear in. The data is 
/// used by the sentence tokenizer to determine if a period is likely 
/// part of an abbreviation, or actually marks the termination of a sentence.
pub struct Trainer<P> { params: PhantomData<P> }

impl<P> Trainer<P> 
  where P : TrainerParameters + DefinesNonPrefixCharacters + DefinesNonWordCharacters 
{
  /// Creates a new Trainer.
  #[inline(always)] pub fn new() -> Trainer<P> { Trainer { params: PhantomData } }

  /// Train on a document. Does tokenization using a WordTokenizer.
  pub fn train(self, doc: &str, data: &mut TrainingData) {
    let mut period_token_count: usize = 0;
    let mut sentence_break_count: usize = 0;
    let tokens: Vec<Token> = WordTokenizer::<P>::new(doc).collect();
    let mut type_fdist: FrequencyDistribution<&str> = FrequencyDistribution::new();
    //let mut collocation_fdist = FrequencyDistribution::new();
    //let mut sentence_starter_fdist = FrequencyDistribution::new();

    for t in tokens.iter() {
      if t.has_final_period() { period_token_count += 1 }
      type_fdist.insert(t.typ());
    }
    
    // Iterate through to see if any tokens need to be reclassified as an 
    // abbreviation or removed as an abbreviation.
    {
      let reclassify_iter: ReclassifyIterator<::std::slice::Iter<Token>, P> = 
        ReclassifyIterator {
          iter: tokens.iter(),
          data: data,
          period_token_count: period_token_count,
          type_fdist: &mut type_fdist,
          params: PhantomData
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

    /*
    for (t, ctxt) in orthography_iter(slice.iter()) {
      if ctxt != 0 {
        self.data.insert_orthographic_context(t.typ_without_break_or_period(), ctxt);
      }
    }
    */

    // Order matters! Sentence break checks are dependent on whether or not 
    // the token is an abbreviation. Must come after the first pass annotation!
    for t in tokens.iter() {
      if t.is_sentence_break() { sentence_break_count += 1; }
    }

    /*
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
      */
    }
  }

/*
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
*/


fn is_rare_abbrev_type<P>(
  data: &TrainingData,
  type_fdist: &FrequencyDistribution<&str>, 
  tok0: &Token, 
  tok1: &Token
) -> bool where P : TrainerParameters {
  use prelude::{BEG_UC, MID_UC};

  if tok0.is_abbrev() || !tok0.is_sentence_break() {
    false
  } else {
    let key = tok0.typ_without_break_or_period();
    let count = (type_fdist[key] + type_fdist[&key[..key.len() - 1]]) as f64;

    // Already an abbreviation...
    if data.contains_abbrev(tok0.typ()) || count >= P::abbrev_upper_bound() {
      false
    } else if P::is_internal_punctuation(&tok1.typ().char_at(0)) {
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


#[inline(always)] fn is_potential_sentence_starter(
  cur: &Token, 
  prev: &Token
) -> bool {
  prev.is_sentence_break() && !(prev.is_numeric() || prev.is_initial()) && 
  cur.is_alphabetic()
}


#[inline(always)] fn is_potential_collocation<P>(
  tok0: &Token,
  tok1: &Token
) -> bool where P : TrainerParameters {
  P::include_all_collocations() ||
  (P::include_abbrev_collocations() && tok0.is_abbrev()) ||
  (tok0.is_sentence_break() && (tok0.is_numeric() || tok0.is_initial())) &&
  tok0.is_non_punct() && tok1.is_non_punct()
}



/// Iterates over every token from the supplied iterator. Only returns 
/// the ones that are 'not obviously' abbreviations. Also returns the associated 
/// score of that token.
struct ReclassifyIterator<'b, I, P> {
  iter: I,
  data: &'b TrainingData<'b>,
  period_token_count: usize,
  type_fdist: &'b FrequencyDistribution<&'b str>,
  params: PhantomData<P>
}

impl<'b, I, P> Iterator for ReclassifyIterator<'b, I, P> 
  where I : Iterator<Item = &'b Token>, P : TrainerParameters 
{
  type Item = (&'b Token, f64);

  #[inline] fn next(&mut self) -> Option<Self::Item> {
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

      let num_periods = t
        .typ_without_period()
        .chars()
        .fold(0, |acc, c| if c == '.' { acc + 1 } else { acc }) + 1;
      let num_nonperiods = t.typ_without_period().chars().count() - num_periods + 1;

      let count_with_period = self.type_fdist.get(t.typ_with_period());
      let count_without_period = self.type_fdist.get(t.typ_without_period());

      let likelihood = util::dunning_log_likelihood(
        (count_with_period + count_without_period) as f64,
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

      return Some((t, score))
    }

    None
  }
}


/// Iterates over every token from the supplied iterator and returns its
/// decided orthography within the given text. 
struct TokenWithContextIterator<I> {
  iter: I,
  ctxt: OrthographyPosition
}

impl<'a, I> Iterator for TokenWithContextIterator<I> 
  where I: Iterator<Item = &'a Token>
{
  type Item = (&'a Token, OrthographicContext); 

  /// Returns tokens annotated with their OrthographicContext. Must keep track 
  /// and modify internal position of where previous tokens were.
  #[inline] fn next(&mut self) -> Option<(&'a Token, OrthographicContext)> {
    match self.iter.next() {
      Some(t) => {
        if t.is_paragraph_start() && self.ctxt != OrthographyPosition::Unknown {
          self.ctxt = OrthographyPosition::Initial;
        }

        if t.is_newline_start() && self.ctxt == OrthographyPosition::Internal {
          self.ctxt = OrthographyPosition::Unknown;
        }

        let flag = *::prelude::ORTHO_MAP
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


/*
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
*/
