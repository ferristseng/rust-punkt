use std::rc::Rc;
use std::cmp::min;
use std::num::Float;
use std::ops::Deref;
use std::hash::Hash;
use std::default::Default;

use phf::Set;
use xxhash::XXState;
use freqdist::{Distribution, FrequencyDistribution};

use token::TrainingToken;
use token::prelude::{
  WordToken,
  WordTypeToken
};

use tokenizer::TrainingWordTokenizer;

use ortho::{
  OrthographyPosition, 
  OrthographicContext,
  BEG_UC, 
  MID_UC, 
  ORTHO_MAP};

/// A collocation. A normal Tuple can not be used, because a collocation
/// as defined by NLTK requires a special hash function. 
#[derive(Show)]
struct Collocation<T> {
  l: T,
  r: T
}

impl<T> Collocation<T> {
  #[inline]
  pub fn new(l: T, r: T) -> Collocation<T> {
    Collocation { l: l, r: r }
  }

  #[inline]
  pub fn left(&self) -> &T {
    &self.l
  }

  #[inline]
  pub fn right(&self) -> &T {
    &self.r
  }
}

impl<T: WordTypeToken, D: Deref<Target = T>> Eq for Collocation<D> { }

impl<T: WordTypeToken, D: Deref<Target = T>> Hash<XXState> for Collocation<D> {
  #[inline]
  fn hash(&self, state: &mut XXState) {
    (*self.l).typ_without_period().hash(state); 
    (*self.r).typ_without_break_or_period().hash(state);
  }
}

impl<T: WordTypeToken, D: Deref<Target = T>> PartialEq for Collocation<D> {
  #[inline]
  fn eq(&self, x: &Collocation<D>) -> bool {
    (*self.l).typ_without_period() == (*x.l).typ_without_period() &&
    (*self.r).typ_without_break_or_period() == (*x.r).typ_without_break_or_period()
  }
}

/// Punctuation within sentence that could indicate an abbreviation
/// if preceded by a period.
/// TODO: Expand this with more unicode definitions.
static INTERNAL_PUNCTUATION: Set<char> = phf_set! { ',', ':', ';', '\u{2014}' };

static DEFAULTS: TrainerParameters = TrainerParameters {
  abbrev_lower_bound: 0.3,
  abbrev_upper_bound: 5f64,
  ignore_abbrev_penalty: false,
  collocation_lower_bound: 7.88,
  sentence_starter_lower_bound: 30f64,
  include_all_collocations: false,
  include_abbrev_collocations: false,
  collocation_frequency_lower_bound: 1f64
};

/// Parameters for the Trainer. These are taken directly from NLTK.
#[derive(Copy, Show, Clone)]
pub struct TrainerParameters {
  pub abbrev_lower_bound: f64,
  pub abbrev_upper_bound: f64,
  pub ignore_abbrev_penalty: bool,
  pub collocation_lower_bound: f64,
  pub sentence_starter_lower_bound: f64,
  pub include_all_collocations: bool,
  pub include_abbrev_collocations: bool,
  pub collocation_frequency_lower_bound: f64
}

impl Default for TrainerParameters {
  /// Creates TrainerParameters based off of Kiss, Tibor, and Strunk's Paper.
  #[inline]
  fn default() -> TrainerParameters {
    DEFAULTS.clone()
  }
}

impl Default for &'static TrainerParameters {
  /// Borrowed pointer to static default.
  #[inline]
  fn default() -> &'static TrainerParameters {
    &DEFAULTS
  }
}

/*
/// Trainer to compile data about frequent sentence staters, collocations, 
/// and potential abbreviations.
///
/// After you've trained on any number of documents, you can call `finalize` 
/// to extract the trained data. You can use and modify preexisting data, if 
/// you instantiate with the `with_data` constructor.
///
/// # Examples
///
/// ```no_run
/// use std::default::Default;
/// use punkt::trainer::Trainer;
/// use punkt::tokenizer::WordTokenizer;
/// use punkt::tokenizer::prelude::Tokenizer; 
///
/// let doc0 = "This is a really long document!";
/// let tokenizer = WordTokenizer::new();
/// let mut data = Default::default();
/// let mut trainer = Trainer::with_data(&mut data);
/// 
/// trainer.train_tokens(tokenizer.tokenize_document(doc0));
///
/// trainer.finalize();
/// ```
pub struct Trainer<'a> {
  /// Number of periods counted in tokens encountered.
  period_token_count: uint,

  /// Number of sentence breaks in tokens encountered.
  sentence_break_count: uint,

  /// The training data. This data object is modified as new tokens 
  /// are trained on.
  data: &'a mut Data,

  /// Trainer parameters.
  parameters: &'a TrainerParameters,

  /// A list of all tokens encountered. Other fields reference Tokens
  /// from here.
  tokens: Vec<Rc<Token>>,

  /// A frequency distribution of all Tokens encountered.
  type_fdist: FrequencyDistribution<Rc<Token>>,

  /// A frequency distribution of all collocations encountered.
  collocation_fdist: FrequencyDistribution<TrainerCollocation>,

  /// A frequency distribution of all sentence starters encountered.
  sentence_starter_fdist: FrequencyDistribution<Rc<Token>>
}

impl<'a> Trainer<'a> {
  #[inline]
  pub fn with_data(data: &'a mut Data) -> Trainer<'a> {
    Trainer::with_data_and_parameters(data, Default::default())
  }

  #[inline]
  pub fn with_data_and_parameters(
    data: &'a mut Data, 
    params: &'a TrainerParameters
  ) -> Trainer<'a> {
    Trainer {
      period_token_count: 0,
      sentence_break_count: 0,
      data: data,
      parameters: params,
      tokens: Vec::new(),
      type_fdist: FrequencyDistribution::new(),
      collocation_fdist: FrequencyDistribution::new(),
      sentence_starter_fdist: FrequencyDistribution::new()
    }
  }

  /// This isn't entirely safe, so should be used with extreme caution. It
  /// returns a mutable reference to data on the Trainer. Mostly this is 
  /// for the scenario when `data` needs to be modified while iterating, and 
  /// self needs to be borrowed. It can be reasoned generally that iterating over
  /// a part of `self` that isn't `data` makes it safe to modify `data` in most cases.
  #[inline]
  unsafe fn borrow_data_mut_unsafe(&self) -> &mut Data {
    &mut *(&*self.data as *const Data as *mut Data)
  }

  /// Train on a series of Tokens. 
  pub fn train_tokens(&mut self, tokens: Vec<Token>) {
    let start = self.tokens.len();

    self.tokens.reserve(tokens.len());

    for t in tokens.into_iter() { self.tokens.push(Rc::new(t)); }

    let slice = self.tokens.slice_from(start);

    for t in slice.iter() {
      self.type_fdist.insert(t.clone());

      if t.has_final_period() { self.period_token_count += 1 }
    }

    // Reclassify the abbreviation types.
    for (t, score) in reclassify_iter(self, self.type_fdist.keys()) {
      if score >= self.parameters.abbrev_lower_bound { 
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

    // Mark abbreviation types if any exist.
    for t in slice.iter() {
      // Rc doesn't provide a mutable interface into a Token by default. 
      // We have to coerce the Token into being mutable.
      unsafe {
        self.annotate_first_pass(
          &mut *(t.deref() as *const Token as *mut Token)); 
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

  /// Empties the trained data, and compiles it with Data. 
  /// Returns the compiled Data.
  pub fn finalize(&mut self) {
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

impl<'a> PunktFirstPassAnnotater for Trainer<'a> {
  #[inline]
  fn data(&self) -> &Data {
    &*self.data
  }
}

#[inline]
fn reclassify_iter<'a, 'b, I: Iterator<&'a Rc<Token>>>(
  trainer: &'b Trainer<'b>,
  iter: I
) -> PunktReclassifyIterator<'a, 'b, I> {
  PunktReclassifyIterator { iter: iter, trainer: trainer }
}

#[inline]
fn orthography_iter<'a, I: Iterator<&'a Rc<Token>>>(
  iter: I
) -> TokenWithContextIterator<'a, I> {
  TokenWithContextIterator { iter: iter, ctxt: OrthographyPosition::Internal }
}

#[inline]
fn potential_sentence_starter_iter<'a, 'b, I: Iterator<&'a Rc<Token>>>(
  trainer: &'b Trainer,
  iter: I
) -> PotentialSentenceStartersIterator<'a, 'b, I> {
  PotentialSentenceStartersIterator { iter: iter, trainer: trainer }
}

#[inline]
fn potential_collocation_iter<'a, 'b, I: Iterator<&'a TrainerCollocation>>(
  trainer: &'b Trainer,
  iter: I
) -> PotentialCollocationsIterator<'a, 'b, I> {
  PotentialCollocationsIterator { iter: iter, trainer: trainer }
}

/// From the original paper.
fn dunning_log_likelihood(count_a: f64, count_b: f64, count_ab: f64, n: f64) -> f64 {
  let p1 = count_b / n;
  let p2 = 0.99;
  let nullh = count_ab * p1.ln() + (count_a - count_ab) * (1.0 - p1).ln();
  let alth = count_ab * p2.ln() + (count_a - count_ab) * (1.0 - p2).ln();

  -2.0 * (nullh - alth)
}

/// From the original paper.
fn col_log_likelihood(count_a: f64, count_b: f64, count_ab: f64, n: f64) -> f64 {
  let p = count_b / n;
  let p1 = count_ab / count_a;
  let p2 = (count_b - count_ab) / (n - count_a);

  let s1 = count_ab * p.ln() + (count_a - count_ab) * (1.0 - p).ln();
  let s2 = (count_b - count_ab) * p.ln() + (n - count_a - count_b + count_ab) * (1.0 - p).ln();
  let s3 = if count_a == count_ab {
    0f64
  } else {
    count_ab * p1.ln() + (count_a - count_ab) * (1.0 - p1).ln()
  };
  let s4 = if count_b == count_ab {
    0f64
  } else {
    (count_b - count_ab) * p2.ln() + (n - count_a - count_b + count_ab) * (1.0 - p2).ln()
  };

  -2.0 * (s1 + s2 - s3 - s4)
}

fn is_rare_abbrev_type(
  trainer: &Trainer,
  tok0: &Token, 
  tok1: &Token
) -> bool {
  if tok0.is_abbrev() || !tok0.is_sentence_break() {
    // Check the first condition, and return if it matches
    false
  } else {
    let key = tok0.typ_without_break_or_period();

    // Count all variations of the token
    let count = 
      *trainer.type_fdist.get(key).unwrap_or(&0) + 
      *trainer.type_fdist.get(key.slice_to(key.len() - 1)).unwrap_or(&0);

    if trainer.data.contains_abbrev(tok0.typ()) || 
      (count as f64) >= trainer.parameters.abbrev_upper_bound 
    {
      // Check the second condition. Return if it's true...the token is 
      // already an abbreviation!
      false
    } else if INTERNAL_PUNCTUATION.contains(&tok1.typ().char_at(0)) {
      // Check the first case of the final condition
      true
    } else if tok1.is_lowercase() {
      let ctxt = *trainer
        .data()
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
  cur: &Token, 
  prev: &Token
) -> bool {
  prev.is_sentence_break() && 
  !(prev.is_numeric() || prev.is_initial()) && 
  cur.is_alphabetic()
}

#[inline]
fn is_potential_collocation(
  trainer: &Trainer,
  tok0: &Token,
  tok1: &Token
) -> bool {
  (trainer.parameters.include_all_collocations ||
  (trainer.parameters.include_abbrev_collocations && tok0.is_abbrev()) ||
  (tok0.is_sentence_break() && (tok0.is_numeric() || tok0.is_initial())))
  && tok0.is_non_punct()
  && tok1.is_non_punct()
}

/// A token and its associated score (likelihood of it being a abbreviation).
type ScoredToken<'a> = (&'a Token, f64);

/// Iterates over every Token from the supplied iterator. Only returns 
/// the ones that are 'not obviously' abbreviations. Also returns the associated 
/// score of that token.
struct PunktReclassifyIterator<'a: 'b, 'b, I: Iterator<&'a Rc<Token>>> {
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'a, 'b, I: Iterator<&'a Rc<Token>>> Iterator<ScoredToken<'a>> 
for PunktReclassifyIterator<'a, 'b, I> {
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
            if self.trainer.data().contains_abbrev(t.typ()) {
              continue;
            }
          } else {
            if !self.trainer.data().contains_abbrev(t.typ()) {
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

          let likelihood = dunning_log_likelihood(
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
type TokenWithContext<'a> = (&'a Token, OrthographicContext);

/// Iterates over every Token from the supplied iterator and returns its
/// decided orthography within the given text. 
struct TokenWithContextIterator<'a, I: Iterator<&'a Rc<Token>>> {
  iter: I,
  ctxt: OrthographyPosition
}

impl<'a, I: Iterator<&'a Rc<Token>>> Iterator<TokenWithContext<'a>>
for TokenWithContextIterator<'a, I>
{
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
struct PotentialCollocationsIterator<'a, 'b, I: Iterator<&'a TrainerCollocation>> 
{
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'a, 'b, I: Iterator<&'a TrainerCollocation>> Iterator<(&'a TrainerCollocation, f64)>
for PotentialCollocationsIterator<'a, 'b, I> {
  #[inline]
  fn next(&mut self) -> Option<(&'a TrainerCollocation, f64)> {
    loop {
      match self.iter.next() {
        Some(col) => {
          if self.trainer.data().contains_sentence_starter(
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
            let likelihood = col_log_likelihood(
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

struct PotentialSentenceStartersIterator<'a, 'b, I: Iterator<&'a Rc<Token>>>
{
  iter: I,
  trainer: &'b Trainer<'b>
}

impl<'a, 'b, I: Iterator<&'a Rc<Token>>> Iterator<ScoredToken<'a>>
for PotentialSentenceStartersIterator<'a, 'b, I> {
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

          let likelihood = col_log_likelihood(
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
*/
