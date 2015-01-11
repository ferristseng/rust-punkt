use std::rc::Rc;
use std::ops::Deref;
use std::default::Default;

use phf::Set;
use freqdist::{Distribution, FrequencyDistribution};

use token::TrainingToken;
use token::prelude::{
  WordToken,
  WordTypeToken,
  WordTokenWithFlagsOps,
  WordTokenWithFlagsOpsExt,
};

use util;
use trainer::iter;
use trainer::col::Collocation;
use trainer::data::TrainingData;
use tokenizer::WordTokenizer;
use ortho::{BEG_UC, MID_UC};

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
  pub period_token_count: usize,

  /// Number of sentence breaks in tokens encountered.
  pub sentence_break_count: usize,

  /// The training data. This data object is modified as new tokens 
  /// are trained on.
  pub data: &'a mut TrainingData,

  /// Trainer parameters.
  pub parameters: &'a TrainerParameters,

  /// A list of all tokens encountered. Other fields reference Tokens
  /// from here.
  tokens: Vec<Rc<TrainingToken>>,

  /// A frequency distribution of all Tokens encountered.
  pub type_fdist: FrequencyDistribution<Rc<TrainingToken>>,

  /// A frequency distribution of all collocations encountered.
  pub collocation_fdist: FrequencyDistribution<Collocation<Rc<TrainingToken>>>,

  /// A frequency distribution of all sentence starters encountered.
  pub sentence_starter_fdist: FrequencyDistribution<Rc<TrainingToken>>
}

impl<'a> Trainer<'a> {
  #[inline]
  pub fn with_data(data: &'a mut TrainingData) -> Trainer<'a> {
    Trainer::with_data_and_parameters(data, Default::default())
  }

  #[inline]
  pub fn with_data_and_parameters(
    data: &'a mut TrainingData, 
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
  unsafe fn borrow_data_mut_unsafe(&self) -> &mut TrainingData {
    &mut *(&*self.data as *const TrainingData as *mut TrainingData)
  }

  /// Train on a series of Tokens. 
  pub fn train_tokens(&mut self, tokens: Vec<TrainingToken>) {
    let start = self.tokens.len();

    self.tokens.reserve(tokens.len());

    for t in tokens.into_iter() { self.tokens.push(Rc::new(t)); }

    let slice = self.tokens.slice_from(start);

    for t in slice.iter() {
      self.type_fdist.insert(t.clone());

      if t.has_final_period() { self.period_token_count += 1 }
    }

    // Reclassify the abbreviation types.
    for (t, score) in iter::reclassify_iter(self, self.type_fdist.keys()) {
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
        util::annotate_first_pass(
          &mut *(t.deref() as *const TrainingToken as *mut TrainingToken),
          self.data,
          &phf_set![]); 
      }
    }

    for (t, ctxt) in iter::orthography_iter(slice.iter()) {
      if ctxt != 0 {
        self.data.insert_orthographic_context(t.typ_without_break_or_period(), ctxt);
      }
    }

    // Order matters! Sentence break checks are dependent on whether or not 
    // the token is an abbreviation. Must come after the first pass annotation!
    for t in slice.iter() {
      if t.is_sentence_break() { self.sentence_break_count += 1; }
    }

    for (lt, rt) in iter::consecutive_token_iter(slice.iter()) {
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
    in iter::potential_sentence_starter_iter(
      self, 
      self.sentence_starter_fdist.keys())
    {
      unsafe {
        self.borrow_data_mut_unsafe().insert_sentence_starter(tok.typ());
      }
    }

    self.data.clear_collocations();

    for (col, _)
    in iter::potential_collocation_iter(
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
  (trainer.parameters.include_all_collocations ||
  (trainer.parameters.include_abbrev_collocations && tok0.is_abbrev()) ||
  (tok0.is_sentence_break() && (tok0.is_numeric() || tok0.is_initial())))
  && tok0.is_non_punct()
  && tok1.is_non_punct()
}