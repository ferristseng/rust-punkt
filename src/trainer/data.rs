use std::hash::{Hasher, Hash};
use std::str::FromStr;
use std::default::Default;
use std::collections::{HashSet, HashMap};

use self::DataString::*;
use prelude::OrthographicContext;

use rustc_serialize::json::Json;


/// Internal object to store a string. Strings that are owned have 
/// already been normalized. In order to check if a new string needs to 
/// be allocated, case-insensitive hash and equality methods need to be 
/// used.
#[derive(Debug, Eq)] enum DataString<'a> {
  NormalizedOwned(String),
  UnnormalizedBorrowed(&'a str)
}

impl<'a> PartialEq for DataString<'a> {
  #[inline(always)] fn eq(&self, other: &DataString) -> bool {
    fn normal_unnormal_eq(norm: &str, raw: &str) -> bool {
      raw
        .chars()
        .flat_map(|c| c.to_lowercase())
        .zip(norm.chars())
        .fold(true, |acc, (c, c0)| acc && c == c0)
    }

    match self {
      &NormalizedOwned(ref s) => match other {
        &NormalizedOwned(ref s0) => s == s0,
        &UnnormalizedBorrowed(ref s0) => normal_unnormal_eq(&s[..], s0)
      },
      &UnnormalizedBorrowed(ref s) => match other {
        &NormalizedOwned(ref s0) => normal_unnormal_eq(&s0[..], s),
        &UnnormalizedBorrowed(ref s0) => s0
          .chars()
          .flat_map(|c| c.to_lowercase())
          .zip(s.chars().flat_map(|c| c.to_lowercase()))
          .fold(true, |acc, (c, c0)| acc && c == c0)
      }
    }
  }
}

impl<'a> Hash for DataString<'a> {
  fn hash<H>(&self, state: &mut H) where H : Hasher {
    match self {
      &NormalizedOwned(ref s) => {
        for c in s.chars() { state.write_u32(c as u32) }
      }
      &UnnormalizedBorrowed(ref s) => {
        for b in s.chars().flat_map(|c| c.to_lowercase()) { state.write_u32(b as u32) }
      }
    }
  }
}


/// Stores data that was obtained during training. 
///
/// # Examples
///
/// Precompiled data can be loaded via a language specific constructor.
///
/// ```
/// use punkt::TrainingData;
///
/// let eng_data = TrainingData::english();
/// let ger_data = TrainingData::german();
///
/// assert!(eng_data.contains_abbrev("va"));
/// assert!(ger_data.contains_abbrev("crz"));
/// ``` 
#[derive(Debug, Default)] pub struct TrainingData<'a> {
  abbrevs: HashSet<DataString<'a>>,
  collocations: HashMap<DataString<'a>, HashSet<DataString<'a>>>,
  sentence_starters: HashSet<DataString<'a>>,
  orthographic_context: HashMap<DataString<'a>, OrthographicContext>
}

impl<'a> TrainingData<'a> {
  /// Check if a token is considered to be an abbreviation.
  #[inline(always)] pub fn contains_abbrev(&self, tok: &str) -> bool {
    self.abbrevs.contains(&UnnormalizedBorrowed(tok))
  }

  /// Insert a newly learned abbreviation.
  #[inline] pub fn insert_abbrev(&mut self, tok: &str) -> bool {
    if !self.contains_abbrev(tok) {
      self.abbrevs.insert(NormalizedOwned(tok.to_lowercase()))
    } else {
      false
    }
  }

  /// Removes a learned abbreviation.
  #[inline] pub fn remove_abbrev(&mut self, tok: &'a str) -> bool {
    self.abbrevs.remove(&UnnormalizedBorrowed(tok))
  }

  /// Check if a token is considered to be a token that commonly starts a 
  /// sentence.
  #[inline(always)] pub fn contains_sentence_starter(&self, tok: &str) -> bool {
    self.sentence_starters.contains(&UnnormalizedBorrowed(tok))
  }

  /// Insert a newly learned word that signifies the start of a sentence.
  #[inline] pub fn insert_sentence_starter(&mut self, tok: &str) -> bool {
    if !self.contains_sentence_starter(tok) {
      self.sentence_starters.insert(NormalizedOwned(tok.to_lowercase()))
    } else {
      false
    }
  }

  /// Checks if a pair of words are commonly known to appear together.
  #[inline] pub fn contains_collocation(&self, left: &str, right: &str) -> bool {
    self
      .collocations
      .get(&UnnormalizedBorrowed(left))
      .map(|s| s.contains(&UnnormalizedBorrowed(right)))
      .unwrap_or(false)
  }

  /// Insert a newly learned pair of words that frequently appear together.
  pub fn insert_collocation(&mut self, left: &'a str, right: &str) -> bool {
    if !self.collocations.contains_key(&UnnormalizedBorrowed(left)) {
      self.collocations.insert(
        NormalizedOwned(left.to_lowercase()), 
        HashSet::new());
    }

    if !self
      .collocations
      .get(&UnnormalizedBorrowed(left))
      .unwrap()
      .contains(&UnnormalizedBorrowed(right))
    {
      self
        .collocations
        .get_mut(&UnnormalizedBorrowed(left))
        .unwrap()
        .insert(NormalizedOwned(right.to_lowercase()));
      true
    } else {
      false
    }
  }

  /// Insert or update the known orthographic context that a word commonly 
  /// appears in.
  #[inline] pub fn insert_orthographic_context(
    &mut self, 
    tok: &str, 
    ctxt: OrthographicContext
  ) -> bool {
    let valu = {
      self.orthographic_context.get_mut(&UnnormalizedBorrowed(tok)).unwrap()
    };

    // `get_mut` isn't allowed here, without adding an unnecessary lifetime
    // qualifier to `tok`. This might be fixed by changing how 
    // `UnnormalizedBorrow` works.
    match self.orthographic_context.get(&UnnormalizedBorrowed(tok)) {
      Some(c) => unsafe { 
        *(c as *const OrthographicContext as *mut OrthographicContext) |= ctxt; 
        return false 
      },
      None => () 
    }

    self
      .orthographic_context
      .insert(NormalizedOwned(tok.to_lowercase()), ctxt);

    true
  }

  /// Gets the orthographic context for a token. Returns 0 if the token 
  /// was not yet encountered.
  #[inline] pub fn get_orthographic_context(&self, tok: &str) -> u8 {
    *self.orthographic_context.get(&UnnormalizedBorrowed(tok)).unwrap_or(&0)
  }
}

impl<'a> FromStr for TrainingData<'a> {
  type Err = &'static str; 

  /// Deserializes JSON and loads the data into a new TrainingData object.
  fn from_str(s: &str) -> Result<TrainingData<'a>, &'static str> { 
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

        read_json_array_data!(
          "abbrev_types", Json::String(st), data.insert_abbrev(&st[..]));

        read_json_array_data!(
          "sentence_starters", Json::String(st), data.insert_sentence_starter(&st[..]));

        // Load collocations, these come as an array with 2 members in them (or they should). 
        // Pop them in reverse order, then insert into the proper bucket. 
        read_json_array_data!("collocations", Json::Array(mut ar), {
          match (ar.pop(), ar.pop()) {
            (Some(Json::String(r)), Some(Json::String(l))) => 
              data
                .collocations
                .entry(NormalizedOwned(l))
                .or_insert(HashSet::new())
                .insert(NormalizedOwned(r)),
            _ => return Err("failed to parse collocations section")
          };
        });

        match obj.remove("ortho_context") {
          Some(Json::Object(obj)) => {
            for (k, ctxt) in obj.into_iter() {
              ctxt
                .as_u64()
                .map(|c| data.orthographic_context.insert(NormalizedOwned(k), c as u8)); 
            }
          }
          _ => return Err("failed to parse orthographic context section") 
        }

        Ok(data)
      }
      _ => Err("no json object found containing training data") 
    }
  }
}


// Macro for generating functions to load precompiled data.
macro_rules! preloaded_data(
  ($lang:ident, $file:expr) => (
    impl<'a> TrainingData<'a> {
      /// Default data for $lang.
      #[inline] pub fn $lang() -> TrainingData<'a> {
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


#[test] fn test_data_load_from_json_test() {
  let data: TrainingData = TrainingData::english();

  assert!(data.orthographic_context.len() > 0);
  assert!(data.abbrevs.len() > 0);
  assert!(data.sentence_starters.len() > 0);
  assert!(data.collocations.len() > 0);
  assert!(data.contains_sentence_starter("among"));
  assert!(data.contains_abbrev("w.va"));
  assert!(data.contains_collocation("##number##", "corrections"));
}


#[test] fn test_data_string_hash_eq() {
  let mut set = HashSet::new();

  set.insert(NormalizedOwned("abc".to_string()));
  set.insert(NormalizedOwned("àáâãä".to_string()));

  assert!(set.contains(&UnnormalizedBorrowed("AbC")));
  assert!(set.contains(&UnnormalizedBorrowed("ÀÁÂÃÄ")));
}