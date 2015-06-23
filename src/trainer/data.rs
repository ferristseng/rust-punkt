use std::ops::Deref;
use std::hash::{Hasher, Hash};
use std::str::FromStr;
use std::default::Default;
use std::collections::{HashSet, HashMap};

use self::DataString::*;
use prelude::OrthographicContext;

use rustc_serialize::json::Json;


/// Internal object to store a string. Gives the ability to query
/// for an owned string within a `HashMap` or `HashSet` using a 
/// borrowed string.
#[derive(Debug, Eq)] enum DataString<'a> {
  Owned(String),
  Borrowed(&'a str)
}

impl<'a> Deref for DataString<'a> {
  type Target = str;

  #[inline(always)] fn deref(&self) -> &str {
    match self {
      &Owned(ref s) => &s[..],
      &Borrowed(ref s) => s
    }
  }
}

impl<'a> PartialEq for DataString<'a> {
  #[inline(always)] fn eq(&self, other: &DataString) -> bool {
    let s: &str = &self; s == other.deref()
  }
}

impl<'a> Hash for DataString<'a> {
  #[inline(always)] fn hash<H>(&self, state: &mut H) where H : Hasher {
    let s: &str = &self; s.hash(state);
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
  /// Creates a new, empty data object.
  #[inline(always)] pub fn new() -> TrainingData<'a> {
    TrainingData { ..Default::default() }
  }

  /// Check if a token is considered to be an abbreviation.
  #[inline(always)] pub fn contains_abbrev(&self, tok: &str) -> bool {
    self.abbrevs.contains(&Borrowed(tok))
  }

  /// Insert a newly learned abbreviation.
  #[inline] pub fn insert_abbrev(&mut self, tok: &str) -> bool {
    if !self.contains_abbrev(tok) {
      self.abbrevs.insert(Owned(tok.to_lowercase()))
    } else {
      false
    }
  }

  /// Removes a learned abbreviation.
  #[inline] pub fn remove_abbrev(&mut self, tok: &'a str) -> bool {
    self.abbrevs.remove(&Borrowed(tok))
  }

  /// Check if a token is considered to be a token that commonly starts a 
  /// sentence.
  #[inline(always)] pub fn contains_sentence_starter(&self, tok: &str) -> bool {
    self.sentence_starters.contains(&Borrowed(tok))
  }

  /// Insert a newly learned word that signifies the start of a sentence.
  #[inline] pub fn insert_sentence_starter(&mut self, tok: &str) -> bool {
    if !self.contains_sentence_starter(tok) {
      self.sentence_starters.insert(Owned(tok.to_string()))
    } else {
      false
    }
  }

  /// Checks if a pair of words are commonly known to appear together.
  #[inline] pub fn contains_collocation(&self, left: &str, right: &str) -> bool {
    self
      .collocations
      .get(&Borrowed(left))
      .map(|s| s.contains(&Borrowed(right)))
      .unwrap_or(false)
  }

  /// Insert a newly learned pair of words that frequently appear together.
  pub fn insert_collocation(&mut self, left: &'a str, right: &str) -> bool {
    if !self.collocations.contains_key(&Borrowed(left)) {
      self.collocations.insert(
        Owned(left.to_string()), 
        HashSet::new());
    }

    if !self
      .collocations
      .get(&Borrowed(left))
      .unwrap()
      .contains(&Borrowed(right))
    {
      self
        .collocations
        .get_mut(&Borrowed(left))
        .unwrap()
        .insert(Owned(right.to_string()));
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
    // `get_mut` isn't allowed here, without adding an unnecessary lifetime
    // qualifier to `tok`. 
    match self.orthographic_context.get(&Borrowed(tok)) {
      Some(c) => unsafe { 
        *(c as *const OrthographicContext as *mut OrthographicContext) |= ctxt; 
        return false 
      },
      None => () 
    }

    self
      .orthographic_context
      .insert(Owned(tok.to_string()), ctxt);

    true
  }

  /// Gets the orthographic context for a token. Returns 0 if the token 
  /// was not yet encountered.
  #[inline(always)] pub fn get_orthographic_context(&self, tok: &str) -> u8 {
    *self.orthographic_context.get(&Borrowed(tok)).unwrap_or(&0)
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
                .entry(Owned(l))
                .or_insert(HashSet::new())
                .insert(Owned(r)),
            _ => return Err("failed to parse collocations section")
          };
        });

        match obj.remove("ortho_context") {
          Some(Json::Object(obj)) => {
            for (k, ctxt) in obj.into_iter() {
              ctxt
                .as_u64()
                .map(|c| data.orthographic_context.insert(Owned(k), c as u8)); 
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