pub use tokenizer::word::{WordTokenizer, WordTokenizerParameters};
pub use tokenizer::sentence::{SentenceTokenizer, SentenceTokenizerParameters};

#[cfg(test)] use std::fs;
#[cfg(test)] use std::fs::PathExt;
#[cfg(test)] use std::path::Path;
#[cfg(test)] use std::io::Read;

mod word;
mod sentence;
mod periodctxt;

#[cfg(test)]
fn get_test_scenarios(
  dir_path: &str, 
  raw_path: &str
) -> Vec<(Vec<String>, String, String)> {
  let mut tests = Vec::new();

  for path in fs::walk_dir(&Path::new(dir_path)).unwrap() {
    let fpath = path.unwrap().path();

    if fpath.is_file() {
      let mut exp_strb = String::new();
      let mut raw_strb = String::new();

      // Files in the directory with raw articles must match the file names of 
      // articles in the directory with test outcomes.
      let rawp = Path::new(raw_path).join(fpath.file_name().unwrap());

      fs::File::open(&fpath).unwrap().read_to_string(&mut exp_strb);
      fs::File::open(&rawp).unwrap().read_to_string(&mut raw_strb);

      // Expected results, split by newlines.
      let exps: Vec<String> = exp_strb.split('\n').map(|s| s.to_string()).collect();

      tests.push((exps, raw_strb, format!("{:?}", fpath.file_name().unwrap())));
    }
  }

  tests // Returns (Expected cases, File contents, File name)
}
