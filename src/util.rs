// Copyright 2016 rust-punkt developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use token::Token;
use trainer::TrainingData;
use prelude::DefinesSentenceEndings;

use num::Float;

/// Peforms a first pass annotation on a Token.
pub fn annotate_first_pass<P: DefinesSentenceEndings>(tok: &mut Token, data: &TrainingData) {
  let is_split_abbrev = tok
    .tok()
    .rsplitn(1, '-')
    .next()
    .map(|s| data.contains_abbrev(s))
    .unwrap_or(false);

  if tok.tok().len() == 1 && P::is_sentence_ending(&tok.tok().chars().nth(0).unwrap()) {
    tok.set_is_sentence_break(true);
  } else if tok.has_final_period() && !tok.is_ellipsis() {
    if is_split_abbrev || data.contains_abbrev(tok.tok_without_period()) {
      tok.set_is_abbrev(true);
    } else {
      tok.set_is_sentence_break(true);
    }
  }
}

pub fn dunning_log_likelihood(count_a: f64, count_b: f64, count_ab: f64, n: f64) -> f64 {
  let p1 = count_b / n;
  let p2 = 0.99;
  let nullh = count_ab * p1.ln() + (count_a - count_ab) * (1.0 - p1).ln();
  let alth = count_ab * p2.ln() + (count_a - count_ab) * (1.0 - p2).ln();

  -2.0 * (nullh - alth)
}

pub fn col_log_likelihood(count_a: f64, count_b: f64, count_ab: f64, n: f64) -> f64 {
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
