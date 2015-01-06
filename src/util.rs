use trainer::TrainingData;
use token::prelude::{WordToken, WordTokenWithFlagsOps};

use phf::Set;

/// Peforms a first pass annotation on a Token.
pub fn annotate_first_pass<F, T: WordToken + WordTokenWithFlagsOps<F>>(
  sent_end: &Set<char>,
  data: &TrainingData,
  tok: &mut T)
{
  let is_split_abbrev = tok
    .token()
    .rsplitn(1, '-')
    .next()
    .map(|s| data.contains_abbrev(s))
    .unwrap_or(false);

  if tok.token().len() == 1 && 
     sent_end.contains(&tok.token().char_at(0)) 
  { 
    tok.set_is_sentence_break(true);
  } else if tok.has_final_period() && !tok.is_ellipsis() {
    if is_split_abbrev || data.contains_abbrev(tok.token()) {
      tok.set_is_abbrev(true);
    } else {
      tok.set_is_sentence_break(true);
    }
  }
}