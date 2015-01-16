use trainer::TrainingData;
use token::prelude::{WordTokenWithoutPeriod, WordTokenWithFlagsOps};

use phf::Set;

/// Peforms a first pass annotation on a Token.
pub fn annotate_first_pass<F, T: WordTokenWithoutPeriod + WordTokenWithFlagsOps<F>>(
  tok: &mut T,
  data: &TrainingData,
  sent_end: &Set<char>)
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
    if is_split_abbrev || data.contains_abbrev(tok.token_without_period()) {
      tok.set_is_abbrev(true);
    } else {
      tok.set_is_sentence_break(true);
    }
  }
}