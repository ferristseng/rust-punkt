extern crate punkt;

use std::default::Default;

use punkt::trainer::Trainer;

fn main() {
  let mut data = Default::default(); 
  
  {
    let mut trainer = Trainer::new(&mut data);

    trainer.train(include_str!("../test/raw/npr-article-01.txt"));
    trainer.train(include_str!("../test/raw/ny-times-article-01.txt"));
    trainer.finalize();
  }

  println!("==Abbreviations==");

  for ab in data.abbrevs_iter() {
    println!("{:?}", ab);
  }

  println!("==Sentence Starters==");

  for ss in data.sentence_starters_iter() {
    println!("{:?}", ss);
  }

  println!("==Collocations==");

  for (l, r) in data.collocations_iter() {
    println!("({:?}, {:?})", l, r);
  }

  /*
  println!("==Orthographic Context==");

  for (s, ctxt) in data.orthographic_context_iter() { 
    println!("{:?}, {:?}", ctxt, s);
  }
  */
}
