extern crate punkt;

use std::default::Default;

use punkt::trainer::Trainer;

fn main() {
  let docs: Vec<&str> = vec![
    include_str!("../test/raw/npr-article-01.txt"),
    include_str!("../test/raw/ny-times-article-01.txt"),
    include_str!("../test/raw/pride-and-prejudice.txt")
  ];
  let mut data = Default::default(); 
  
  {
    let mut trainer = Trainer::new(&mut data);

    for d in docs.iter() {
      trainer.train(*d);
    }

    trainer.finalize();
  }

  println!("==Abbreviations ({})==", data.abbrevs_len());

  for ab in data.abbrevs_iter() {
    println!("{:?}", ab);
  }

  println!("==Sentence Starters ({})==", data.sentence_starters_len());

  for ss in data.sentence_starters_iter() {
    println!("{:?}", ss);
  }

  println!("==Collocations ({})==", data.collocations_len());

  for (l, r) in data.collocations_iter() {
    println!("({:?}, {:?})", l, r);
  }

  println!("==Orthographic Context ({})==", data.orthographic_context_len());

  /*
  for (s, ctxt) in data.orthographic_context_iter() { 
    println!("{:?}, {:?}", ctxt, s);
  }
  */
}
