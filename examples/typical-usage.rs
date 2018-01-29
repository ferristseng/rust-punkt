extern crate punkt;

use punkt::{SentenceTokenizer, Trainer, TrainingData};
use punkt::params::Standard;

fn main() {
  let docs = [
    "This is sentence one. This is sentence two.",
    "The U.S. is a great country.",
    "I bought $5.50 worth of apples from the store. I gave them to my dog when I came home.",
  ];

  // The punkt algorithm can derive all the knowledge it needs to know from
  // the input document.
  println!("\n-- Trained only using document --");
  for d in docs.iter() {
    let trainer: Trainer<Standard> = Trainer::new();
    let mut data = TrainingData::new();

    trainer.train(d, &mut data);

    for s in SentenceTokenizer::<Standard>::new(d, &data) {
      println!("{:?}", s);
    }
  }

  // Alternatively, you can use some pretrained data.
  let english = TrainingData::english();

  println!("\n-- Using pretrained data --");
  for d in docs.iter() {
    for s in SentenceTokenizer::<Standard>::new(d, &english) {
      println!("{:?}", s);
    }
  }

  // You can incrementally build up training data too.
  let trainer: Trainer<Standard> = Trainer::new();
  let mut data = TrainingData::new();

  println!("\n-- Trained incrementally --");
  for d in docs.iter() {
    trainer.train(d, &mut data);

    for s in SentenceTokenizer::<Standard>::new(d, &data) {
      println!("{:?}", s);
    }
  }

  println!(
    "\nIs 'u.s' an abbreviation? : {:?}",
    data.contains_abbrev("u.s")
  );

  assert!(data.contains_abbrev("u.s"));
}
