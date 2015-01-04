pub use tokenizer::word::PunktWordTokenizer as WordTokenizer;
pub use tokenizer::sentence::PunktSentenceTokenizer as SentenceTokenizer;
pub use tokenizer::token::PunktToken as Token;

mod word;
mod token;
mod sentence;

pub mod prelude {
  pub use tokenizer::token::{
    DocumentSlice, 
    DocumentIndexedSlice, 
    LetterCase};

  /// A Tokenizer. Can take a corpus text and return its 
  /// constituent tokens. The definition of a `Token` is up
  /// to the tokenizer.
  pub trait Tokenizer<'a, T: 'a> {
    fn tokenize_document(&'a self, &'a str) -> Vec<T>;
  }
}
