use std::default::Default;

use phf::Set;

use iter::consecutive_token_iter_mut;
use ortho::{ORTHO_LC, MIDDLE_UC, BEGIN_LC, ORTHO_UC};
use prelude::PunktFirstPassAnnotater;

use trainer::Data;
use tokenizer::prelude::{DocumentIndexedSlice, DocumentSlice};

static PUNCTUATION: Set<u8> = phf_set! { b'.', b';', b':', b',', b'!', b'?' };

pub struct PunktSentenceTokenizer<'a> {
  data: &'a Data,
}

impl<'a> PunktSentenceTokenizer<'a> {
  #[inline]
  pub fn new(data: &'a Data) -> PunktSentenceTokenizer<'a> {
    PunktSentenceTokenizer { data: data }
  }
}

impl<'a> PunktFirstPassAnnotater for PunktSentenceTokenizer<'a> {
  #[inline]
  fn data(&self) -> &Data {
    &*self.data
  }
}
