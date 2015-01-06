use std::default::Default;

use phf::Set;

static PUNCTUATION: Set<u8> = phf_set! { b'.', b';', b':', b',', b'!', b'?' };

pub struct SentenceTokenizer;

/*
impl Iterator for SentenceTokenizer {
  
}
*/
