use phf::Map;

pub type OrthographicContext = u8;

/// Context that a token can be in.
#[derive(Show, Eq, PartialEq)]
pub enum OrthographyPosition {
  Initial,
  Internal,
  Unknown
}

impl OrthographyPosition {
  pub fn as_byte(&self) -> u8 {
    match *self {
      OrthographyPosition::Initial   => 0b01000000,
      OrthographyPosition::Internal  => 0b00100000,
      OrthographyPosition::Unknown   => 0b01100000
    }
  }
}

pub const BEG_UC: OrthographicContext = 0b00000010;
pub const MID_UC: OrthographicContext = 0b00000100;
pub const UNK_UC: OrthographicContext = 0b00001000;
pub const BEG_LC: OrthographicContext = 0b00010000;
pub const MID_LC: OrthographicContext = 0b00100000;
pub const UNK_LC: OrthographicContext = 0b01000000;

/// Map mapping a combination of LetterCase and OrthographyPosition 
/// to an OrthographicConstant describing orthographic attributes about the 
/// token. The chars (in ASCII) map to the result of ORing certains
/// OrthographyPosition and LetterCase with one another.
pub static ORTHO_MAP: Map<u8, OrthographicContext> = phf_map! {
  b'B' => BEG_UC, // 66
  b'"' => MID_UC, // 34
  b'b' => UNK_UC, // 98
  b'A' => BEG_LC, // 65
  b'!' => MID_LC, // 33
  b'a' => UNK_LC  // 97
};