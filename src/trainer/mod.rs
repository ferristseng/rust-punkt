pub use trainer::data::TrainingData;

use std::ops::Deref;
use std::hash::{Hasher, Hash};

use token::Token; 


mod data;
mod trainer;