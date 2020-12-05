#![crate_name = "geometry_kernel"]

// Сильно урезанная заимствованная библиотека для работы с stl-файлами.

extern crate core;
extern crate time;
extern crate num;
extern crate gmp;
extern crate bidir_map;
extern crate byteorder;
extern crate test;
extern crate rulinalg;


#[macro_use]
extern crate lazy_static;

#[macro_use]
extern crate log;
extern crate env_logger;

pub mod primitives;
