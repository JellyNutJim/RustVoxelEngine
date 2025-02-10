use std::io::Cursor;
use image::{FlatSamples, ImageBuffer, ImageReader};
use winit::window::CustomCursor;



fn get_img_samples(path: &str) -> Vec<u8>{
    let img = ImageReader::open(path).expect("").decode().expect("").to_rgb8();
    img.into_flat_samples().samples
}

pub fn get_rgba8_image_samples(path: &str) -> (Vec<u8>, u16, u16) {
    let img = ImageReader::open(path).expect("").decode().expect("").to_rgb8();
    let samples = img.into_flat_samples();
    let (_, w, h) = samples.extents();
    let (w, h) = (w as u16, h as u16);
    CustomCursor::from_rgba(samples.samples.clone(), w, h, w / 2, h / 2).unwrap();
    (samples.samples, w, h)
}


