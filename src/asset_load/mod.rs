mod rgba8_img;

use rgba8_img::*;
use winit::window::CustomCursorSource;

pub fn get_rbga8_img_samples(path: &str) -> (Vec<u8>, u16, u16){
    get_rgba8_image_samples(path)
}