use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct BiomeMap {
    map: VecDeque<VecDeque<Biome>>,
}

#[derive(Debug, Clone, Copy)]
pub enum Biome {
    Single(u8),
    Double(u8, u8, f64),
    Triple(u8, u8, u8, f64, f64)
}

#[allow(unused)]
impl BiomeMap {
    pub fn new(initial_biome: Biome, width: usize) -> Self {

        Self {
            map: VecDeque::from(vec![ VecDeque::from(vec![initial_biome; width]); width]),
        }
    }

    pub fn get(&self, x: usize, z: usize) -> Biome{
        return self.map[x][z]
    }

    pub fn get_mut(&mut self, x: usize, z: usize) -> &mut Biome{
        return &mut self.map[x][z]
    }

    pub fn set(&mut self, x: usize, z: usize, biome: Biome) {
        self.map[x][z] = biome;
    }

    // Always assumes single chunk shift
    pub fn shift(&mut self, axis: usize, dir: i32, shift_length: usize) {

        if axis == 0 {
            self.shift_x(dir, shift_length);
        } 
        else {
            self.shift_y(dir, shift_length);
        }
    }

    fn shift_x(&mut self, dir: i32, length: usize) {
        if dir == 1 {
            self.map.rotate_left(length);
        }
        else {
            self.map.rotate_right(length);
        }
    }

    fn shift_y(&mut self, dir: i32, length: usize) { 
        if dir == 1 {
            for row in self.map.iter_mut() {
                row.rotate_left(length);
            }
        }
        else {
            for row in self.map.iter_mut() {
                row.rotate_right(length);
            }
        }
    
    }
}