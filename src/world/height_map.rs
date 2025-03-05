use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct HeightMap {
    map: VecDeque<VecDeque<f64>>,
}

impl HeightMap {
    pub fn new(initial_height: f64, width: usize) -> Self {
        Self {
            map: VecDeque::from(vec![ VecDeque::from(vec![initial_height; width]); width]),
        }
    }

    pub fn get(&self, x: usize, z: usize) -> f64{
        return self.map[x][z]
    }

    pub fn get_mut(&mut self, x: usize, z: usize) -> &mut f64{
        return &mut self.map[x][z]
    }

    pub fn set(&mut self, x: usize, z: usize, y: f64) {
        self.map[x][z] = y;
    }

    // Always assumes single chunk shift
    pub fn shift(&mut self, axis: usize, dir: i32) {
        let shift_length = 64;

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