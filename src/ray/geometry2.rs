// Basic 2D raytracable geometry
//
// This just concerns itself with the distance along the ray we make contact

use cgmath::prelude::*;
use cgmath::{vec2, dot, Point2, Vector2};

// Helper to remember which we want to always be pre-unitized
type UnitVector2 = Vector2<f64>;


pub struct Ray2 {
    pub start : Point2<f64>,
    pub direction : Vector2<f64>
}

impl Ray2 {
    fn at(&self, t : f64) -> Point2<f64> {
        self.start + (self.direction * t)
    }
}

pub struct Circle {
    pub radius : f64
}

pub struct OriginSquare {
    pub x_half : f64,
    pub y_half : f64
}

pub trait TraceGeometry2: std::marker::Sync {
    fn trace(&self, ray : &Ray2) -> Option<(f64, UnitVector2)>;
    fn inside(&self, point : &Point2<f64>) -> bool;
}

impl TraceGeometry2 for Circle {
    fn inside(&self, point : &Point2<f64>) -> bool {
        point.to_vec().magnitude2() <= (self.radius * self.radius)
    }

    fn trace(&self, ray : &Ray2) -> Option<(f64, UnitVector2)> {
        // Basic ray intersection
        let diff = -1.0*ray.start.to_vec();
        // Get nearest point on ray to centre
        let t_nearest = dot(ray.direction, diff);
        // Hit behind
        if t_nearest < 0.0 {
            return None;
        }
        let projection_nearest = diff - (ray.direction * t_nearest);
        let dist_nearest_sq = projection_nearest.magnitude2();
        // Ray does not contact sphere
        let radius_sq = self.radius * self.radius;
        if dist_nearest_sq > radius_sq {
            return None
        }
        // Amount before nearest point on ray that we contact
        let t_before = (radius_sq - dist_nearest_sq).sqrt();
        let t_contact = t_nearest - t_before;
        // Inside the sphere
        if t_contact < 0.0 {
            return None;
        }
        let pos = ray.start + ray.direction *t_contact;
        Some(
            (t_contact, pos.to_vec().normalize())
        )
    }
}

impl TraceGeometry2 for OriginSquare {
    fn inside(&self, point : &Point2<f64>) -> bool {
        point.x.abs() <= self.x_half && point.y.abs() <= self.y_half
    }

    fn trace(&self, ray : &Ray2) -> Option<(f64, UnitVector2)> {
        if ray.start.y > self.y_half {
            if ray.direction.y >= 0.0 {
                return None;
            }
            // Get intersection with top plane 
            let t_contact_top = (self.y_half - ray.start.y) / ray.direction.y;
            let pos_top = ray.at(t_contact_top).x;
            if pos_top.abs() <= self.x_half {
                return Some((t_contact_top, vec2(0.0, 1.0)));
            }
        }
        if ray.start.y < -self.y_half {
            if ray.direction.y <= 0.0 {
                return None;
            }
            // Get intersection with top plane 
            let t_contact_bot = (-self.y_half - ray.start.y) / ray.direction.y;
            let pos_bot = ray.at(t_contact_bot).x;
            if pos_bot.abs() <= self.x_half {
                return Some((t_contact_bot, vec2(0.0, 1.0)));
            }
        }
        // Are we going to hit the extension
        if ray.direction.x.abs() < 1e-9 {
            return None;
        }
        if ray.direction.x > 0.0 {
            if ray.start.x > -self.x_half {
                return None;
            }
            let diff_x = -self.x_half - ray.start.x;
            let t_contact = diff_x / ray.direction.x;
            let y = ray.start.y + (t_contact * ray.direction.y);
            if y.abs() > self.y_half {
                return None;
            }
            Some((t_contact, vec2(-1.0, 0.0)))
        } else {
            if ray.start.x < self.x_half {
                return None;
            }
            let diff_x = self.x_half - ray.start.x;
            let t_contact = diff_x / ray.direction.x;
            let y = ray.start.y + (t_contact * ray.direction.y);
            if y.abs() > self.y_half {
                return None;
            }
            Some((t_contact, vec2(1.0, 0.0)))
        }
    }
}


#[cfg(test)]
mod tests {
    use cgmath::{point2, assert_abs_diff_eq};

    use super::*;

    #[test]
    fn test_trace_circle_hit() {
        let circle = Circle {radius : 1.0};
        // Fire directy down
        let direct_down = circle.trace(&
            Ray2 {start : point2(0.0,2.0), direction : vec2(0.0,-1.0)}
        );
        assert!(direct_down.is_some());
        assert_abs_diff_eq!(direct_down.unwrap().0, 1.0);
        assert_abs_diff_eq!(direct_down.unwrap().1, vec2(0.0, 1.0));
    }

    #[test]
    fn test_trace_circle_miss() {
        let circle = Circle {radius : 1.0};
        // Fire directy down
        let direct_down = circle.trace(&
            Ray2 {start : point2(1.1,1.0), direction : vec2(0.0,-1.0)}
        );
        assert!(direct_down.is_none());
    }


    #[test]
    fn test_trace_square_axis_hit() {
        let cube = OriginSquare { x_half : 1.0, y_half : 1.0};
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray2 {start : point2(-10.0,0.0), direction : vec2(1.0,0.0)}
        );
        // Miss
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().0, 9.0);
        assert_abs_diff_eq!(trace.unwrap().1, vec2(-1.0, 0.0));
    }

    #[test]
    fn test_trace_box_slanted_down_hit() {
        let cube = OriginSquare { x_half : 1.0, y_half : 1.0};
        // Fire along axis and hit
        let dir = vec2(1.0, -1e-3).normalize();
        let trace = cube.trace(&
            Ray2 {start : point2(-10.0,0.0), direction : dir}
        );
        // Miss
        assert!(trace.is_some());
        assert_abs_diff_eq!(trace.unwrap().1, vec2(-1.0, 0.0));
    }

    #[test]
    fn test_trace_box_axis_miss() {
        let cube = OriginSquare { x_half : 1.0, y_half : 1.0};
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray2 {start : point2(-10.0,4.0), direction : vec2(1.0,0.0)}
        );
        // Miss
        assert!(trace.is_none());
    }
    #[test]
    fn test_trace_box_axis_inside() {
        let cube = OriginSquare { x_half : 1.0, y_half : 1.0};
        // Fire along axis and hit
        let trace = cube.trace(&
            Ray2 {start : point2(0.0,0.0), direction : vec2(1.0,0.0)}
        );
        // Miss
        assert!(trace.is_none());
    }
}