#[macro_use]
extern crate glium;
extern crate gtk;
extern crate gdk;
extern crate libc;
extern crate epoxy;
extern crate shared_library;
extern crate glm;
extern crate image;
extern crate csv;
extern crate geometry_kernel;
extern crate glib;
extern crate plotlib;
extern crate gst;
extern crate gst_video;

use gst::prelude::*;
use gst_video::prelude::*;
use std::process;
use std::os::raw::c_void;


mod madgwick;

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::view::ContinuousView;
use plotlib::style::LineStyle;

use std::str::FromStr;
use std::io::Cursor;
use std::ptr;
use std::rc::Rc;
use std::vec::Vec;

use self::gtk::traits::*;
use self::gtk::Inhibit;
use self::gtk::{GLArea, Window};

use geometry_kernel::primitives::mesh::Mesh;

use std::fs::File;

use glium::Surface;

use std::f32::MAX;
use std::f32::MIN;

use std::io::BufRead;

use std::f32::consts::PI;

use madgwick::Madgwick;


use self::shared_library::dynamic_library::DynamicLibrary;

// make moving clones into closures more convenient
macro_rules! clone {
    ($($n:ident),+; || $body:block) => (
        {
            $( let $n = $n.clone(); )+
            move || { $body }
        }
    );
    ($($n:ident),+; |$($p:ident),+| $body:block) => (
        {
            $( let $n = $n.clone(); )+
            move |$($p),+| { $body }
        }
    );
}

#[derive(Copy, Clone)]
struct VertexModel {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}
implement_vertex!(VertexModel, position, tex_coords, normal);

fn make_model(path : &str) -> (Vec<VertexModel>, (f32, f32, f32, f32)){
    let mut maximum = MIN;
    let mut max_x = MIN;
    let mut max_y = MIN;
    let mut max_z = MIN;
    let mut min_x = MAX;
    let mut min_y = MAX;
    let mut min_z = MAX;
    let mut model = vec![];
    let mut model_file = File::open(path).unwrap();
    let model_mesh = Mesh::read_stl(&mut model_file).unwrap();
    let triangle_indices = model_mesh.get_it_iterator();
    use geometry_kernel::primitives::number::NumberTrait;
    for i in triangle_indices {
        let triangle = model_mesh.get_triangle(i);
        let n = triangle.get_normal();
        let points = triangle.get_points();
        for p in points {
            let px = p.x.convert_to_f32();
            let py = p.y.convert_to_f32();
            let pz = p.z.convert_to_f32();
            let step_max = px.abs().max(py.max(pz.abs()));
            maximum = maximum.max(step_max);
            max_x = max_x.max(px);
            max_y = max_y.max(py);
            max_z = max_z.max(pz);
            min_x = min_x.min(px);
            min_y = min_y.min(py);
            min_z = min_z.min(pz);
            let nx = n.clone().x.convert_to_f32();
            let ny = n.clone().y.convert_to_f32();
            let nz = n.clone().z.convert_to_f32();
            model.push(VertexModel {
                position: [px, py, pz],
                tex_coords: [px, py], //tex_coords[index],
                normal: [nx, ny, nz]
            });
        }
    }
    let delta_x = max_x - (max_x-min_x)/2.0;
    let delta_y = max_y - (max_y-min_y)/2.0;
    let delta_z = max_z - (max_z-min_z)/2.0;
    let mut normalize_model = vec![];
    for vertex in &model {
        normalize_model.push(VertexModel {
            position: [vertex.position[0]- delta_x,
                vertex.position[1] - delta_y, vertex.position[2]-delta_z],
            tex_coords: vertex.tex_coords,
            normal: vertex.normal
        })
    }
    return (normalize_model, (0.0, 0.0, 0.0, 1.0 / maximum));
}

fn run_video(uri: &str) -> (std::option::Option<gst::Element>, std::option::Option<gst::Bus>) {
    gst::init().unwrap();
    let playbin = gst::ElementFactory::make("playbin", None).unwrap();
    playbin.set_property("uri", &uri).unwrap();
    let videoflip = gst::ElementFactory::make("videoflip", None).unwrap();
    videoflip.set_property_from_str("method", "clockwise");
    playbin.set_property("video-filter", &videoflip).unwrap();
    let bus = playbin.get_bus().unwrap();

    playbin
        .set_state(gst::State::Playing)
        .expect("Unable to set the pipeline to the `Playing` state");

    return (Some(playbin), Some(bus));
}

fn get_kr_val(last_time: f32, last_dg: f32, time: f32, dg: f32, full_time_kr: f32) -> f32 {
    let rdg = -((last_dg - dg) * full_time_kr + (last_time * dg - time * last_dg)) / (time - last_time);
//    println!("lt: {} - ldg: {}\n t: {} - dg {}\n rt: {} - rdg: {}\n\n", last_time, last_dg, time, dg, full_time_kr, rdg);
    return rdg;
}

struct ModelState {
    model: std::vec::Vec<VertexModel>,
    is_render: bool,
    dx: f32, dy: f32, dz: f32,
    max_scale: f32
}

fn main() {
    if gtk::init().is_err() {
        println!("Failed to initialize GTK.");
        return;
    }

    let window = Window::new(gtk::WindowType::Toplevel);
    let glarea = GLArea::new();
    glarea.set_has_depth_buffer(true);
    window.connect_delete_event(|_, _| {
        gtk::main_quit();
        Inhibit(false)
    });

    epoxy::load_with(|s| {
        unsafe {
            match DynamicLibrary::open(None).unwrap().symbol(s) {
                Ok(v) => v,
                Err(_) => ptr::null(),
            }
        }
    });

    struct Backend {
        glarea: GLArea,
    }

    unsafe impl glium::backend::Backend for Backend {
        fn swap_buffers(&self) -> Result<(), glium::SwapBuffersError> {
            Ok(())
        }

        unsafe fn get_proc_address(&self, symbol: &str) -> *const std::os::raw::c_void {
            epoxy::get_proc_addr(symbol)
        }

        fn get_framebuffer_dimensions(&self) -> (u32, u32) {
            (self.glarea.get_allocated_width() as u32, self.glarea.get_allocated_height() as u32)
        }

        fn is_current(&self) -> bool {
            unsafe { self.make_current() };
            true
        }

        unsafe fn make_current(&self) {
            if self.glarea.get_realized() {
                self.glarea.make_current();
            }
        }
    }

    struct Facade {
        context: Rc<glium::backend::Context>,
    }

    impl glium::backend::Facade for Facade {
        fn get_context(&self) -> &Rc<glium::backend::Context> {
            &self.context
        }
    }

    impl Facade {
        fn draw(&self) -> glium::Frame {
            glium::Frame::new(self.context.clone(), self.context.get_framebuffer_dimensions())
        }
    }

    #[derive(Copy, Clone)]
    struct VertexLight {
        position: [f32; 3],
    }

    implement_vertex!(VertexLight, position);
    struct StateInfo {
        display: Facade,
        light_buffer: glium::VertexBuffer<VertexLight>,
        light_indices: glium::index::NoIndices,
        program_light: glium::program::Program,
        model_buffer: glium::VertexBuffer<VertexModel>,
        model_indices: glium::index::NoIndices,
        program_model: glium::program::Program,
        texture: glium::texture::Texture2d,
    }

    struct State {
        tx: f32, ty: f32, tz: f32,
        rx: f32, ry: f32, rz: f32,
        scale: f32,
        is_draw: bool,
        is_light: bool, is_texture: bool,
        int: f32, amb: f32, diff: f32, spec: f32,
        back_color : gdk::RGBA, model_color : gdk::RGBA,
        anime_list: Vec<(f32, f32, f32, f32)>,
        al_ind: usize,
        is_anime: bool,
        arx: f32, ary: f32, arz: f32,
        playbin: std::option::Option<gst::Element>,
        bus: std::option::Option<gst::Bus>,
    }

    let state_info: std::sync::Arc<std::sync::Mutex<Option<StateInfo>>> = std::sync::Arc::new(std::sync::Mutex::new(None));
    let state: std::sync::Arc<std::sync::Mutex<Option<State>>> = std::sync::Arc::new(std::sync::Mutex::new(None));

    glarea.connect_realize(clone!(glarea, state, state_info; |_widget| {
        let mut state = state.lock().unwrap();
        let mut state_info = state_info.lock().unwrap();

        let display = Facade {
            context: unsafe {
                glium::backend::Context::new::<_, >(
                    Backend {
                        glarea: glarea.clone(),
                    }, true, Default::default())
            }.unwrap(),
        };

        let cube_light = vec![VertexLight {position: [-0.18, -0.18, -0.18]},
                              VertexLight {position: [-0.18, 0.18, -0.18]}, VertexLight {position: [0.18, -0.18, -0.18]},
                              VertexLight {position: [-0.18, 0.18, -0.18]},
                              VertexLight {position: [0.18, 0.18, -0.18]}, VertexLight {position: [0.18, -0.18, -0.18]},
                              VertexLight {position: [-0.18, -0.18, 0.18]},
                              VertexLight {position: [-0.18, 0.18, 0.18]}, VertexLight {position: [0.18, -0.18, 0.18]},
                              VertexLight {position: [-0.18, 0.18, 0.18]},
                              VertexLight {position: [0.18, 0.18, 0.18]}, VertexLight {position: [0.18, -0.18, 0.18]},
                              VertexLight {position: [0.18, -0.18, -0.18]},
                              VertexLight {position: [0.18, 0.18, -0.18]}, VertexLight {position: [0.18, -0.18, 0.18]},
                              VertexLight {position: [0.18, 0.18, -0.18]},
                              VertexLight {position: [0.18, 0.18, 0.18]}, VertexLight {position: [0.18, -0.18, 0.18]},
                              VertexLight {position: [-0.18, -0.18, -0.18]},
                              VertexLight {position: [-0.18, 0.18, -0.18]}, VertexLight {position: [-0.18, -0.18, 0.18]},
                              VertexLight {position: [-0.18, 0.18, -0.18]},
                              VertexLight {position: [-0.18, 0.18, 0.18]}, VertexLight {position: [-0.18, -0.18, 0.18]},
                              VertexLight {position: [-0.18, 0.18, -0.18]},
                              VertexLight {position: [0.18, 0.18, 0.18]}, VertexLight {position: [0.18, 0.18, -0.18]},
                              VertexLight {position: [-0.18, 0.18, 0.18]},
                              VertexLight {position: [0.18, 0.18, 0.18]}, VertexLight {position: [0.18, 0.18, -0.18]},
                              VertexLight {position: [-0.18, -0.18, -0.18]},
                              VertexLight {position: [-0.18, -0.18, 0.18]}, VertexLight {position: [0.18, -0.18, -0.18]},
                              VertexLight {position: [-0.18, -0.18, 0.18]},
                              VertexLight {position: [0.18, -0.18, 0.18]}, VertexLight {position: [0.18, -0.18, -0.18]}];

        let model = make_model("/home/alexander/IdeaProjects/sport_visual/models/union.stl").0;
        let light_buffer = glium::VertexBuffer::new(&display, &cube_light).unwrap();
        let light_indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
        let model_buffer = glium::VertexBuffer::new(&display, &model).unwrap();
        let model_indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

        let vertex_shader_light = r#"
            #version 330
            in vec3 position;
            uniform mat4 modelMatrix, projectionMatrix;
            void main() {
                gl_Position = projectionMatrix * modelMatrix * vec4(position, 1.0);
            }
        "#;

        let fragment_shader_light = r#"
            #version 330
            out vec4 color;
            void main() {
                color = vec4(1.0, 1.0, 1.0, 1.0);
            }
        "#;

         let vertex_shader_model = r#"
            #version 330
            in vec3 position;
            in vec2 tex_coords;
            in vec3 normal;
            out vec2 v_tex_coords;
            out vec3 v_normal;
            out vec3 v_position;
            uniform mat4 modelMatrix, projectionMatrix;
            void main() {
                v_tex_coords = tex_coords;
                v_normal = normalize(mat3(transpose(inverse(modelMatrix)))*normal);
                v_position = vec3(modelMatrix*vec4(position, 1.0));
                gl_Position = projectionMatrix * modelMatrix * vec4(position, 1.0);
            }
        "#;

        let fragment_shader_model = r#"
            #version 330
            in vec2 v_tex_coords;
            in vec3 v_normal;
            in vec3 v_position;
            out vec4 color;
            uniform sampler2D tex;
            uniform vec3 LightPosition;
            uniform vec3 LightIntensity;
            uniform vec3 MaterialKa;
            uniform vec3 MaterialKd;
            uniform float MaterialKs;
            uniform bool is_light;
            uniform bool is_texture;
            uniform vec4 model_color;
            out vec4 FragColor;
            void phongModel(vec3 pos, vec3 norm, out vec3 ambAndDiffspec) {
                vec3 ambient = LightIntensity*MaterialKa;
                vec3 lightDir = normalize(LightPosition - v_position);
                float diff = max(dot(v_normal, lightDir), 0.0);
                vec3 diffuse = LightIntensity*(diff * MaterialKd);
                vec3 viewPos = vec3(0.0, 0.0, 2.0);
                vec3 viewDir = normalize(viewPos - pos);
                vec3 r = reflect(-lightDir, norm);
                vec3 specular = vec3(pow(max(dot(r,viewDir), 0.0), 32)*MaterialKs*diff);
                ambAndDiffspec = ambient  + diffuse + specular;
            }
            void main() {
                vec3 ambAndDiffspec;
                vec4 texColor = texture(tex, v_tex_coords);
                phongModel(v_position, v_normal, ambAndDiffspec);
                if(is_light) {
                    FragColor = vec4(ambAndDiffspec, 1.0) * model_color;
                } else {
                    FragColor = model_color;
                }
                if(is_texture) {
                    FragColor *= texColor;
                }
            }
        "#;

        let program_light = glium::Program::from_source(&display, vertex_shader_light, fragment_shader_light, None).unwrap();
        let program_model = glium::Program::from_source(&display, vertex_shader_model, fragment_shader_model, None).unwrap();
        let image = image::load(
            Cursor::new(&include_bytes!("../textures/t2.jpg")[..]),image::ImageFormat::Jpeg).unwrap().to_rgba();
        let image_dimensions = image.dimensions();
        let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
        let texture = glium::texture::Texture2d::new(&display, image).unwrap();

        let tx = 0.0f32; let ty = 0.0f32; let tz = 0.0f32;
        let rx = 30.0f32; let ry = 45.0f32; let rz = 0.0f32;
        let scale = 0.5f32;
        let is_draw = true;
        let is_light = true;
        let is_texture = true;
        let int = 1.0f32; let amb = 0.5f32; let diff = 1.0f32; let spec = 0.8f32;
        let back_color = gdk::RGBA{red : 0.0, green : 0.0, blue : 0.0, alpha : 1.0};
        let model_color = gdk::RGBA{red : 1.0, green : 1.0, blue : 1.0, alpha : 1.0};
        let is_anime = false;
        let anime_list: Vec<(f32, f32, f32, f32)> = Vec::new();
        let al_ind = 0;

        *state = Some(State {
                tx : tx, ty : ty, tz : tz,
                rx : rx, ry : ry, rz: rz, scale : scale,
                is_draw : is_draw,
                is_light : is_light, is_texture : is_texture,
                int : int, amb : amb, diff : diff, spec : spec,
                back_color : back_color, model_color : model_color,
                anime_list: anime_list,
                al_ind: al_ind,
                is_anime: is_anime,
                arx: 0.0f32, ary: 0.0f32, arz: 0.0f32,
                playbin: None,
                bus: None,
        });
        *state_info = Some(StateInfo {
                display: display,
                light_buffer: light_buffer,
                light_indices: light_indices,
                program_light: program_light,
                model_buffer: model_buffer,
                model_indices: model_indices,
                program_model: program_model,
                texture : texture,
        })

    }));
    let model = make_model("/home/alexander/IdeaProjects/sport_visual/models/union.stl");
    let model_state: std::sync::Arc<std::sync::Mutex<ModelState>> = std::sync::Arc::new(std::sync::Mutex::new(ModelState{
        model : model.0, is_render : false, dx : (model.1).0, dy : (model.1).1, dz : (model.1).2, max_scale: (model.1).3,
    }));

    glarea.connect_unrealize(clone!(state; |_widget| {
            let mut state = state.lock().unwrap();
            *state = None;
        }));

    glarea.connect_render(clone!(state, state_info, model_state; |_glarea, _glctx| {
            let mut state = state.lock().unwrap();
            let state = state.as_mut().unwrap();
            let mut state_info = state_info.lock().unwrap();
            let state_info = state_info.as_mut().unwrap();

            state_info.model_buffer = glium::VertexBuffer::new(&state_info.display, &model_state.lock().unwrap().model).unwrap();
            let max_scale = model_state.lock().unwrap().max_scale;
//            let dx = model_state.lock().unwrap().dx;
//            let dy = model_state.lock().unwrap().dy;
//            let dz = model_state.lock().unwrap().dz;
            let int = [state.int, state.int, state.int];
            let amb = [state.amb, state.amb, state.amb];
            let diff = [state.diff, state.diff, state.diff];
            let spec = state.spec;
            let back = state.back_color;
            let color = state.model_color;
            let mut target = state_info.display.draw();
            target.clear_color_and_depth((back.red as f32,
                back.green as f32 , back.blue as f32, back.alpha as f32), 1.0);
            let (w, h) = target.get_dimensions();
            let lm = glm::ext::look_at(glm::vec3(0.0, 0.0, 2.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));
            let tm_light0 = glm::ext::translate(&lm, glm::vec3(state.tx-0.5, state.ty+0.5, state.tz));
            let tm_light = glm::ext::scale(&tm_light0, glm::vec3(0.25, 0.25, 0.25));
            let tm_light = tm_light.as_array();
//            let tm = glm::ext::translate(&lm, glm::vec3(-dx / (w as f32), -dy / (h as f32), -dz));
            let rmx = glm::ext::rotate(&lm, glm::radians(state.rx), glm::vec3(1.0, 0.0, 0.0));
            let rmy = glm::ext::rotate(&rmx, glm::radians(state.ry), glm::vec3(0.0, 1.0, 0.0));
            let rmz = glm::ext::rotate(&rmy, glm::radians(state.rz), glm::vec3(0.0, 0.0, 1.0));
            let sm = glm::ext::scale(&rmz, glm::vec3(state.scale*max_scale, state.scale*max_scale, state.scale*max_scale));
            let sm = sm.as_array();
            let pmv  = glm::ext::perspective_rh(glm::radians(45.0f32),
                w as f32 / h as f32, 0.1f32, 100.0f32);
            let pmv = pmv.as_array();
            let pm = [
                *pmv[0].as_array(), *pmv[1].as_array(), *pmv[2].as_array(), *pmv[3].as_array(),
            ];
            let uniforms_light = uniform! {
                modelMatrix : [
                    *tm_light[0].as_array(), *tm_light[1].as_array(),
                    *tm_light[2].as_array(), *tm_light[3].as_array(),
                ],
                projectionMatrix: pm,
            };

            let uniforms_model = uniform! {
                modelMatrix : [
                    *sm[0].as_array(), *sm[1].as_array(), *sm[2].as_array(), *sm[3].as_array(),
                ],
                projectionMatrix: pm,
                tex: glium::uniforms::Sampler::wrap_function(glium::uniforms::Sampler::new
                    (&state_info.texture),glium::uniforms::SamplerWrapFunction::Repeat),
                LightIntensity: int,
                LightPosition: [state.tx-0.5, state.ty+0.5, 0.0f32],
                MaterialKa: amb,
                MaterialKd: diff,
                MaterialKs: spec,
                is_light: state.is_light,
                is_texture: state.is_texture,
                model_color: [color.red as f32, color.green as f32, color.blue as f32, color.alpha as f32],
            };
            let params = glium::DrawParameters {
                depth: glium::Depth {
                    clamp: glium::draw_parameters::DepthClamp::Clamp,
                    .. Default::default()
                },
                viewport: Some(glium::Rect {
                    left : 0, bottom : 0,  width : w, height : h
                }),
                .. Default::default()
            };
            if state.is_draw {
                if state.is_light {
                    target.draw(&state_info.light_buffer, &state_info.light_indices, &state_info.program_light,
                        &uniforms_light,&params).unwrap();
                }
                target.draw(&state_info.model_buffer, &state_info.model_indices, &state_info.program_model,
                    &uniforms_model,&params).unwrap();
            }
            target.finish().unwrap();
            Inhibit(false)
        }));
    window.set_title("GLArea Example");
    window.set_default_size(1400, 700);

    let hbox = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    hbox.set_homogeneous(false);

    let model_frame = gtk::Frame::new("Model");
    let model_box = gtk::Box::new(gtk::Orientation::Vertical, 5);
    let lightning_frame = gtk::Frame::new("Lightning");
    let lightning_box = gtk::Box::new(gtk::Orientation::Vertical, 5);
    let animation_frame = gtk::Frame::new("Animation");
    let animation_box = gtk::Box::new(gtk::Orientation::Vertical, 10);

    model_frame.add(&model_box);
    lightning_frame.add(&lightning_box);
    animation_frame.add(&animation_box);

    let colours_box = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    colours_box.set_homogeneous(true);
    colours_box.set_spacing(5);
    colours_box.set_border_width(3);

    let light_box1 = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    light_box1.set_homogeneous(true);
    light_box1.set_spacing(5);
    light_box1.set_border_width(3);

    let light_box2 = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    light_box2.set_homogeneous(true);
    light_box2.set_spacing(5);
    light_box2.set_border_width(3);

    let c_box = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let b_box = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let int_box = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let amb_box = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let spec_box = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let diff_box = gtk::Box::new(gtk::Orientation::Vertical, 1);

    colours_box.add(&c_box);
    colours_box.add(&b_box);
    light_box2.add(&diff_box);
    light_box2.add(&spec_box);
    light_box1.add(&int_box);
    light_box1.add(&amb_box);

    lightning_box.add(&light_box1);
    lightning_box.add(&light_box2);

    model_frame.set_border_width(10);
    lightning_frame.set_border_width(10);
    animation_frame.set_border_width(10);

    let button_box = gtk::Box::new(gtk::Orientation::Vertical, 20);
    button_box.set_homogeneous(false);
    let area_box = gtk::Box::new(gtk::Orientation::Vertical, 5);
    area_box.set_homogeneous(false);
    area_box.set_hexpand(true);
    area_box.set_vexpand(true);
    let area_sub_box = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    area_sub_box.set_homogeneous(true);
    area_sub_box.set_hexpand(true);
    area_sub_box.set_vexpand(true);
    area_sub_box.set_border_width(5);
    area_box.add(&area_sub_box);
    let scale_box = gtk::Box::new(gtk::Orientation::Vertical, 0);
    area_box.add(&scale_box);
    scale_box.set_hexpand(true);
    scale_box.set_vexpand(false);
    scale_box.set_border_width(5);

    let scroll_video_button = gtk::Scale::new_with_range(gtk::Orientation::Horizontal, 0.0, 1.0, 0.05);

    let progress = gtk::ProgressBar::new();
    progress.set_text("MODEL LOADING");
    progress.set_fraction(0.0);
    progress.set_pulse_step(0.1);
    progress.set_show_text(true);
    let progress_box = gtk::Box::new(gtk::Orientation::Vertical, 0);
    area_box.add(&progress_box);
    progress_box.set_hexpand(true);
    progress_box.set_vexpand(false);

    progress_box.add(&scroll_video_button);
    progress_box.add(&progress);
    progress_box.set_border_width(5);

    let light_button = gtk::CheckButton::new_with_label("enable");
    light_button.clicked();
    light_button.connect_clicked(clone!(state, glarea; |_light_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.is_light= !state.is_light;
        glarea.queue_render();
    }));
    lightning_box.add(&light_button);

    let int_button = gtk::SpinButton::new_with_range(0.0, 1.0, 0.05);
    int_button.set_value(1.0);
    int_button.connect_property_value_notify(clone!(state, glarea; |int_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.int = int_button.get_value() as f32;
        glarea.queue_render();
    }));

    let int_label = gtk::Label::new("intensivity");
    int_box.add(&int_label);
    int_box.add(&int_button);

    let amb_button = gtk::SpinButton::new_with_range(0.0, 1.0, 0.05);
    amb_button.set_value(0.5);
    amb_button.connect_property_value_notify(clone!(state, glarea; |amb_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.amb = amb_button.get_value() as f32;
        glarea.queue_render();
    }));

    let amb_label = gtk::Label::new("ambience");
    amb_box.add(&amb_label);
    amb_box.add(&amb_button);

    let diff_button = gtk::SpinButton::new_with_range(0.0, 1.0, 0.05);
    diff_button.set_value(1.0);
    diff_button.connect_property_value_notify(clone!(state, glarea; |diff_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.diff = diff_button.get_value() as f32;
        glarea.queue_render();
    }));

    let diff_label = gtk::Label::new("diffuse");
    diff_box.add(&diff_label);
    diff_box.add(&diff_button);

    let spec_button = gtk::SpinButton::new_with_range(0.0, 1.0, 0.05);
    spec_button.set_value(0.8);
    spec_button.connect_property_value_notify(clone!(state, glarea; |spec_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.spec = spec_button.get_value() as f32;
        glarea.queue_render();
    }));

    let spec_label = gtk::Label::new("specaluraty");
    spec_box.add(&spec_label);
    spec_box.add(&spec_button);

    let texture_button = gtk::CheckButton::new_with_label("");
    texture_button.clicked();
    texture_button.connect_clicked(clone!(state, glarea; |_texture_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.is_texture= !state.is_texture;
        glarea.queue_render();
    }));

    let scale_button = gtk::Scale::new_with_range(gtk::Orientation::Horizontal, 0.0, 1.0, 0.05);
    scale_button.set_value(0.5);
    scale_button.connect_value_changed(clone!(state, glarea; |scale_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.scale = scale_button.get_value() as f32;
        glarea.queue_render();
    }));


    let color_button = gtk::ColorButton::new_with_rgba(
        &gdk::RGBA{red : 1.0, green : 1.0, blue : 1.0, alpha : 1.0});
    color_button.set_title("model`s colour");
    color_button.connect_color_set(clone!(state, glarea; |color_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.model_color = color_button.get_rgba();
        glarea.queue_render();
    }));

    let color_label = gtk::Label::new("colour");
    c_box.add(&color_label);
    c_box.add(&color_button);

    let back_button = gtk::ColorButton::new_with_rgba(
        &gdk::RGBA{red : 0.0, green : 0.0, blue : 0.0, alpha : 1.0});
    back_button.set_title("glarea`s color");
    back_button.set_name("background");
    back_button.connect_color_set(clone!(state, glarea; |back_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        state.back_color = back_button.get_rgba();
        glarea.queue_render();
    }));

    let back_label = gtk::Label::new("back");
    b_box.add(&back_label);
    b_box.add(&back_button);

    let menu = gtk::Menu::new();
    let open = gtk::MenuItem::new_with_label("Open");
    let exit = gtk::MenuItem::new_with_label("Exit");
    let about = gtk::MenuItem::new_with_label("About");
    menu.append(&open);
    menu.append(&exit);
    menu.append(&about);
    about.connect_activate(clone!(window; |_about| {
        let dialog = gtk::MessageDialog::new(Some(&window), gtk::DialogFlags::empty(), gtk::MessageType::Info,
                                gtk::ButtonsType::None, "use WASD and 1234");
        dialog.run();
    }));
    exit.connect_activate(|_exit| {
        gtk::main_quit();
    });

    let open_button = gtk::FileChooserButton::new("load model", gtk::FileChooserAction::Open);
    open_button.set_width_chars(19);
    open_button.set_filename(std::path::Path::new("/home/alexander/IdeaProjects/sport_visual/models/union.stl"));
    let open_dialog_filter = gtk::FileFilter::new();
    open_dialog_filter.add_pattern("*.stl");
    open_dialog_filter.set_name("*.stl");
    open_button.add_filter(&open_dialog_filter);
    open_button.connect_file_set(clone!(model_state, progress; |open_button| {
        use std::thread;
        progress.set_visible(true);
        let path = open_button.get_filename().unwrap();
        thread::spawn(clone!(path, model_state; || {
                model_state.lock().unwrap().is_render = true;
                let model = make_model(path.to_str().unwrap());
                if model_state.lock().unwrap().is_render {
                    model_state.lock().unwrap().model  = model.0;
                    model_state.lock().unwrap().dx  = (model.1).0;
                    model_state.lock().unwrap().dy  = (model.1).1;
                    model_state.lock().unwrap().dz  = (model.1).2;
                    model_state.lock().unwrap().max_scale  = (model.1).3;
                }
                model_state.lock().unwrap().is_render = false;
            }));
    }));
    let open_box = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let open_label = gtk::Label::new("STL-file");
    open_box.add(&open_label);
    open_box.add(&open_button);

    model_box.add(&open_box);
    model_box.add(&colours_box);

    let animation_progress = gtk::ProgressBar::new();
    animation_progress.set_text("running animation");
    animation_progress.set_fraction(0.0);
    animation_progress.set_pulse_step(0.25);
    animation_progress.set_show_text(true);
    let animation_progress_box = gtk::Box::new(gtk::Orientation::Vertical, 0);
    animation_progress_box.add(&animation_progress);
    animation_progress_box.set_border_width(5);

    let anime_control_box = gtk::Box::new(gtk::Orientation::Horizontal, 5);
    let video_back_button = gtk::Button::new_from_icon_name("media-seek-backward", gtk::IconSize::__Unknown(0));
    let pause_button = gtk::Button::new_from_icon_name("media-playback-pause", gtk::IconSize::__Unknown(0));
    let start_button = gtk::Button::new_from_icon_name("media-playback-start", gtk::IconSize::__Unknown(0));
    let video_forward_button = gtk::Button::new_from_icon_name("media-seek-forward", gtk::IconSize::__Unknown(0));
    anime_control_box.pack_start(&video_back_button, true, true, 3);
    anime_control_box.pack_start(&pause_button, true, true, 0);
    anime_control_box.pack_start(&start_button, true, true, 0);
    anime_control_box.pack_start(&video_forward_button, true, true, 3);
    start_button.set_sensitive(false);

    let csv_button = gtk::FileChooserButton::new("load animation", gtk::FileChooserAction::Open);
    csv_button.set_width_chars(19);
    let csv_dialog_filter = gtk::FileFilter::new();
    csv_dialog_filter.add_pattern("*.txt");
    csv_dialog_filter.set_name("*.txt");
    csv_button.add_filter(&csv_dialog_filter);

    let vbox = gtk::Box::new(gtk::Orientation::Vertical, 0);

    csv_button.connect_file_set(clone!(state, animation_progress, start_button, vbox, pause_button, scroll_video_button; |csv_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();

        let path = csv_button.get_filename().unwrap();
        let path_str = path.to_str().unwrap();

        let file = File::open(path_str).unwrap();
        let mut iter = std::io::BufReader::new(file).lines();

        let mut list: Vec<(f32, f32, f32, f32)> = Vec::new();
        let mut madgwick = Madgwick::new(0.0);
        let to_rad = (PI / 180.0) as f64;
        let to_degree = 180.0 / PI;

        let mut yaws: std::vec::Vec<(f64, f64)> = vec![];
        let mut pitches: std::vec::Vec<(f64, f64)> = vec![];
        let mut rolls: std::vec::Vec<(f64, f64)> = vec![];

        let mut last_pitch = 0.0f32;
        let mut last_yaw = 0.0f32;
        let mut last_roll = 0.0f32;
        let mut last_time = 0.0f32;
        let mut full_time_kr = 0.0f32;

        loop {
            let line = iter.next();
            if line.is_none() {
                break;
            }
            let accel_str = (line.unwrap().unwrap()).replace("Accel: ", "").replace(" g","");
            let mut accel_split = accel_str.split(", ").into_iter();
            let gyro_str = ((iter.next()).unwrap().unwrap()).replace("Gyro:  ", "").replace(" dps", "");
            let mut gyro_split = gyro_str.split(", ").into_iter();
            let mag_str = ((iter.next()).unwrap().unwrap()).replace("Mag:   ", "").replace(" uT", "");
            let mut mag_split = mag_str.split(", ").into_iter();
            let dtime_str = ((iter.next()).unwrap().unwrap()).replace("dTime: ", "").replace(" us", "");
            let time_str = ((iter.next()).unwrap().unwrap()).replace("sTime: ", "").replace(" s", "");
            let dtime = f64::from_str(&dtime_str).unwrap() / 1000000.0;
            let time = f64::from_str(&time_str).unwrap();

            let gyro = [
                 f64::from_str(gyro_split.next().unwrap()).unwrap() * to_rad,
                 f64::from_str(gyro_split.next().unwrap()).unwrap() * to_rad,
                -f64::from_str(gyro_split.next().unwrap()).unwrap() * to_rad,
            ];

            let accel = [
                 f64::from_str(accel_split.next().unwrap()).unwrap(),
                 f64::from_str(accel_split.next().unwrap()).unwrap(),
                -f64::from_str(accel_split.next().unwrap()).unwrap(),

            ];

            let mag = [
                 f64::from_str(mag_split.next().unwrap()).unwrap(),
                -f64::from_str(mag_split.next().unwrap()).unwrap(),
                -f64::from_str(mag_split.next().unwrap()).unwrap(),

            ];
            madgwick.update(&gyro, &accel, &mag, dtime);
            let (roll, pitch, yaw, _q_z) = madgwick.q.to_euler_angles();

            if time as f32 >= full_time_kr + 0.02{
                full_time_kr += 0.02;
                let pitch_kr = get_kr_val(last_time, last_pitch, time as f32, pitch, full_time_kr);
                let yaw_kr = get_kr_val(last_time, last_yaw, time as f32, yaw, full_time_kr);
                let roll_kr = get_kr_val(last_time, last_roll, time as f32, roll, full_time_kr);
                list.push((pitch_kr*to_degree, yaw_kr*to_degree, roll_kr*to_degree, 0.02f32));
            }
            last_time = time as f32;
            last_pitch = pitch;
            last_yaw = yaw;
            last_roll = roll;

            yaws.push((time, (yaw*to_degree) as f64));
            pitches.push((time, (pitch*to_degree) as f64));
            rolls.push((time, (roll*to_degree) as f64));
        }

        let s1: Plot = Plot::new(rolls).line_style(
            LineStyle::new().colour("#ff0000")).legend("roll".to_string());

        let s2: Plot = Plot::new(pitches).line_style(
            LineStyle::new().colour("#0000ff")).legend("pitch".to_string());

        let s3: Plot = Plot::new(yaws).line_style(
            LineStyle::new().colour("#00ff00")).legend("yaw".to_string());

        let v = ContinuousView::new()
            .add(s1)
            .add(s2)
            .add(s3)
            .x_label("Time (s)")
            .y_label("Angle (dgr)");

        Page::single(&v).save(path_str.replace(".txt", ".svg")).unwrap();

        state.anime_list = list;
        state.al_ind = 0;
        state.arx = state.rx;
        state.ary = state.ry;
        state.arz = state.rz;

        start_button.set_sensitive(false);
        start_button.set_visible(false);
        pause_button.set_sensitive(true);
        pause_button.set_visible(true);
        animation_progress.set_visible(true);
        scroll_video_button.set_visible(true);
        scroll_video_button.set_sensitive(true);
        scroll_video_button.set_range(0.0, state.anime_list.len() as f64);
        scroll_video_button.set_increments((state.anime_list.len() as f64 / 30.0).ceil(), 0.0);
        scroll_video_button.set_value(0.0);

        if state.playbin.is_some() {
            state.playbin.as_ref().unwrap().set_state(gst::State::Null).unwrap();
            vbox.foreach(|w| {
                vbox.remove(w);
            })
        }

        let (playbin, bus) = run_video(&(std::string::String::from("file://") + &path_str.replace(".txt", ".mp4")));
        state.playbin = playbin;
        state.bus = bus;

        let playbin = state.playbin.as_ref().unwrap();
        use gdk::WindowExt;

        let video_window = gtk::DrawingArea::new();

        let video_overlay = playbin
            .clone()
            .dynamic_cast::<gst_video::VideoOverlay>()
            .unwrap();

        video_window.connect_realize(move |video_window| {
            let video_overlay = &video_overlay;
            let gdk_window = video_window.get_window().unwrap();

            if !gdk_window.ensure_native() {
                println!("Can't create native window for widget");
                process::exit(-1);
            }

            let display_type_name = gdk_window.get_display().get_type().name();

            // Check if we're using X11 or ...
            if display_type_name == "GdkX11Display" {
                extern "C" {
                    pub fn gdk_x11_window_get_xid(
                        window: *mut glib::object::GObject,
                    ) -> *mut c_void;
                }

                #[allow(clippy::cast_ptr_alignment)]
                    unsafe {
                    let xid = gdk_x11_window_get_xid(gdk_window.as_ptr() as *mut _);
                    video_overlay.set_window_handle(xid as usize);
                }
            } else {
                println!("Add support for display type '{}'", display_type_name);
                process::exit(-1);
            }

        });
        vbox.pack_start(&video_window, true, true, 0);
        vbox.show_all();
        state.is_anime = true;
    }));

    scroll_video_button.connect_format_value(move |scroll_video_button, _|{
        let mut seconds = (scroll_video_button.get_value() / 50.0).round();
        let minutes = (seconds / 60.0).floor();
        seconds = seconds % 60.0;
        return format!("{}:{}", minutes, seconds);
    });

//    scroll_video_button.connect_value_changed(clone!(state; |scroll_video_button| {
//        let mut state = state.borrow_mut();
//        let state = state.as_mut().unwrap();
//        let seconds = (scroll_video_button.get_value() / 50.0).ceil();
//        if state.playbin.as_ref().unwrap().seek_simple(gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT,
//            seconds as u64 * gst::SECOND,).is_err() {
//                eprintln!("Seekition to {} failed", seconds);
//            }
//    }));

    let csv_box = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let csv_label = gtk::Label::new("Animation-file");
    csv_box.add(&csv_label);
    csv_box.add(&csv_button);

    pause_button.connect_clicked(clone!(state, start_button; |pause_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();

        match state.is_anime {
            true => {
                state.is_anime = false;
                start_button.set_sensitive(true);
                start_button.set_visible(true);
                pause_button.set_sensitive(false);
                pause_button.set_visible(false);
                state.playbin.as_ref().unwrap().change_state(gst::StateChange::PlayingToPaused).unwrap();
            },
            false => {},
        }
    }));

    start_button.connect_clicked(clone!(state, pause_button; |start_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();

        start_button.set_sensitive(false);
        start_button.set_visible(false);
        pause_button.set_sensitive(true);
        pause_button.set_visible(true);

        state.playbin.as_ref().unwrap().change_state(gst::StateChange::PausedToPlaying).unwrap();

        state.is_anime = true;
    }));

    video_forward_button.connect_clicked(clone!(state, start_button, scroll_video_button; |_video_forward_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        if state.is_anime || start_button.is_sensitive() {
            let mem = state.is_anime;
            state.is_anime = false;
            let n = 50.0 * 10.0; // 10 sec
            state.al_ind += n as usize;
            let mseconds = (scroll_video_button.get_value() * 20.0) as u64 + 10*1000;
            scroll_video_button.set_value(scroll_video_button.get_value() + n);
            if state.playbin.as_ref().unwrap().seek_simple(gst::SeekFlags::FLUSH,
            mseconds * gst::MSECOND,).is_err() {
                eprintln!("Seekition to {} failed", mseconds);
            }
            state.is_anime = mem;
            if !state.is_anime {
                state.playbin.as_ref().unwrap().set_state(gst::State::Paused).unwrap();
            } else {
                state.playbin.as_ref().unwrap().set_state(gst::State::Playing).unwrap();
            }
        }
    }));

    video_back_button.connect_clicked(clone!(state, start_button, scroll_video_button; |_video_back_button| {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        if state.is_anime || start_button.is_sensitive() {
            let mem = state.is_anime;
            state.is_anime = false;
            let n = 50.0 * 10.0; // 10 sec
            if state.al_ind < n as usize {
                state.al_ind = 0;
            } else {
                state.al_ind -= n as usize;
            }
            let mut mseconds = (scroll_video_button.get_value() * 20.0) as u64;
            scroll_video_button.set_value(scroll_video_button.get_value() - n);
            if mseconds < 10*1000 {
                mseconds = 0;
            } else {
                mseconds -= 10*1000;
            }
            if state.playbin.as_ref().unwrap().seek_simple(gst::SeekFlags::FLUSH,
            mseconds as u64 * gst::MSECOND,).is_err() {
                eprintln!("Seekition to {} failed", mseconds);
            }
            state.is_anime = mem;
            if !state.is_anime {
                state.playbin.as_ref().unwrap().set_state(gst::State::Paused).unwrap();
            } else {
                state.playbin.as_ref().unwrap().set_state(gst::State::Playing).unwrap();
            }
        }
    }));

    animation_box.add(&csv_box);
    animation_box.add(&anime_control_box);
    animation_box.add(&animation_progress_box);

    let open_texture = gtk::FileChooserButton::new("load texture", gtk::FileChooserAction::Open);
    open_texture.set_width_chars(19);
    open_texture.set_filename(std::path::Path::new("/home/alexander/IdeaProjects/sport_visual/textures/t2.jpg"));
    let open_texture_filter = gtk::FileFilter::new();
    open_texture_filter.add_pattern("*.jpg");
    open_texture_filter.set_name("*.jpg");
    open_texture.add_filter(&open_texture_filter);
    open_texture.connect_file_set(clone!(state_info; |open_texture| {
            let mut state_info = state_info.lock().unwrap();
            let state_info = state_info.as_mut().unwrap();
            let path = open_texture.get_filename().unwrap();
            let path_str = path.to_str().unwrap();
            let file = File::open(&path_str).unwrap();
            use std::io::Read;
            let mut reader = std::io::BufReader::new(file);
            let mut buf = Vec::new();
            let _length = reader.read_to_end(&mut buf);
            let image = image::load(
                Cursor::new(&buf.as_slice() as &dyn std::convert::AsRef<[u8]>),image::ImageFormat::Jpeg).unwrap().to_rgba();
            let image_dimensions = image.dimensions();
            let image = glium::texture::RawImage2d::from_raw_rgba_reversed(&image.into_raw(), image_dimensions);
            state_info.texture = glium::texture::Texture2d::new(&state_info.display, image).unwrap();
    }));

    let texture_box = gtk::Box::new(gtk::Orientation::Vertical, 1);
    let texture_label = gtk::Label::new("JPG-file");
    let texture_sub_box = gtk::Box::new(gtk::Orientation::Horizontal, 3);
    texture_sub_box.add(&texture_button);
    texture_sub_box.add(&open_texture);
    texture_box.add(&texture_label);
    texture_box.add(&texture_sub_box);

    model_box.add(&texture_box);

    open.connect_activate(clone!(window, model_state, progress, open_button; |_open| {
        let open_dialog = gtk::FileChooserDialog::new(Some("load model"),
                                             Some(&window), gtk::FileChooserAction::Open);
        let open_dialog_filter = gtk::FileFilter::new();
        open_dialog_filter.add_pattern("*.stl");
        open_dialog_filter.set_name("*.stl");
        open_dialog.add_filter(&open_dialog_filter);
        open_dialog.connect_file_activated(clone!(model_state, progress, open_button; |open_dialog| {
        use std::thread;
        progress.set_visible(true);
        let path = open_dialog.get_filename().unwrap();
        thread::spawn(clone!(path, model_state; || {
                model_state.lock().unwrap().is_render = true;
                let model = make_model(path.to_str().unwrap());
                if model_state.lock().unwrap().is_render {
                    model_state.lock().unwrap().model  = model.0;
                    model_state.lock().unwrap().dx  = (model.1).0;
                    model_state.lock().unwrap().dy  = (model.1).1;
                    model_state.lock().unwrap().dz  = (model.1).2;
                    model_state.lock().unwrap().max_scale  = (model.1).3;
                }
                model_state.lock().unwrap().is_render = false;
            }));
        open_button.set_filename(std::path::Path::new(path.to_str().unwrap()));
            open_dialog.destroy();
        }));
        open_dialog.run();
    }));

    let menu_bar = gtk::MenuBar::new();
    let file = gtk::MenuItem::new_with_label("File");
    file.set_submenu(Some(&menu));
    menu_bar.append(&file);

    window.connect_key_press_event(clone!(state, glarea; |_window, key| {
        let keyval = gdk::EventKey::get_keyval(&key);
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        match keyval {
            gdk::enums::key::Escape => gtk::main_quit(),
            gdk::enums::key::a => if state.is_light { state.tx -= 0.1 },
            gdk::enums::key::d => if state.is_light { state.tx += 0.1 },
            gdk::enums::key::s => if state.is_light { state.ty -= 0.1 },
            gdk::enums::key::w => if state.is_light { state.ty += 0.1 },
            gdk::enums::key::f => if state.is_light { state.tz -= 0.1 },
            gdk::enums::key::r => if state.is_light { state.tz += 0.1 },
            gdk::enums::key::_4 => state.rx -= 5.0,
            gdk::enums::key::_3 => state.rx += 5.0,
            gdk::enums::key::_2 => state.ry += 5.0,
            gdk::enums::key::_1 => state.ry -= 5.0,
            gdk::enums::key::_5 => state.rz += 5.0,
            gdk::enums::key::_6 => state.rz -= 5.0,
            _ => (),
        }
        glarea.queue_render();
        Inhibit(false)
    }));

    button_box.add(&model_frame);
    button_box.add(&lightning_frame);
    button_box.add(&animation_frame);
    area_sub_box.add(&glarea);
    area_sub_box.add(&vbox);
    hbox.add(&button_box);
    scale_box.add(&scale_button);
    hbox.add(&area_box);
    button_box.pack_start(&menu_bar, false, false, 0);
    window.add(&hbox);
    window.show_all();

    glarea.set_visible(true);
    progress.set_visible(false);
    animation_progress.set_visible(false);
    start_button.set_visible(false);

    scroll_video_button.set_visible(false);
    scroll_video_button.set_sensitive(false);

    vbox.hide();

    gtk::timeout_add(1, clone!(glarea; || {
        glarea.queue_render();
        return glib::Continue(true);
    }));

    gtk::timeout_add(1000, clone!(model_state, state, scroll_video_button; || {
        if !model_state.lock().unwrap().is_render {
            progress.set_visible(false);
        } else  {
            progress.pulse();
        }
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        let ind = scroll_video_button.get_value() as usize;
        if state.al_ind + 55 < ind || (state.al_ind > 55 && state.al_ind > ind + 55 ) {
            let mem = state.is_anime;
            state.is_anime = false;
//            let seconds = (ind as f64 / 50.0).round() as u64;
            let mseconds = scroll_video_button.get_value() * 20.0;
            if state.playbin.as_ref().unwrap().seek_simple(gst::SeekFlags::FLUSH,
            mseconds as u64 * gst::MSECOND,).is_err() {
                eprintln!("Seekition to {} failed", mseconds);
            }
            state.al_ind = ind;
            state.is_anime = mem;
            if state.is_anime {
                state.playbin.as_ref().unwrap().set_state(gst::State::Playing).unwrap();
            } else {
                state.playbin.as_ref().unwrap().set_state(gst::State::Paused).unwrap();
            }
        }
        scroll_video_button.set_value(state.al_ind as f64);
        if state.al_ind > state.anime_list.len() {
            animation_progress.set_visible(false);
            scroll_video_button.set_visible(false);
            scroll_video_button.set_sensitive(false);
            vbox.hide();
            state.playbin.as_ref().unwrap().set_state(gst::State::Null).unwrap();
        }
            return glib::Continue(true);
    }));

//    use std::time::SystemTime;
//    let mut now = SystemTime::now();

    extern crate periodic;
    use std::time::Duration;

    let mut planner = periodic::Planner::new();
    planner.add(clone!(state; || {
        let mut state = state.lock().unwrap();
        let state = state.as_mut().unwrap();
        if state.is_anime {
            let ind = state.al_ind;
            if ind >= state.anime_list.len() {
                state.is_anime = false;
                state.al_ind += 1;
            } else {
                let angles = state.anime_list[ind];
                state.rx = angles.0 + state.arx;
                state.ry = angles.1 + state.ary;
                state.rz = angles.2 + state.arz;
                state.al_ind = ind + 1;
            }
        }
    }), periodic::Every::new(Duration::from_millis(20)), );
    planner.start();

    gtk::main();
}