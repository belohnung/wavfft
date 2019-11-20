use clap::Arg;
use hound::{SampleFormat, WavReader};
use nannou::event::Update;
use nannou::geom::pt2;
use nannou::{App, Frame};
use nannou_audio::Buffer;
use rustfft::num_complex::Complex32;
use rustfft::FFT;
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};

pub const AUDIO_BUFFER_SIZE: usize = 2048;
pub const FFT_SIZE: usize = 512;

fn main() {
    nannou::app(model).update(update).simple_window(view).run();
}

// A function that renders the given `Audio` to the given `Buffer`.
// In this case we play a simple sine wave at the audio's current frequency in `hz`.
fn audio(audio: &mut ExactStreamer<f32>, buffer: &mut Buffer) {
    let mut buf = vec![0.0; buffer.len_frames()];
    audio.fill(&mut buf);
    buffer
        .frames_mut()
        .zip(buf.into_iter())
        .for_each(|(frame, buf_sample)| {
            for frame_channel_sample in frame {
                *frame_channel_sample = buf_sample;
            }
        });

    println!(
        "A callback {}hz {}frames",
        buffer.sample_rate(),
        buffer.len_frames()
    );
}

/// Used as a kind of `BufReader` for input from a `Receiver<Vec<T>>` to read an exact number of `T`s by buffering and not peeking in the channel
pub struct ExactStreamer<T> {
    /// stores data if the callback's slice's size is not a multiple of `GENERATOR_BUFFER_SIZE`
    remainder: Vec<T>,
    remainder_len: usize,
    /// receives audio from the worker thread
    receiver: crossbeam_channel::Receiver<Vec<T>>,
}

impl<T> ExactStreamer<T>
where
    T: Copy + Default,
{
    pub fn new(
        remainder_buffer_size: usize,
        receiver: crossbeam_channel::Receiver<Vec<T>>,
    ) -> ExactStreamer<T> {
        ExactStreamer {
            remainder: vec![T::default(); remainder_buffer_size],
            remainder_len: 0,
            receiver,
        }
    }

    pub fn fill(&mut self, out: &mut [T]) {
        let mut i = self.remainder_len.min(out.len());

        out[..i].copy_from_slice(&self.remainder[..i]);

        // move old data to index 0 for next read
        self.remainder.copy_within(i..self.remainder_len, 0);
        self.remainder_len -= i;

        while i < out.len() {
            let generated = self
                .receiver
                .recv()
                .expect("Stream channel unexpectedly disconnected");
            if generated.len() > out.len() - i {
                let left = out.len() - i;
                out[i..].copy_from_slice(&generated[..left]);

                self.remainder_len = generated.len() - left;

                let vec_len = self.remainder.len();
                if vec_len < self.remainder_len {
                    self.remainder
                        .extend(std::iter::repeat(T::default()).take(self.remainder_len - vec_len));
                }

                self.remainder[..self.remainder_len].copy_from_slice(&generated[left..]);
                break;
            } else {
                out[i..(i + generated.len())].copy_from_slice(&generated);
                i += generated.len();
            }
        }
    }
}

struct Model {
    stream: nannou_audio::Stream<ExactStreamer<f32>>,
    latest_fft: Arc<Mutex<Vec<f32>>>,
}

fn model(_app: &App) -> Model {
    let clap = clap::App::new("wavfft xd")
        .arg(
            Arg::with_name("input")
                .short("i")
                .takes_value(true)
                .required(true),
        )
        .get_matches();

    let audio_host = nannou_audio::Host::new();
    let (audio_sender, audio_receiver) = crossbeam_channel::bounded(3);

    let input = clap.value_of("input").unwrap();

    let (mut audio_samples, sample_rate) = load_audio(input);

    let stream = audio_host
        .new_output_stream(ExactStreamer::new(AUDIO_BUFFER_SIZE, audio_receiver))
        .frames_per_buffer(AUDIO_BUFFER_SIZE)
        .sample_rate(sample_rate)
        .render(audio)
        .build()
        .unwrap();

    println!("loaded audio");

    let latest_fft = Arc::new(Mutex::new(vec![1.0]));

    std::thread::spawn({
        let latest_fft = latest_fft.clone();
        move || {
            let mut audio_buf = vec![0.0; AUDIO_BUFFER_SIZE];
            let mut fft_buf = vec![0.0; FFT_SIZE];
            loop {
                audio_buf
                    .iter_mut()
                    .zip(audio_samples.drain(..AUDIO_BUFFER_SIZE))
                    .for_each(|(audio_buf, audio_in)| {
                        *audio_buf = audio_in as f32 / std::i16::MAX as f32
                    });

                audio_fft(&mut audio_buf[..FFT_SIZE], &mut fft_buf);
                {
                    *latest_fft.lock().unwrap() = fft_buf.clone();
                }

                println!("A sending");

                if audio_sender.send(audio_buf.clone()).is_err() {
                    break;
                }
            }

            println!("A stopped");
        }
    });

    Model { stream, latest_fft }
}

fn update(app: &App, _model: &mut Model, _update: Update) {
    println!("{}", app.time);
}

fn view(app: &App, model: &Model, frame: &Frame) {
    let draw = app.draw();

    draw.background().rgb(0.2, 0.2, 0.2);

    let win = app.window_rect();

    let points = {
        let frequencies = &*model.latest_fft.lock().unwrap();
        let frequencies = &frequencies[..frequencies.len() / 2];

        let vertical_map = |mag: f32| -> f32 { (mag.exp() - 1.0) * 0.2 };
        let horizontal_map = |x: f32| -> f32 { x.powf(1.2) };

        let len = frequencies.len();
        frequencies
            .iter()
            .enumerate()
            .map(|(i, frequency_magnitude)| {
                pt2(
                    horizontal_map(i as f32 / len as f32) * (win.right() - win.left()) + win.left(),
                    vertical_map(*frequency_magnitude) * (win.top() - win.bottom()) + win.bottom(),
                )
            })
            .collect::<Vec<_>>()
    };

    draw.polyline()
        .weight(5.0)
        .points(points)
        .rgb(0.4, 0.5, 0.8);

    draw.to_frame(app, &frame).unwrap();
}

pub fn audio_fft(input: &[f32], output: &mut [f32]) {
    /// https://github.com/rust-lang/rust/pull/65092
    /// Checks if `n` supplied integer is 2^floor(log2(`n`))
    pub const fn is_pot(n: u32) -> bool {
        ((n.wrapping_sub(1)) & n == 0) & !(n == 0)
    }

    assert_eq!(input.len(), output.len());

    if !is_pot(input.len() as u32) {
        eprintln!("supplied non-POT audio data: {}", input.len());
        return;
    }

    let mut input = input
        .into_iter()
        .map(|sample| Complex32 {
            re: *sample,
            im: 0.0,
        })
        .collect::<Vec<Complex32>>();
    let mut out = input.clone();

    rustfft::algorithm::Radix4::new(input.len(), false).process(&mut input, &mut out);

    out.into_iter().enumerate().for_each(|(i, complex)| {
        output[i] = (complex.re * complex.re + complex.im * complex.im).sqrt()
    });
}

pub fn load_audio(path: &str) -> (Vec<i16>, u32) {
    let wav_in = WavReader::new(BufReader::new(
        File::open(path).expect("failed to read sound input file"),
    ))
    .expect("failed to parse wav");
    if wav_in.spec().channels != 1 {
        println!("converting {} channel wav to mono", wav_in.spec().channels);
    }

    let spec = wav_in.spec();

    (
        match spec.sample_format {
            SampleFormat::Float => wav_in
                .into_samples::<f32>()
                .map(|v| v.unwrap())
                .collect::<Vec<f32>>()
                .chunks(spec.channels as usize)
                .map(|channels| channels.iter().sum::<f32>() / channels.len() as f32)
                .collect::<Vec<f32>>(),
            SampleFormat::Int => match wav_in.spec().bits_per_sample {
                16 => wav_in
                    .into_samples::<i16>()
                    .map(|v| v.unwrap() as f32 / std::i16::MAX as f32)
                    .collect::<Vec<f32>>()
                    .chunks(spec.channels as usize)
                    .map(|channels| channels.iter().sum::<f32>() / channels.len() as f32)
                    .collect::<Vec<f32>>(),
                32 => wav_in
                    .into_samples::<i32>()
                    .map(|v| v.unwrap() as f32 / std::i32::MAX as f32)
                    .collect::<Vec<f32>>()
                    .chunks(spec.channels as usize)
                    .map(|channels| channels.iter().sum::<f32>() / channels.len() as f32)
                    .collect::<Vec<f32>>(),
                _ => unreachable!(),
            },
        }
        .into_iter()
        .map(|f| {
            (f * std::i16::MAX as f32)
                .min(std::i16::MAX as f32)
                .max(std::i16::MIN as f32) as i16
        })
        .collect::<Vec<i16>>(),
        spec.sample_rate,
    )
}
