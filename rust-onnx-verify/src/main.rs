use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use image::{GenericImageView, RgbImage, imageops::FilterType};
use ndarray::{Array, ArrayD, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde::Deserialize;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

#[derive(Debug, Deserialize)]
struct Manifest {
    tensors: Tensors,
    onnx: OnnxPaths,
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Tensors {
    input_ids: TensorFile,
    text_ref: TensorFile,
    pixel_values: TensorFile,
    pixel_attention_mask: TensorFile,
    pos_embed: TensorFile,
    image_ref: TensorFile,
}

#[derive(Debug, Deserialize)]
struct TensorFile {
    file: PathBuf,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct OnnxPaths {
    text: PathBuf,
    image: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Expected {
    text_image_cosine: f32,
}

#[derive(Debug, Deserialize)]
struct LogitParams {
    logit_scale_exp: f32,
    logit_bias: f32,
}

struct RuntimePaths {
    text_onnx: PathBuf,
    image_onnx: PathBuf,
    vision_pos_embedding: PathBuf,
    logit_params: PathBuf,
    tokenizer_json: PathBuf,
}

struct ImageInputs {
    pixel_values: ArrayD<f32>,
    pixel_attention_mask: ArrayD<i32>,
    spatial_height: usize,
    spatial_width: usize,
}

impl RuntimePaths {
    fn from_repo_root(root: &Path) -> Self {
        let variant = std::env::var("FGCLIP2_RUNTIME_VARIANT").unwrap_or_default();
        let use_dynamic_int8 = matches!(
            variant.as_str(),
            "dynamic-int8" | "dynamic_int8" | "int8" | "quantized"
        );
        let use_split_text = matches!(
            variant.as_str(),
            "lowmem" | "low-memory" | "split-text" | "split_text"
        );
        let use_quantized_image = use_dynamic_int8 || matches!(variant.as_str(), "lowmem" | "low-memory");
        let text_default = if use_dynamic_int8 {
            ".onnx-wrapper-test/quantized/fgclip2_text_short_b1_s64_dynamic_int8.onnx"
        } else if use_split_text {
            ".onnx-wrapper-test/split/fgclip2_text_short_b1_s64_token_embeds.onnx"
        } else {
            ".onnx-wrapper-test/fgclip2_text_short_b1_s64.onnx"
        };
        let image_default = if use_quantized_image {
            ".onnx-wrapper-test/quantized/fgclip2_image_core_posin_dynamic_int8.onnx"
        } else {
            ".onnx-wrapper-test/fgclip2_image_core_posin_dynamic.onnx"
        };

        Self {
            text_onnx: env_path_or_default(root, "FGCLIP2_TEXT_ONNX", text_default),
            image_onnx: env_path_or_default(root, "FGCLIP2_IMAGE_ONNX", image_default),
            vision_pos_embedding: root
                .join(".onnx-wrapper-test/assets/vision_pos_embedding_16x16x768_f32.bin"),
            logit_params: root.join(".onnx-wrapper-test/assets/logit_params.json"),
            tokenizer_json: root.join("models/fg-clip2-base/tokenizer.json"),
        }
    }
}

fn env_path_or_default(root: &Path, env_key: &str, default_relative: &str) -> PathBuf {
    if let Some(value) = std::env::var_os(env_key) {
        let path = PathBuf::from(value);
        if path.is_absolute() {
            path
        } else {
            root.join(path)
        }
    } else {
        root.join(default_relative)
    }
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|arg| arg == "run") {
        return run_end_to_end(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "run-text") {
        return run_text_only(&args[1..]);
    }
    if args.first().is_some_and(|arg| arg == "run-batch") {
        return run_batch(&args[1..]);
    }

    let manifest_path = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../.onnx-wrapper-test/fixtures/manifest.json"));
    let manifest_path = absolutize(&manifest_path)?;
    let repo_root = manifest_path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .context("manifest must be inside .onnx-wrapper-test/fixtures")?
        .to_path_buf();

    let manifest: Manifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )
    .context("failed to parse manifest JSON")?;

    println!("manifest: {}", manifest_path.display());
    println!("reference text/image cosine: {:.8}", manifest.expected.text_image_cosine);

    let text_input = read_i64_array(&repo_root, &manifest.tensors.input_ids)?;
    let text_ref = read_f32_array(&repo_root, &manifest.tensors.text_ref)?;
    let image_pixel_values = read_f32_array(&repo_root, &manifest.tensors.pixel_values)?;
    let image_mask = read_i32_array(&repo_root, &manifest.tensors.pixel_attention_mask)?;
    let image_pos_embed = read_f32_array(&repo_root, &manifest.tensors.pos_embed)?;
    let image_ref = read_f32_array(&repo_root, &manifest.tensors.image_ref)?;

    let text_out = run_text(
        &repo_root.join(&manifest.onnx.text),
        &text_input,
    )?;
    report("text", &text_out, &text_ref)?;

    let image_out = run_image(
        &repo_root.join(&manifest.onnx.image),
        &image_pixel_values,
        &image_mask,
        &image_pos_embed,
    )?;
    report("image", &image_out, &image_ref)?;

    let text_slice = text_out.as_slice().context("text output is not contiguous")?;
    let image_slice = image_out.as_slice().context("image output is not contiguous")?;
    println!(
        "rust text/image cosine: {:.8}",
        dot(text_slice, image_slice)
    );

    Ok(())
}

fn run_text_only(args: &[std::ffi::OsString]) -> Result<()> {
    if args.is_empty() {
        bail!("usage: fgclip2-onnx-verify run-text <query>");
    }
    let repo_root = std::env::current_dir()?;
    let runtime = RuntimePaths::from_repo_root(&repo_root);
    let query = args[0].to_string_lossy().to_lowercase();
    println!("query: {query}");

    let input_ids = tokenize_query(&runtime.tokenizer_json, &query, 64)?;
    let text_features = run_text(&runtime.text_onnx, &input_ids)?;
    let text = text_features
        .as_slice()
        .context("text output is not contiguous")?;
    println!("text norm: {:.6}", l2(text));
    if let Some(dump_dir) = std::env::var_os("FGCLIP2_DUMP_DIR").map(PathBuf::from) {
        fs::create_dir_all(&dump_dir)
            .with_context(|| format!("failed to create {}", dump_dir.display()))?;
        dump_array_i64(&dump_dir.join("input_ids_i64.bin"), &input_ids)?;
        write_f32_slice(&dump_dir.join("text_features_f32.bin"), text)?;
        println!("dumped text features -> {}", dump_dir.display());
    }
    Ok(())
}

fn run_batch(args: &[std::ffi::OsString]) -> Result<()> {
    if args.len() < 2 {
        bail!(
            "usage: fgclip2-onnx-verify run-batch <max-patches> <image-path> [image-path ...]"
        );
    }
    let repo_root = std::env::current_dir()?;
    let max_patches = args[0]
        .to_string_lossy()
        .parse::<usize>()
        .context("run-batch first argument must be max-patches")?;
    let paths = args[1..]
        .iter()
        .map(|arg| absolutize(&PathBuf::from(arg)))
        .collect::<Result<Vec<_>>>()?;
    let runtime = RuntimePaths::from_repo_root(&repo_root);

    let start = Instant::now();
    let mut images = Vec::with_capacity(paths.len());
    let mut pos_embeds = Vec::with_capacity(paths.len());
    let base_pos = read_f32_vec(&runtime.vision_pos_embedding)?;
    for path in &paths {
        let image = preprocess_siglip2_image(path, max_patches)
            .with_context(|| format!("failed to preprocess {}", path.display()))?;
        let pos_embed = make_pos_embed_no_antialias(
            &base_pos,
            image.spatial_height,
            image.spatial_width,
            max_patches,
        )?;
        images.push(image);
        pos_embeds.push(pos_embed);
    }
    let preprocess_elapsed = start.elapsed();

    let pixel_values = stack_image_pixels(&images)?;
    let pixel_attention_mask = stack_image_masks(&images)?;
    let pos_embed = stack_f32_arrays(&pos_embeds, &[paths.len(), max_patches, 768])?;

    let mut session = load_session(&runtime.image_onnx)?;
    let pixel_values_input = TensorRef::from_array_view(pixel_values.view())?;
    let pixel_attention_mask_input = TensorRef::from_array_view(pixel_attention_mask.view())?;
    let pos_embed_input = TensorRef::from_array_view(pos_embed.view())?;
    let infer_start = Instant::now();
    let outputs = session.run(ort::inputs![
        "pixel_values" => pixel_values_input,
        "pixel_attention_mask" => pixel_attention_mask_input,
        "pos_embed" => pos_embed_input,
    ])?;
    let image_features = outputs["image_features"]
        .try_extract_array::<f32>()?
        .to_owned();
    let infer_elapsed = infer_start.elapsed();

    println!("batch size: {}", paths.len());
    println!("max patches: {max_patches}");
    println!("preprocess total: {:.2?}", preprocess_elapsed);
    println!("image inference total: {:.2?}", infer_elapsed);
    println!("image inference per image: {:.2?}", infer_elapsed / paths.len() as u32);
    println!("output shape: {:?}", image_features.shape());
    for (index, path) in paths.iter().enumerate() {
        let feature = image_features.index_axis(ndarray::Axis(0), index);
        println!(
            "{index}: norm={:.6} {}",
            l2(feature.as_slice().context("image output row is not contiguous")?),
            path.display()
        );
    }
    Ok(())
}

fn run_end_to_end(args: &[std::ffi::OsString]) -> Result<()> {
    if args.len() < 2 {
        bail!(
            "usage: fgclip2-onnx-verify run <image-path> <query> [max-patches]\n\
             example: cargo run --release --manifest-path .\\rust-onnx-verify\\Cargo.toml -- run .\\images\\x.jpg 山 1024"
        );
    }

    let repo_root = std::env::current_dir()?;
    let image_path = absolutize(&PathBuf::from(&args[0]))?;
    let query = args[1].to_string_lossy().to_lowercase();
    let max_patches = args
        .get(2)
        .and_then(|value| value.to_string_lossy().parse::<usize>().ok())
        .unwrap_or_else(|| determine_max_patches_for_image(&image_path).unwrap_or(256));

    let runtime = RuntimePaths::from_repo_root(&repo_root);
    println!("image: {}", image_path.display());
    println!("query: {query}");
    println!("max patches: {max_patches}");

    let input_ids = tokenize_query(&runtime.tokenizer_json, &query, 64)?;
    let image_inputs = preprocess_siglip2_image(&image_path, max_patches)?;
    let base_pos = read_f32_vec(&runtime.vision_pos_embedding)?;
    let pos_embed = make_pos_embed_no_antialias(
        &base_pos,
        image_inputs.spatial_height,
        image_inputs.spatial_width,
        max_patches,
    )?;

    let text_features = run_text(&runtime.text_onnx, &input_ids)?;
    let image_features = run_image(
        &runtime.image_onnx,
        &image_inputs.pixel_values,
        &image_inputs.pixel_attention_mask,
        &pos_embed,
    )?;
    let text = text_features.as_slice().context("text output is not contiguous")?;
    let image = image_features.as_slice().context("image output is not contiguous")?;

    let cosine = dot(text, image);
    let logit_params: LogitParams =
        serde_json::from_slice(&fs::read(&runtime.logit_params).with_context(|| {
            format!("failed to read {}", runtime.logit_params.display())
        })?)?;
    let logit = cosine * logit_params.logit_scale_exp + logit_params.logit_bias;
    println!("cosine: {cosine:.8}");
    println!("logit:  {logit:.8}");
    if let Some(dump_dir) = std::env::var_os("FGCLIP2_DUMP_DIR").map(PathBuf::from) {
        fs::create_dir_all(&dump_dir)
            .with_context(|| format!("failed to create {}", dump_dir.display()))?;
        dump_array_f32(&dump_dir.join("pixel_values_f32.bin"), &image_inputs.pixel_values)?;
        dump_array_i32(
            &dump_dir.join("pixel_attention_mask_i32.bin"),
            &image_inputs.pixel_attention_mask,
        )?;
        dump_array_i64(&dump_dir.join("input_ids_i64.bin"), &input_ids)?;
        dump_array_f32(&dump_dir.join("pos_embed_f32.bin"), &pos_embed)?;
        write_f32_slice(&dump_dir.join("text_features_f32.bin"), text)?;
        write_f32_slice(&dump_dir.join("image_features_f32.bin"), image)?;
        println!("dumped features -> {}", dump_dir.display());
    }
    Ok(())
}

fn run_text(model_path: &Path, input_ids: &ArrayD<i64>) -> Result<ArrayD<f32>> {
    let mut session = load_session(model_path)?;
    let token_embedding_path = text_token_embedding_path_for(model_path);

    let start = Instant::now();
    let outputs = if let Some(token_embedding_path) = token_embedding_path {
        let token_embeds = gather_text_token_embeddings(&token_embedding_path, input_ids)?;
        let token_embeds = TensorRef::from_array_view(token_embeds.view())?;
        session.run(ort::inputs!["token_embeds" => token_embeds])?
    } else {
        let input_ids = TensorRef::from_array_view(input_ids.view())?;
        session.run(ort::inputs!["input_ids" => input_ids])?
    };
    let elapsed = start.elapsed();
    let features = outputs["text_features"]
        .try_extract_array::<f32>()?
        .to_owned();
    println!("text inference: {:.2?}", elapsed);
    Ok(features)
}

fn text_token_embedding_path_for(model_path: &Path) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("FGCLIP2_TEXT_TOKEN_EMBEDDING").map(PathBuf::from) {
        return Some(path);
    }
    let file_name = model_path.file_name()?.to_string_lossy();
    if !file_name.contains("token_embeds") {
        return None;
    }
    let onnx_dir = model_path.parent()?.parent()?;
    Some(onnx_dir.join("assets/text_token_embedding_256000x768_f16.bin"))
}

fn gather_text_token_embeddings(
    embedding_path: &Path,
    input_ids: &ArrayD<i64>,
) -> Result<ArrayD<f32>> {
    let shape = input_ids.shape();
    if shape.len() != 2 {
        bail!("input_ids must have shape [B,S], got {:?}", shape);
    }
    let input_ids = input_ids.as_slice().context("input ids are not contiguous")?;
    let dtype = text_token_embedding_dtype(embedding_path);
    let row_bytes = dtype.bytes_per_value() * 768;
    let token_count = fs::metadata(embedding_path)
        .with_context(|| format!("failed to stat {}", embedding_path.display()))?
        .len()
        / row_bytes as u64;
    let mut file = fs::File::open(embedding_path)
        .with_context(|| format!("failed to open {}", embedding_path.display()))?;
    let mut row_bytes_buffer = vec![0u8; row_bytes];
    let mut values = vec![0.0f32; input_ids.len() * 768];

    for (token_index, token_id) in input_ids.iter().enumerate() {
        if *token_id < 0 || *token_id as u64 >= token_count {
            bail!(
                "token id {token_id} is outside embedding table with {token_count} rows"
            );
        }
        file.seek(SeekFrom::Start(*token_id as u64 * row_bytes as u64))?;
        file.read_exact(&mut row_bytes_buffer)?;
        let output = &mut values[token_index * 768..(token_index + 1) * 768];
        match dtype {
            TextTokenEmbeddingDtype::F16 => {
                for (value, bytes) in output.iter_mut().zip(row_bytes_buffer.chunks_exact(2)) {
                    *value = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
                }
            }
            TextTokenEmbeddingDtype::F32 => {
                for (value, bytes) in output.iter_mut().zip(row_bytes_buffer.chunks_exact(4)) {
                    *value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                }
            }
        }
    }

    Ok(Array::from_shape_vec(
        IxDyn(&[shape[0], shape[1], 768]),
        values,
    )?)
}

#[derive(Clone, Copy)]
enum TextTokenEmbeddingDtype {
    F16,
    F32,
}

impl TextTokenEmbeddingDtype {
    fn bytes_per_value(self) -> usize {
        match self {
            TextTokenEmbeddingDtype::F16 => 2,
            TextTokenEmbeddingDtype::F32 => 4,
        }
    }
}

fn text_token_embedding_dtype(path: &Path) -> TextTokenEmbeddingDtype {
    if std::env::var("FGCLIP2_TEXT_TOKEN_EMBEDDING_DTYPE")
        .is_ok_and(|value| value.eq_ignore_ascii_case("f32"))
        || path.to_string_lossy().contains("_f32")
    {
        TextTokenEmbeddingDtype::F32
    } else {
        TextTokenEmbeddingDtype::F16
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;

    let f32_bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut fraction = fraction as u32;
            let mut exponent = -14i32;
            while fraction & 0x0400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            fraction &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (fraction << 13)
        }
        0x1f => sign | 0x7f80_0000 | ((fraction as u32) << 13),
        _ => sign | (((exponent as u32) + 112) << 23) | ((fraction as u32) << 13),
    };
    f32::from_bits(f32_bits)
}

fn run_image(
    model_path: &Path,
    pixel_values: &ArrayD<f32>,
    pixel_attention_mask: &ArrayD<i32>,
    pos_embed: &ArrayD<f32>,
) -> Result<ArrayD<f32>> {
    let mut session = load_session(model_path)?;
    let pixel_values = TensorRef::from_array_view(pixel_values.view())?;
    let pixel_attention_mask = TensorRef::from_array_view(pixel_attention_mask.view())?;
    let pos_embed = TensorRef::from_array_view(pos_embed.view())?;

    let start = Instant::now();
    let outputs = session.run(ort::inputs![
        "pixel_values" => pixel_values,
        "pixel_attention_mask" => pixel_attention_mask,
        "pos_embed" => pos_embed,
    ])?;
    let elapsed = start.elapsed();
    let features = outputs["image_features"]
        .try_extract_array::<f32>()?
        .to_owned();
    println!("image inference: {:.2?}", elapsed);
    Ok(features)
}

fn tokenize_query(tokenizer_path: &Path, query: &str, max_len: usize) -> Result<ArrayD<i64>> {
    let mut tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|error| anyhow!("{error}"))
        .with_context(|| format!("failed to load tokenizer {}", tokenizer_path.display()))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_len,
            ..Default::default()
        }))
        .map_err(|error| anyhow!("{error}"))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(max_len),
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "<pad>".to_owned(),
        ..Default::default()
    }));

    let encoding = tokenizer
        .encode(query, true)
        .map_err(|error| anyhow!("{error}"))?;
    let ids = encoding
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();
    Ok(Array::from_shape_vec(IxDyn(&[1, max_len]), ids)?)
}

fn preprocess_siglip2_image(image_path: &Path, max_patches: usize) -> Result<ImageInputs> {
    let image = load_rgb_image(image_path)?;
    let (original_width, original_height) = image.dimensions();
    let (target_height, target_width) =
        get_image_size_for_max_num_patches(original_height as usize, original_width as usize, 16, max_patches);
    let resized = image::imageops::resize(
        &image,
        target_width as u32,
        target_height as u32,
        FilterType::Triangle,
    );
    let spatial_height = target_height / 16;
    let spatial_width = target_width / 16;
    let valid_patches = spatial_height * spatial_width;
    if valid_patches > max_patches {
        bail!("internal error: {valid_patches} valid patches > max_patches {max_patches}");
    }

    let mut pixel_values = vec![0.0f32; max_patches * 16 * 16 * 3];
    for patch_y in 0..spatial_height {
        for patch_x in 0..spatial_width {
            let patch_index = patch_y * spatial_width + patch_x;
            let patch_base = patch_index * 16 * 16 * 3;
            let mut dst = patch_base;
            for y in 0..16 {
                for x in 0..16 {
                    let pixel = resized.get_pixel((patch_x * 16 + x) as u32, (patch_y * 16 + y) as u32);
                    for channel in 0..3 {
                        pixel_values[dst] = pixel[channel] as f32 / 127.5 - 1.0;
                        dst += 1;
                    }
                }
            }
        }
    }

    let mut mask = vec![0i32; max_patches];
    for item in mask.iter_mut().take(valid_patches) {
        *item = 1;
    }

    Ok(ImageInputs {
        pixel_values: Array::from_shape_vec(IxDyn(&[1, max_patches, 768]), pixel_values)?,
        pixel_attention_mask: Array::from_shape_vec(IxDyn(&[1, max_patches]), mask)?,
        spatial_height,
        spatial_width,
    })
}

fn stack_image_pixels(images: &[ImageInputs]) -> Result<ArrayD<f32>> {
    let batch = images.len();
    let max_patches = images
        .first()
        .context("cannot stack an empty image batch")?
        .pixel_values
        .shape()[1];
    let mut values = Vec::with_capacity(batch * max_patches * 768);
    for image in images {
        if image.pixel_values.shape() != [1, max_patches, 768] {
            bail!("all image pixel arrays must have shape [1,{max_patches},768]");
        }
        values.extend_from_slice(image.pixel_values.as_slice().context("pixel array is not contiguous")?);
    }
    Ok(Array::from_shape_vec(IxDyn(&[batch, max_patches, 768]), values)?)
}

fn stack_image_masks(images: &[ImageInputs]) -> Result<ArrayD<i32>> {
    let batch = images.len();
    let max_patches = images
        .first()
        .context("cannot stack an empty image batch")?
        .pixel_attention_mask
        .shape()[1];
    let mut values = Vec::with_capacity(batch * max_patches);
    for image in images {
        if image.pixel_attention_mask.shape() != [1, max_patches] {
            bail!("all image mask arrays must have shape [1,{max_patches}]");
        }
        values.extend_from_slice(
            image
                .pixel_attention_mask
                .as_slice()
                .context("mask array is not contiguous")?,
        );
    }
    Ok(Array::from_shape_vec(IxDyn(&[batch, max_patches]), values)?)
}

fn stack_f32_arrays(arrays: &[ArrayD<f32>], shape: &[usize; 3]) -> Result<ArrayD<f32>> {
    let mut values = Vec::with_capacity(shape.iter().product());
    for array in arrays {
        if array.shape() != [1, shape[1], shape[2]] {
            bail!("all arrays must have shape [1,{},{}]", shape[1], shape[2]);
        }
        values.extend_from_slice(array.as_slice().context("array is not contiguous")?);
    }
    Ok(Array::from_shape_vec(IxDyn(shape), values)?)
}

fn load_rgb_image(image_path: &Path) -> Result<RgbImage> {
    Ok(image::open(image_path)
        .with_context(|| format!("failed to open image {}", image_path.display()))?
        .to_rgb8())
}

fn get_image_size_for_max_num_patches(
    image_height: usize,
    image_width: usize,
    patch_size: usize,
    max_num_patches: usize,
) -> (usize, usize) {
    fn scaled_size(scale: f64, size: usize, patch_size: usize) -> usize {
        let scaled = size as f64 * scale;
        let patched = (scaled / patch_size as f64).ceil() as usize * patch_size;
        patched.max(patch_size)
    }

    let eps = 1e-5f64;
    let mut scale_min = eps / 10.0;
    let mut scale_max = 100.0;
    while scale_max - scale_min >= eps {
        let scale = (scale_min + scale_max) / 2.0;
        let target_height = scaled_size(scale, image_height, patch_size);
        let target_width = scaled_size(scale, image_width, patch_size);
        let num_patches = (target_height / patch_size) * (target_width / patch_size);
        if num_patches <= max_num_patches {
            scale_min = scale;
        } else {
            scale_max = scale;
        }
    }
    (
        scaled_size(scale_min, image_height, patch_size),
        scaled_size(scale_min, image_width, patch_size),
    )
}

fn make_pos_embed_no_antialias(
    base_pos: &[f32],
    target_height: usize,
    target_width: usize,
    max_patches: usize,
) -> Result<ArrayD<f32>> {
    let source_height = 16usize;
    let source_width = 16usize;
    let channels = 768usize;
    if base_pos.len() != source_height * source_width * channels {
        bail!("unexpected vision position embedding length: {}", base_pos.len());
    }

    let mut output = vec![0.0f32; max_patches * channels];
    for out_y in 0..target_height {
        let in_y = linear_source_coordinate(out_y, target_height, source_height);
        let y0 = in_y.floor().clamp(0.0, (source_height - 1) as f32) as usize;
        let y1 = (y0 + 1).min(source_height - 1);
        let wy = in_y - y0 as f32;
        for out_x in 0..target_width {
            let in_x = linear_source_coordinate(out_x, target_width, source_width);
            let x0 = in_x.floor().clamp(0.0, (source_width - 1) as f32) as usize;
            let x1 = (x0 + 1).min(source_width - 1);
            let wx = in_x - x0 as f32;
            let token = out_y * target_width + out_x;
            for channel in 0..channels {
                let top = lerp(
                    base_pos[((y0 * source_width + x0) * channels) + channel],
                    base_pos[((y0 * source_width + x1) * channels) + channel],
                    wx,
                );
                let bottom = lerp(
                    base_pos[((y1 * source_width + x0) * channels) + channel],
                    base_pos[((y1 * source_width + x1) * channels) + channel],
                    wx,
                );
                output[token * channels + channel] = lerp(top, bottom, wy);
            }
        }
    }

    let valid = target_height * target_width;
    if valid > 0 && valid < max_patches {
        for token in valid..max_patches {
            let (dst, src) = output.split_at_mut(token * channels);
            let first = &dst[..channels];
            let current = &mut src[..channels];
            current.copy_from_slice(first);
        }
    }

    Ok(Array::from_shape_vec(IxDyn(&[1, max_patches, channels]), output)?)
}

fn linear_source_coordinate(output_index: usize, output_size: usize, input_size: usize) -> f32 {
    // align_corners=False mapping used by interpolate(..., mode="bilinear").
    let source = (output_index as f32 + 0.5) * input_size as f32 / output_size as f32 - 0.5;
    source.clamp(0.0, (input_size - 1) as f32)
}

fn lerp(a: f32, b: f32, weight: f32) -> f32 {
    a + (b - a) * weight
}

fn load_session(model_path: &Path) -> Result<Session> {
    println!("loading ONNX: {}", model_path.display());
    let start = Instant::now();
    let session = Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?
        .with_intra_threads(4)
        .map_err(ort_error)?
        .commit_from_file(model_path)
        .map_err(ort_error)
        .with_context(|| format!("failed to load ONNX model {}", model_path.display()))?;
    println!("loaded in {:.2?}", start.elapsed());
    Ok(session)
}

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow!("{error}")
}

fn report(name: &str, actual: &ArrayD<f32>, expected: &ArrayD<f32>) -> Result<()> {
    if actual.shape() != expected.shape() {
        bail!(
            "{name} output shape mismatch: actual {:?}, expected {:?}",
            actual.shape(),
            expected.shape()
        );
    }
    let actual = actual.as_slice().context("actual output is not contiguous")?;
    let expected = expected.as_slice().context("expected output is not contiguous")?;
    let max_abs_diff = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cosine = dot(actual, expected) / (l2(actual) * l2(expected)).max(f32::MIN_POSITIVE);
    println!("{name} max_abs_diff: {max_abs_diff:.9}");
    println!("{name} cosine_vs_pytorch: {cosine:.9}");
    Ok(())
}

fn read_f32_array(root: &Path, file: &TensorFile) -> Result<ArrayD<f32>> {
    array_from_vec(file, read_f32_vec(&root.join(&file.file))?)
}

fn read_i64_array(root: &Path, file: &TensorFile) -> Result<ArrayD<i64>> {
    array_from_vec(file, read_i64_vec(&root.join(&file.file))?)
}

fn read_i32_array(root: &Path, file: &TensorFile) -> Result<ArrayD<i32>> {
    array_from_vec(file, read_i32_vec(&root.join(&file.file))?)
}

fn array_from_vec<T>(file: &TensorFile, values: Vec<T>) -> Result<ArrayD<T>> {
    let expected_len = file.shape.iter().product::<usize>();
    if values.len() != expected_len {
        bail!(
            "{} element count mismatch: got {}, expected {} for shape {:?}",
            file.file.display(),
            values.len(),
            expected_len,
            file.shape
        );
    }
    Ok(Array::from_shape_vec(IxDyn(&file.shape), values)?)
}

fn read_f32_vec(path: &Path) -> Result<Vec<f32>> {
    read_pod_vec(path, 4, |chunk| {
        f32::from_le_bytes(chunk.try_into().expect("chunk size checked"))
    })
}

fn read_i64_vec(path: &Path) -> Result<Vec<i64>> {
    read_pod_vec(path, 8, |chunk| {
        i64::from_le_bytes(chunk.try_into().expect("chunk size checked"))
    })
}

fn read_i32_vec(path: &Path) -> Result<Vec<i32>> {
    read_pod_vec(path, 4, |chunk| {
        i32::from_le_bytes(chunk.try_into().expect("chunk size checked"))
    })
}

fn read_pod_vec<T>(path: &Path, width: usize, decode: impl Fn(&[u8]) -> T) -> Result<Vec<T>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() % width != 0 {
        bail!("{} has {} bytes, not divisible by {width}", path.display(), bytes.len());
    }
    Ok(bytes.chunks_exact(width).map(decode).collect())
}

fn write_f32_slice(path: &Path, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn dump_array_f32(path: &Path, array: &ArrayD<f32>) -> Result<()> {
    write_f32_slice(path, array.as_slice().context("f32 array is not contiguous")?)
}

fn dump_array_i32(path: &Path, array: &ArrayD<i32>) -> Result<()> {
    let mut bytes = Vec::with_capacity(array.len() * 4);
    for value in array.as_slice().context("i32 array is not contiguous")? {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn dump_array_i64(path: &Path, array: &ArrayD<i64>) -> Result<()> {
    let mut bytes = Vec::with_capacity(array.len() * 8);
    for value in array.as_slice().context("i64 array is not contiguous")? {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn l2(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn determine_max_patches_for_image(image_path: &Path) -> Result<usize> {
    let image = image::open(image_path)
        .with_context(|| format!("failed to open image {}", image_path.display()))?;
    let (width, height) = image.dimensions();
    let max_value = (width as usize / 16) * (height as usize / 16);
    let max_patches = if max_value > 784 {
        1024
    } else if max_value > 576 {
        784
    } else if max_value > 256 {
        576
    } else if max_value > 128 {
        256
    } else {
        128
    };
    Ok(max_patches)
}

fn absolutize(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}
