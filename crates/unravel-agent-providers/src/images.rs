//! Bounded validation at the canonical message boundary; never fetches URLs.

use crate::{ProviderError, ProviderResult};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{ImageDecoder, ImageFormat, ImageReader, Limits};
use std::io::Cursor;
use std::net::{IpAddr, Ipv4Addr};
use unravel_agent_runtime::{ContentPart, ImageSource, Message};

const MAX_IMAGES: usize = 8;
const MAX_IMAGE_BYTES: usize = 4 * 1024 * 1024;
const MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;
const MAX_BASE64_BYTES: usize = MAX_IMAGE_BYTES.div_ceil(3) * 4;
const MAX_DIMENSION: u32 = 4096;
const MAX_PIXELS: u64 = 8 * 1024 * 1024;
const MAX_TOTAL_PIXELS: u64 = 16 * 1024 * 1024;
const MAX_DECODER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_URL_BYTES: usize = 2048;

/// Validate canonical image parts before serialization or HTTP dispatch.
///
/// Accepts fully decodable, single-frame PNG/JPEG base64 with matching MIME
/// (`None` retains the canonical PNG default). Limits: eight images, 4 MiB
/// compressed bytes each, 16 MiB total, 4096 pixels per axis, 8 megapixels each,
/// 16 megapixels total. Decoder allocations are constrained before pixel decode.
/// Remote HTTP(S) URLs are bounded and screened for credentials and local hosts;
/// their bytes, dimensions, DNS resolution and redirects are **not** validated.
/// Errors never include source bytes or secret-bearing URLs.
pub fn validate_image_messages(messages: &[Message]) -> ProviderResult<()> {
    let mut validation = ImageValidation::default();
    for message in messages {
        if let Message::User { content } = message {
            for part in &content.parts {
                if let ContentPart::Image { media_type, source } = part {
                    let input = match source {
                        ImageSource::Url { url } => ImageInput::Url(url),
                        ImageSource::Base64 { data } => ImageInput::Base64(data),
                    };
                    validation.validate(media_type.as_deref(), input)?;
                }
            }
        }
    }
    Ok(())
}

// Inspect only image parts, leaving application-owned history and arguments
// untouched. Canonical messages and raw chat share the same byte/pixel budgets.
pub(crate) fn validate_chat_images(messages: &[serde_json::Value]) -> ProviderResult<()> {
    let mut validation = ImageValidation::default();
    for part in chat_content_parts(messages) {
        match part.get("type").and_then(serde_json::Value::as_str) {
            Some("image_url") => {
                let url = chat_image_url(part)
                    .ok_or_else(|| invalid("image_url requires a string URL"))?;
                if let Some(data_url) = url.strip_prefix("data:") {
                    let (mime, data) = data_url
                        .split_once(";base64,")
                        .ok_or_else(|| invalid("image data URL requires canonical base64"))?;
                    validation.validate(Some(mime), ImageInput::Base64(data))?;
                } else {
                    validation.validate(None, ImageInput::Url(url))?;
                }
            }
            Some("image" | "input_image") => {
                return Err(invalid("chat images must use image_url content parts"));
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn chat_has_images(messages: &[serde_json::Value]) -> bool {
    chat_content_parts(messages).any(|part| {
        matches!(
            part.get("type").and_then(serde_json::Value::as_str),
            Some("image_url" | "image" | "input_image")
        )
    })
}

// An upstream may echo an image URL or its encoded bytes in an error body.
// Never let that turn a rejected request into raw frame/source disclosure.
pub(crate) fn redact_chat_images(mut body: String, messages: &[serde_json::Value]) -> String {
    for part in chat_content_parts(messages) {
        if let Some(url) = chat_image_url(part) {
            body = body.replace(url, "[REDACTED IMAGE]");
            if let Some((_, data)) = url
                .strip_prefix("data:")
                .and_then(|source| source.split_once(";base64,"))
            {
                if !data.is_empty() {
                    body = body.replace(data, "[REDACTED IMAGE]");
                }
            } else if let Ok(url) = reqwest::Url::parse(url) {
                for (_, value) in url.query_pairs() {
                    if !value.is_empty() {
                        body = body.replace(value.as_ref(), "[REDACTED IMAGE]");
                    }
                }
            }
        }
    }
    body
}

fn chat_content_parts(messages: &[serde_json::Value]) -> impl Iterator<Item = &serde_json::Value> {
    messages
        .iter()
        .filter_map(|message| message.get("content").and_then(serde_json::Value::as_array))
        .flat_map(|parts| parts.iter())
}

fn chat_image_url(part: &serde_json::Value) -> Option<&str> {
    if part.get("type").and_then(serde_json::Value::as_str) != Some("image_url") {
        return None;
    }
    let image = part.get("image_url")?;
    image.as_str().or_else(|| image.get("url")?.as_str())
}

#[derive(Default)]
struct ImageValidation {
    count: usize,
    total_bytes: usize,
    total_pixels: u64,
}

enum ImageInput<'a> {
    Url(&'a str),
    Base64(&'a str),
}

impl ImageValidation {
    fn validate(&mut self, media_type: Option<&str>, source: ImageInput<'_>) -> ProviderResult<()> {
        self.count += 1;
        if self.count > MAX_IMAGES {
            return Err(invalid("request exceeds eight image parts"));
        }
        if media_type.is_some_and(|mime| {
            mime.len() > 64
                || !mime.starts_with("image/")
                || mime.len() <= 6
                || !mime[6..]
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'+' | b'-'))
        }) {
            return Err(invalid(
                "image media type must be a bounded image MIME token",
            ));
        }
        match source {
            ImageInput::Url(url) => validate_remote_url(url)?,
            ImageInput::Base64(data) => {
                let format = match media_type {
                    None | Some("image/png") => ImageFormat::Png,
                    Some("image/jpeg") => ImageFormat::Jpeg,
                    _ => {
                        return Err(invalid(
                            "base64 image media type must be image/png or image/jpeg",
                        ))
                    }
                };
                if data.is_empty() || data.len() > MAX_BASE64_BYTES {
                    return Err(invalid(
                        "image base64 payload exceeds the 4 MiB byte limit or is empty",
                    ));
                }
                let bytes = STANDARD
                    .decode(data)
                    .map_err(|_| invalid("image payload is not canonical base64"))?;
                if bytes.len() > MAX_IMAGE_BYTES {
                    return Err(invalid("image payload exceeds the 4 MiB byte limit"));
                }
                self.total_bytes += bytes.len();
                if self.total_bytes > MAX_TOTAL_BYTES {
                    return Err(invalid(
                        "request exceeds the 16 MiB aggregate image byte limit",
                    ));
                }
                validate_container(&bytes, format)?;
                if format == ImageFormat::Jpeg {
                    validate_jpeg(&bytes, &mut self.total_pixels)?;
                    return Ok(());
                }
                let mut limits = Limits::default();
                limits.max_image_width = Some(MAX_DIMENSION);
                limits.max_image_height = Some(MAX_DIMENSION);
                limits.max_alloc = Some(MAX_DECODER_BYTES);
                let mut reader = ImageReader::with_format(Cursor::new(&bytes), format);
                reader.limits(limits.clone());
                let mut decoder = reader.into_decoder().map_err(|_| {
                    invalid("image format/header is invalid or exceeds decoder limits")
                })?;
                let (width, height) = decoder.dimensions();
                count_pixels(width, height, &mut self.total_pixels)?;
                let output_bytes = decoder.total_bytes();
                limits
                    .reserve(output_bytes)
                    .map_err(|_| invalid("image exceeds decoder allocation limit"))?;
                decoder
                    .set_limits(limits)
                    .map_err(|_| invalid("image exceeds decoder limits"))?;
                let mut pixels = vec![0; output_bytes as usize];
                decoder
                    .read_image(&mut pixels)
                    .map_err(|_| invalid("image pixel data is invalid or truncated"))?;
            }
        }
        Ok(())
    }
}

fn invalid(reason: &'static str) -> ProviderError {
    ProviderError::invalid(reason)
}

fn count_pixels(width: u32, height: u32, total: &mut u64) -> ProviderResult<()> {
    let pixels = u64::from(width) * u64::from(height);
    if width == 0
        || height == 0
        || width > MAX_DIMENSION
        || height > MAX_DIMENSION
        || pixels > MAX_PIXELS
    {
        return Err(invalid(
            "image dimensions exceed the nonzero 8 megapixel limit",
        ));
    }
    *total += pixels;
    if *total > MAX_TOTAL_PIXELS {
        return Err(invalid(
            "request exceeds the 16 megapixel aggregate image limit",
        ));
    }
    Ok(())
}

fn validate_jpeg(bytes: &[u8], total_pixels: &mut u64) -> ProviderResult<()> {
    // Decoders that pad missing entropy with zero bits can claim success even
    // in strict mode. TurboJPEG returns corruption warnings as errors.
    let mut decoder = turbojpeg::Decompressor::new()
        .map_err(|_| invalid("JPEG decoder initialization failed"))?;
    decoder
        .set_scan_limit(100)
        .map_err(|_| invalid("JPEG decoder scan limit could not be set"))?;
    let header = decoder
        .read_header(bytes)
        .map_err(|_| invalid("JPEG format/header is invalid"))?;
    let width = u32::try_from(header.width).map_err(|_| invalid("JPEG width exceeds limit"))?;
    let height = u32::try_from(header.height).map_err(|_| invalid("JPEG height exceeds limit"))?;
    count_pixels(width, height, total_pixels)?;
    let format = match header.colorspace {
        turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK => turbojpeg::PixelFormat::CMYK,
        _ => turbojpeg::PixelFormat::RGB,
    };
    let pitch = header.width * format.size();
    let length = pitch * header.height;
    if length as u64 > MAX_DECODER_BYTES {
        return Err(invalid("JPEG exceeds output allocation limit"));
    }
    // Output is byte-capped; coefficient scratch is dimension-bounded.
    let mut pixels = vec![0; length];
    decoder
        .decompress(
            bytes,
            turbojpeg::Image {
                pixels: pixels.as_mut_slice(),
                width: header.width,
                height: header.height,
                pitch,
                format,
            },
        )
        .map_err(|_| invalid("JPEG pixel data is invalid or truncated"))?;
    Ok(())
}

// Validate complete framing, not merely a signature or truncated header. PNG
// decoding stops after the first frame, so inspect every chunk and its CRC too.
fn validate_container(bytes: &[u8], format: ImageFormat) -> ProviderResult<()> {
    if image::guess_format(bytes).ok() != Some(format) {
        return Err(invalid(
            "image bytes do not match declared PNG/JPEG media type",
        ));
    }
    if format == ImageFormat::Jpeg {
        if !bytes.ends_with(&[0xff, 0xd9]) {
            return Err(invalid("JPEG image is missing its end marker"));
        }
        return Ok(());
    }
    let mut offset = 8;
    let mut first = true;
    while offset + 12 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let end = offset
            .checked_add(length)
            .and_then(|n| n.checked_add(12))
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| invalid("PNG chunk is truncated"))?;
        let kind = &bytes[offset + 4..offset + 8];
        if first && (kind != b"IHDR" || length != 13) {
            return Err(invalid("PNG image must begin with a complete IHDR"));
        }
        first = false;
        let crc = u32::from_be_bytes(bytes[end - 4..end].try_into().unwrap());
        if crc32fast::hash(&bytes[offset + 4..end - 4]) != crc {
            return Err(invalid("PNG image has an invalid chunk checksum"));
        }
        if kind == b"acTL" {
            return Err(invalid("animated PNG images are not supported"));
        }
        if kind == b"IEND" {
            return if length == 0 && end == bytes.len() {
                Ok(())
            } else {
                Err(invalid("PNG end marker or trailing data is invalid"))
            };
        }
        offset = end;
    }
    Err(invalid("PNG image is missing its complete end marker"))
}

fn validate_remote_url(source: &str) -> ProviderResult<()> {
    if source.is_empty()
        || source.len() > MAX_URL_BYTES
        || source.chars().any(|c| c.is_whitespace() || c.is_control())
    {
        return Err(invalid(
            "image URL is empty, oversized, or contains whitespace",
        ));
    }
    let url = reqwest::Url::parse(source).map_err(|_| invalid("image URL is invalid"))?;
    if !matches!(url.scheme(), "https" | "http")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "image URL must be HTTP(S), without credentials or fragment",
        ));
    }
    let host = url
        .host_str()
        .ok_or_else(|| invalid("image URL requires a public host"))?;
    let host = host.trim_end_matches('.');
    let literal = host.trim_start_matches('[').trim_end_matches(']');
    let safe = match literal.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => public_v4(ip),
        Ok(IpAddr::V6(ip)) => ip.to_ipv4_mapped().map(public_v4).unwrap_or_else(|| {
            // Global unicast only, excluding documentation and transition ranges.
            let segments = ip.segments();
            segments[0] & 0xe000 == 0x2000
                && !(segments[0] == 0x2001 && (segments[1] == 0 || segments[1] == 0xdb8))
                && segments[0] != 0x2002
                && segments[0] != 0x3fff
        }),
        Err(_) => {
            host.contains('.')
                && host != "localhost"
                && ![".localhost", ".local", ".internal", ".lan", ".home"]
                    .iter()
                    .any(|suffix| host.ends_with(suffix))
        }
    };
    if !safe {
        return Err(invalid(
            "image URL local/private/reserved hosts are not allowed",
        ));
    }
    Ok(())
}

fn public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(matches!(a, 0 | 10 | 127 | 224..=255)
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 168)
        || (a == 100 && (64..=127).contains(&b))
        || (a == 192 && b == 0 && matches!(c, 0 | 2))
        || (a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100)))
        || (a == 203 && b == 0 && c == 113))
}
