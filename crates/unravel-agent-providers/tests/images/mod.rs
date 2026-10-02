//! Consumer-visible image request and pre-dispatch safety contracts.
use super::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, ImageFormat};
use unravel_agent_providers::{ChatEventSink, ChatRequest, ChatStreamEvent, ProviderError};

struct RawSink;

impl ChatEventSink for RawSink {
    fn on_event(&mut self, _: ChatStreamEvent<'_>) {}
}

fn raw_image_request(messages: &[Message]) -> ChatRequest {
    let messages = messages
        .iter()
        .map(|message| {
            let Message::User { content } = message else {
                panic!("image fixtures require user messages");
            };
            let parts: Vec<Value> = content
                .parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => json!({"type":"text", "text":text}),
                    ContentPart::Image { media_type, source } => {
                        let url = match source {
                            ImageSource::Url { url } => url.clone(),
                            ImageSource::Base64 { data } => {
                                format!(
                                    "data:{};base64,{data}",
                                    media_type.as_deref().unwrap_or("image/png")
                                )
                            }
                        };
                        json!({"type":"image_url", "image_url":{"url":url}})
                    }
                })
                .collect();
            json!({"role":"user", "content":parts})
        })
        .collect();
    ChatRequest {
        messages,
        ..Default::default()
    }
}

fn encoded_image(format: ImageFormat, width: u32, height: u32) -> Vec<u8> {
    let mut output = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(image::GrayImage::from_pixel(
        width,
        height,
        image::Luma([127]),
    ))
    .write_to(&mut output, format)
    .unwrap();
    output.into_inner()
}

fn encoded_rgb_jpeg(subsampling: turbojpeg::Subsamp, progressive: bool) -> Vec<u8> {
    // Odd dimensions cover partial MCU edges as well as chroma subsampling.
    let (width, height) = (17, 19);
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|index| (index * 37 % 256) as u8)
        .collect();
    let mut encoder = turbojpeg::Compressor::new().unwrap();
    encoder.set_subsamp(subsampling).unwrap();
    encoder.set_progressive(progressive).unwrap();
    encoder.set_quality(85).unwrap();
    encoder
        .compress_to_vec(turbojpeg::Image {
            pixels: pixels.as_slice(),
            width,
            height,
            pitch: width * 3,
            format: turbojpeg::PixelFormat::RGB,
        })
        .unwrap()
}

fn image_message(bytes: &[u8], media_type: &str) -> Message {
    Message::User {
        content: Content::from_parts(vec![
            ContentPart::text("inspect this frame"),
            ContentPart::Image {
                media_type: Some(media_type.into()),
                source: ImageSource::Base64 {
                    data: STANDARD.encode(bytes),
                },
            },
        ]),
    }
}

#[tokio::test]
async fn image_bytes_and_remote_urls_reach_both_completion_modes() {
    let server = MockServer::start(Box::new(|_, _, body, _| {
        let payload: Value = serde_json::from_str(body).unwrap();
        if payload["stream"] == true {
            MockResponse::sse(
                200,
                vec![
                    json!({"choices":[{"delta":{"content":"observed"}}]}).to_string(),
                    json!({"choices":[{"delta":{},"finish_reason":"stop"}]}).to_string(),
                    "[DONE]".into(),
                ],
            )
        } else {
            MockResponse::json(
                200,
                json!({
                    "choices":[{"message":{"content":"observed"},"finish_reason":"stop"}]
                }),
            )
        }
    }))
    .await;
    // No capability metadata: honestly unknown, not a denial.
    let provider = make_provider(&server, "test-model").with_vision_support(None);
    let fixtures = [
        (encoded_image(ImageFormat::Png, 2, 3), "image/png"),
        (encoded_image(ImageFormat::Jpeg, 2, 3), "image/jpeg"),
        (
            encoded_rgb_jpeg(turbojpeg::Subsamp::None, false),
            "image/jpeg",
        ),
        (
            encoded_rgb_jpeg(turbojpeg::Subsamp::Sub2x2, false),
            "image/jpeg",
        ),
        (
            encoded_rgb_jpeg(turbojpeg::Subsamp::Sub2x2, true),
            "image/jpeg",
        ),
    ];
    for (bytes, mime) in fixtures {
        let messages = vec![
            image_message(&bytes, mime),
            Message::User {
                content: Content::from_parts(vec![ContentPart::Image {
                    media_type: None,
                    source: ImageSource::Url {
                        url: "https://example.com/frame.jpg".into(),
                    },
                }]),
            },
        ];
        for stream in [false, true] {
            let request = ModelRequest::new("image-request", messages.clone());
            let response = if stream {
                provider
                    .stream(request, &mut CollectingSink::default())
                    .await
                    .unwrap()
            } else {
                provider.complete(request).await.unwrap()
            };
            assert_eq!(response.content, "observed");
            let captured = server.captured_requests();
            let payload: Value = serde_json::from_str(&captured.last().unwrap().body).unwrap();
            let url = payload["messages"][0]["content"][1]["image_url"]["url"]
                .as_str()
                .unwrap();
            let data = url.strip_prefix(&format!("data:{mime};base64,")).unwrap();
            assert_eq!(STANDARD.decode(data).unwrap(), bytes);
            assert_eq!(
                payload["messages"][1]["content"][0]["image_url"]["url"],
                "https://example.com/frame.jpg"
            );
        }
    }
    assert_eq!(server.completions_call_count(), 10);
}

async fn assert_images_refused(
    provider: &OpenAiCompatProvider,
    messages: Vec<Message>,
    case: &str,
) {
    for stream in [false, true] {
        let request = ModelRequest::new("invalid-image", messages.clone());
        let error = if stream {
            provider
                .stream(request, &mut CollectingSink::default())
                .await
                .unwrap_err()
        } else {
            provider.complete(request).await.unwrap_err()
        };
        assert!(
            matches!(&error, unravel_agent_runtime::Error::ModelTyped(error)
                if matches!(error.retryability(), unravel_agent_runtime::Retryability::Permanent)),
            "case={case}, stream={stream}: expected permanent pre-dispatch validation error, got {error}"
        );
        let request = raw_image_request(&messages);
        let error = if stream {
            provider
                .stream_chat(request, &mut RawSink)
                .await
                .unwrap_err()
        } else {
            provider.complete_chat(request).await.unwrap_err()
        };
        assert!(
            matches!(error, ProviderError::Invalid { .. }),
            "case={case}, stream={stream}: raw chat must reject images before dispatch, got {error}"
        );
    }
}

#[tokio::test]
async fn invalid_and_bounded_images_never_dispatch_http() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::error(500, "image validation must run before HTTP")
    }))
    .await;
    let provider = make_provider(&server, "test-model");
    let png = encoded_image(ImageFormat::Png, 2, 3);
    let mut truncated = png.clone();
    truncated.truncate(truncated.len() - 1);
    assert_images_refused(
        &provider,
        vec![image_message(&truncated, "image/png")],
        "end marker",
    )
    .await;
    let jpeg = encoded_image(ImageFormat::Jpeg, 2, 3);
    assert_images_refused(
        &provider,
        vec![image_message(&jpeg[..jpeg.len() - 2], "image/jpeg")],
        "end marker",
    )
    .await;
    // Headers and end marker alone are not a valid JPEG: strict entropy
    // decoding must refuse a scan whose pixel bytes were removed.
    let scan = jpeg
        .windows(2)
        .position(|pair| pair == [0xff, 0xda])
        .unwrap();
    let scan_length = u16::from_be_bytes(jpeg[scan + 2..scan + 4].try_into().unwrap()) as usize;
    let mut missing_pixels = jpeg[..scan + 2 + scan_length].to_vec();
    missing_pixels.extend_from_slice(&[0xff, 0xd9]);
    assert_images_refused(
        &provider,
        vec![image_message(&missing_pixels, "image/jpeg")],
        "invalid",
    )
    .await;
    let rgb_jpeg = encoded_rgb_jpeg(turbojpeg::Subsamp::Sub2x2, false);
    let scan = rgb_jpeg
        .windows(2)
        .position(|pair| pair == [0xff, 0xda])
        .unwrap();
    let scan_length = u16::from_be_bytes(rgb_jpeg[scan + 2..scan + 4].try_into().unwrap()) as usize;
    let entropy_start = scan + 2 + scan_length;
    let mut partial_entropy =
        rgb_jpeg[..entropy_start + (rgb_jpeg.len() - 2 - entropy_start) / 2].to_vec();
    partial_entropy.extend_from_slice(&[0xff, 0xd9]);
    assert_images_refused(
        &provider,
        vec![image_message(&partial_entropy, "image/jpeg")],
        "partial JPEG entropy",
    )
    .await;
    let mut jpeg_bomb = rgb_jpeg.clone();
    let frame = jpeg_bomb
        .windows(2)
        .position(|pair| pair == [0xff, 0xc0])
        .unwrap();
    jpeg_bomb[frame + 5..frame + 7].copy_from_slice(&4096u16.to_be_bytes());
    jpeg_bomb[frame + 7..frame + 9].copy_from_slice(&4096u16.to_be_bytes());
    assert_images_refused(
        &provider,
        vec![image_message(&jpeg_bomb, "image/jpeg")],
        "JPEG dimension bomb",
    )
    .await;
    assert_images_refused(
        &provider,
        vec![image_message(&png, "image/jpeg")],
        "media type",
    )
    .await;
    assert_images_refused(
        &provider,
        vec![image_message(&png, "image/gif")],
        "media type",
    )
    .await;
    let mut corrupt = png.clone();
    corrupt[29] ^= 1;
    assert_images_refused(
        &provider,
        vec![image_message(&corrupt, "image/png")],
        "checksum",
    )
    .await;
    // A complete, checksum-correct PNG with invalid compressed pixels must
    // also fail; checksum and header recognition are not sufficient.
    let mut invalid_pixels = png.clone();
    let mut offset = 8;
    loop {
        let length =
            u32::from_be_bytes(invalid_pixels[offset..offset + 4].try_into().unwrap()) as usize;
        let end = offset + length + 12;
        if &invalid_pixels[offset + 4..offset + 8] == b"IDAT" {
            invalid_pixels[offset + 8] = 0;
            let crc = crc32fast::hash(&invalid_pixels[offset + 4..end - 4]);
            invalid_pixels[end - 4..end].copy_from_slice(&crc.to_be_bytes());
            break;
        }
        offset = end;
    }
    assert_images_refused(
        &provider,
        vec![image_message(&invalid_pixels, "image/png")],
        "pixel data",
    )
    .await;
    let mut bad_base64 = image_message(&png, "image/png");
    if let Message::User { content } = &mut bad_base64 {
        if let ContentPart::Image {
            source: ImageSource::Base64 { data },
            ..
        } = &mut content.parts[1]
        {
            *data = "not-base64!".into();
        }
    }
    assert_images_refused(&provider, vec![bad_base64], "base64").await;
    // The cap is checked before base64 decoder allocation.
    let too_large = vec![0u8; 4 * 1024 * 1024 + 1];
    assert_images_refused(
        &provider,
        vec![image_message(&too_large, "image/png")],
        "4 MiB",
    )
    .await;
    assert_images_refused(
        &provider,
        vec![image_message(&png, "image/png"); 9],
        "eight",
    )
    .await;
    // Valid IHDR and CRC, but bomb dimensions: no pixel allocation/decode.
    let mut bomb = png.clone();
    bomb[16..20].copy_from_slice(&4096u32.to_be_bytes());
    bomb[20..24].copy_from_slice(&4096u32.to_be_bytes());
    let crc = crc32fast::hash(&bomb[12..29]);
    bomb[29..33].copy_from_slice(&crc.to_be_bytes());
    assert_images_refused(
        &provider,
        vec![image_message(&bomb, "image/png")],
        "8 megapixel",
    )
    .await;
    bomb[16..20].copy_from_slice(&4097u32.to_be_bytes());
    let crc = crc32fast::hash(&bomb[12..29]);
    bomb[29..33].copy_from_slice(&crc.to_be_bytes());
    assert_images_refused(
        &provider,
        vec![image_message(&bomb, "image/png")],
        "decoder limits",
    )
    .await;
    // Small compressed payloads can exceed aggregate pixel budgets.
    let large_pixels = encoded_image(ImageFormat::Png, 2048, 4096);
    assert_images_refused(
        &provider,
        vec![image_message(&large_pixels, "image/png"); 3],
        "aggregate image limit",
    )
    .await;
    // Valid ancillary padding tests compressed-byte limits independently.
    let mut padded = png[..png.len() - 12].to_vec();
    let length = 3 * 1024 * 1024;
    padded.extend_from_slice(&(length as u32).to_be_bytes());
    let start = padded.len();
    padded.extend_from_slice(b"npAD");
    padded.resize(padded.len() + length, 0);
    let crc = crc32fast::hash(&padded[start..]);
    padded.extend_from_slice(&crc.to_be_bytes());
    padded.extend_from_slice(&png[png.len() - 12..]);
    assert_images_refused(
        &provider,
        vec![image_message(&padded, "image/png"); 6],
        "aggregate image byte limit",
    )
    .await;
    assert!(server.captured_requests().is_empty());
}

#[tokio::test]
async fn unsafe_remote_image_sources_never_dispatch_http() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::error(500, "unexpected HTTP")
    }))
    .await;
    let provider = make_provider(&server, "test-model");
    for source in [
        "file:///tmp/frame.png",
        "data:image/png;base64,aGVsbG8=",
        "https://user:secret@example.com/frame",
        "http://localhost/frame",
        "http://127.0.0.1/frame",
        "http://2130706433/frame",
        "http://10.0.0.1/frame",
        "http://169.254.169.254/frame",
        "http://[::1]/frame",
        "http://[::ffff:127.0.0.1]/frame",
        "https://camera.local/frame",
        "https://example.com/frame#fragment",
    ] {
        let message = Message::User {
            content: Content::from_parts(vec![ContentPart::Image {
                media_type: None,
                source: ImageSource::Url { url: source.into() },
            }]),
        };
        assert_images_refused(&provider, vec![message], "image URL").await;
    }
    let message = Message::User {
        content: Content::from_parts(vec![ContentPart::Image {
            media_type: None,
            source: ImageSource::Url {
                url: format!("https://example.com/{}", "x".repeat(2048)),
            },
        }]),
    };
    assert_images_refused(&provider, vec![message], "oversized").await;
    assert!(server.captured_requests().is_empty());
}

#[tokio::test]
async fn explicit_no_vision_refuses_valid_bytes_but_allows_text() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::json(
            200,
            json!({
                "choices":[{"message":{"content":"text accepted"},"finish_reason":"stop"}]
            }),
        )
    }))
    .await;
    let provider = make_provider(&server, "text-model").with_vision_support(Some(false));
    assert_images_refused(
        &provider,
        vec![image_message(
            &encoded_image(ImageFormat::Png, 2, 3),
            "image/png",
        )],
        "does not support image",
    )
    .await;
    assert!(server.captured_requests().is_empty());
    let response = provider
        .complete(ModelRequest::new("text", vec![Message::user_text("hello")]))
        .await
        .unwrap();
    assert_eq!(response.content, "text accepted");
    assert_eq!(server.completions_call_count(), 1);
}

#[tokio::test]
async fn loaded_discovery_vision_denial_blocks_without_network_preflight_and_expires() {
    let server = MockServer::start(Box::new(|_, path, _, _| {
        if path == "/v1/models" {
            MockResponse::json(200, json!({"data":[{"id":"text-model"},{"id":"unknown-model"}]}))
        } else if path == "/catalog.json" {
            MockResponse::json(200, json!({"mock":{
                "name":"Mock", "models":{
                    "text-model":{"name":"Text", "modalities":{"input":["text"],"output":["text"]}},
                    "unknown-model":{"name":"Unknown"}
                }
            }}))
        } else {
            MockResponse::json(200, json!({"choices":[{"message":{"content":"accepted"},"finish_reason":"stop"}]}))
        }
    })).await;
    let clock = Arc::new(TestClock::new(Instant::now()));
    let discovery = Arc::new(Discovery::with_clock(
        Duration::from_secs(30),
        clock.clone(),
    ));
    let provider = make_provider(&server, "text-model").with_discovery(discovery.clone());
    let options = DiscoveryOptions {
        models_dev_url_override: Some(format!("http://{}/catalog.json", server.addr)),
        ..Default::default()
    };
    let models = discovery.discover(provider.spec(), &options).await.unwrap();
    assert_eq!(
        models
            .iter()
            .find(|m| m.model_id == "text-model")
            .unwrap()
            .modalities
            .image_input,
        Some(false)
    );
    assert_eq!(
        models
            .iter()
            .find(|m| m.model_id == "unknown-model")
            .unwrap()
            .modalities
            .image_input,
        None
    );
    let requests_after_discovery = server.captured_requests().len();
    let messages = vec![image_message(
        &encoded_image(ImageFormat::Png, 2, 3),
        "image/png",
    )];
    assert_images_refused(&provider, messages.clone(), "does not support image").await;
    let claimed_support = make_provider(&server, "text-model")
        .with_discovery(discovery.clone())
        .with_vision_support(Some(true));
    assert_images_refused(&claimed_support, messages.clone(), "does not support image").await;
    assert_eq!(server.captured_requests().len(), requests_after_discovery);
    let unknown = make_provider(&server, "unknown-model").with_discovery(discovery);
    assert_eq!(
        unknown
            .complete(ModelRequest::new("unknown", messages.clone()))
            .await
            .unwrap()
            .content,
        "accepted"
    );
    clock.advance(Duration::from_secs(30));
    assert_eq!(
        provider
            .complete(ModelRequest::new("expired", messages))
            .await
            .unwrap()
            .content,
        "accepted"
    );
    assert_eq!(
        server.models_call_count(),
        1,
        "completion must not refresh discovery"
    );
    assert_eq!(server.completions_call_count(), 2);
}

#[tokio::test]
async fn raw_chat_validates_images_without_normalizing_application_history() {
    let server = MockServer::start(Box::new(|_, _, body, _| {
        let payload: Value = serde_json::from_str(body).unwrap();
        if payload["stream"] == true {
            MockResponse::sse(
                200,
                vec![
                    json!({"choices":[{"delta":{},"finish_reason":"stop"}]}).to_string(),
                    "[DONE]".into(),
                ],
            )
        } else {
            MockResponse::json(
                200,
                json!({"choices":[{"message":{"content":""},"finish_reason":"stop"}]}),
            )
        }
    }))
    .await;
    let provider = make_provider(&server, "vision-model").with_vision_support(Some(true));
    for (bytes, mime) in [
        (encoded_image(ImageFormat::Png, 2, 3), "image/png"),
        (
            encoded_rgb_jpeg(turbojpeg::Subsamp::Sub2x2, true),
            "image/jpeg",
        ),
    ] {
        let data_url = format!("data:{mime};base64,{}", STANDARD.encode(&bytes));
        for stream in [false, true] {
            let messages = vec![
                json!({"role":"user","content":[
                    {"type":"image_url","image_url":{"url":data_url,"detail":"low"}},
                    {"type":"image_url","image_url":"https://example.com/frame.jpg"}
                ]}),
                json!({"role":"assistant","content":null,"tool_calls":[
                    {"id":"application-id","function":{"name":"inspect","arguments":"{\"broken\":"}}
                ]}),
            ];
            let request = ChatRequest {
                messages: messages.clone(),
                ..Default::default()
            };
            let response = if stream {
                provider.stream_chat(request, &mut RawSink).await.unwrap()
            } else {
                provider.complete_chat(request).await.unwrap()
            };
            assert_eq!(response.finish_reason, Some(FinishReason::Stop));
            let captured = server.captured_requests();
            let payload: Value = serde_json::from_str(&captured.last().unwrap().body).unwrap();
            assert_eq!(payload["messages"], json!(messages));
        }
    }
}

#[tokio::test]
async fn raw_chat_rejects_malformed_and_alternate_image_paths_without_source_leakage() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::error(500, "unexpected HTTP")
    }))
    .await;
    let provider = make_provider(&server, "vision-model");
    let secret = "private-source-token";
    for part in [
        json!({"type":"image_url","image_url":{}}),
        json!({"type":"image_url","image_url":{"url":17}}),
        json!({"type":"image_url","image_url":{"url":format!("data:image/png,{secret}")}}),
        json!({"type":"image_url","image_url":{"url":format!("data:image/gif;base64,{secret}")}}),
        json!({"type":"image_url","image_url":{"url":format!("https://user:{secret}@example.com/frame")}}),
        json!({"type":"image","source":{"data":secret}}),
        json!({"type":"input_image","image_url":format!("http://localhost/{secret}")}),
    ] {
        // A role change must not bypass the raw boundary's image policy.
        for role in ["user", "assistant", "system", "tool"] {
            for stream in [false, true] {
                let request = ChatRequest {
                    messages: vec![json!({"role":role,"content":[part]})],
                    ..Default::default()
                };
                let error = if stream {
                    provider
                        .stream_chat(request, &mut RawSink)
                        .await
                        .unwrap_err()
                } else {
                    provider.complete_chat(request).await.unwrap_err()
                };
                assert!(matches!(error, ProviderError::Invalid { .. }));
                assert!(!format!("{error:?} {error}").contains(secret));
            }
        }
    }
    assert!(server.captured_requests().is_empty());
}

#[tokio::test]
async fn upstream_image_echo_is_redacted_from_both_public_boundaries() {
    let server = MockServer::start(Box::new(|_, _, body, _| {
        let payload: Value = serde_json::from_str(body).unwrap();
        let parts = payload["messages"][0]["content"].as_array().unwrap();
        let data = parts[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .split_once(";base64,")
            .unwrap()
            .1;
        let url = parts[2]["image_url"]["url"].as_str().unwrap();
        let secret = reqwest::Url::parse(url)
            .unwrap()
            .query_pairs()
            .find(|(key, _)| key == "key")
            .unwrap()
            .1
            .into_owned();
        MockResponse::error(
            400,
            &format!("rejected {data} and {url}, source token {secret}"),
        )
    }))
    .await;
    let provider = make_provider(&server, "vision-model");
    let bytes = encoded_image(ImageFormat::Png, 2, 3);
    let mut message = image_message(&bytes, "image/png");
    if let Message::User { content } = &mut message {
        content.parts.push(ContentPart::Image {
            media_type: None,
            source: ImageSource::Url {
                url: "https://example.com/private-frame-token?key=source-secret".into(),
            },
        });
    }
    for stream in [false, true] {
        let request = ModelRequest::new("echo", vec![message.clone()]);
        let error = if stream {
            provider
                .stream(request, &mut CollectingSink::default())
                .await
                .unwrap_err()
        } else {
            provider.complete(request).await.unwrap_err()
        };
        let raw = raw_image_request(&[message.clone()]);
        let raw_error = if stream {
            provider.stream_chat(raw, &mut RawSink).await.unwrap_err()
        } else {
            provider.complete_chat(raw).await.unwrap_err()
        };
        for detail in [
            format!("{error:?} {error}"),
            format!("{raw_error:?} {raw_error}"),
        ] {
            assert!(!detail.contains(&STANDARD.encode(&bytes)));
            assert!(!detail.contains("private-frame-token"));
            assert!(!detail.contains("source-secret"));
        }
    }
    assert_eq!(server.completions_call_count(), 4);
}

#[tokio::test]
async fn single_frame_complete_container_policy_rejects_animation_and_trailing_data() {
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::error(500, "unexpected HTTP")
    }))
    .await;
    let provider = make_provider(&server, "vision-model");
    let png = encoded_image(ImageFormat::Png, 2, 3);
    let mut animated = png[..33].to_vec();
    animated.extend_from_slice(&8u32.to_be_bytes());
    let start = animated.len();
    animated.extend_from_slice(b"acTL");
    animated.extend_from_slice(&2u32.to_be_bytes());
    animated.extend_from_slice(&0u32.to_be_bytes());
    let crc = crc32fast::hash(&animated[start..]);
    animated.extend_from_slice(&crc.to_be_bytes());
    animated.extend_from_slice(&png[33..]);
    assert_images_refused(
        &provider,
        vec![image_message(&animated, "image/png")],
        "animation",
    )
    .await;
    let mut trailing = png.clone();
    trailing.extend_from_slice(b"private-trailing-frame");
    assert_images_refused(
        &provider,
        vec![image_message(&trailing, "image/png")],
        "trailing bytes",
    )
    .await;
    assert_images_refused(&provider, vec![image_message(&[], "image/png")], "empty").await;
    assert!(server.captured_requests().is_empty());
}

#[tokio::test]
async fn unspecified_mime_preserves_png_default_and_rejects_jpeg_mismatch() {
    let png = encoded_image(ImageFormat::Png, 2, 3);
    let jpeg = encoded_image(ImageFormat::Jpeg, 2, 3);
    let mut messages = [
        image_message(&png, "image/png"),
        image_message(&jpeg, "image/jpeg"),
    ];
    for message in &mut messages {
        if let Message::User { content } = message {
            if let ContentPart::Image { media_type, .. } = &mut content.parts[1] {
                *media_type = None;
            }
        }
    }
    let [png_message, jpeg_message] = messages;
    unravel_agent_providers::validate_image_messages(std::slice::from_ref(&png_message)).unwrap();
    assert!(matches!(
        unravel_agent_providers::validate_image_messages(std::slice::from_ref(&jpeg_message)),
        Err(ProviderError::Invalid { .. })
    ));
    let server = MockServer::start(Box::new(|_, _, _, _| {
        MockResponse::json(
            200,
            json!({"choices":[{"message":{"content":""},"finish_reason":"stop"}]}),
        )
    }))
    .await;
    let provider = make_provider(&server, "vision-model");
    provider
        .complete(ModelRequest::new("default-mime", vec![png_message]))
        .await
        .unwrap();
    let captured = server.captured_requests();
    let payload: Value = serde_json::from_str(&captured[0].body).unwrap();
    assert_eq!(
        payload["messages"][0]["content"][1]["image_url"]["url"],
        format!("data:image/png;base64,{}", STANDARD.encode(png))
    );
    assert_images_refused(&provider, vec![jpeg_message], "default PNG mismatch").await;
    assert_eq!(server.completions_call_count(), 1);
}
