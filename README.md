# Unravel Agent Runtime

Unravel Agent Runtime provides a Rust agent loop, OpenAI-compatible model
providers, and a Node.js bridge that runs the Rust loop with JavaScript tools.

## Install

For Rust, add the runtime and provider crates to your project:

```sh
cargo add unravel-agent-runtime unravel-agent-providers
```

For Node.js 20 or later, install the TypeScript-compatible package:

```sh
npm install @unravelai/unravel-agent-runtime
```

The npm package includes compiled Rust executables for Linux (x64 and arm64),
macOS (x64 and arm64), and Windows (x64). It does not require Cargo on the
machine where you install it. See the [Node package guide](packages/unravel-agent-runtime/README.md)
for an agent and tool example.

## Unreleased 0.2.0 Rust changes

These changes require a Git revision containing them; pushing this repository
does not publish new crates, npm binaries, or a release tag.

Rust tools can return `ToolOutput::with_observation(ToolObservation::new(content,
valid_for))` to provide untrusted sensor input for exactly the next model turn.
Ordinary tool text stays in history; the observation is separate canonical user
content after all tool results. Raw observations never enter saved messages or
events. The default encoded observation budget is 8 MiB per batch, independent
of the ordinary text budget. Retries reuse the same frame and check its earliest
monotonic expiry before every attempt; they never recapture automatically.

Adjacent `AgentLoop::turn` calls retain observations in private in-memory session
scratch until the next turn consumes them. Serialization, deserialization,
cloning, reconciliation, fresh `run` calls, errors and interrupted turns discard
that scratch. This preserves the serialized session schema but changes Rust
struct construction: use `Session::new` and the `ToolOutput` constructors instead
of struct literals. Fully specified `LoopConfig` literals must include
`max_observation_bytes`; `..LoopConfig::default()` remains supported.
Strict sequential dispatch remains the default; opt-in parallel tool groups
retain ordered results and the same aggregate budgets.

The provider exposes `validate_image_messages` and
`with_vision_support(Option<bool>)`. Image input capability uses the exact
Models.dev provider/model `modalities.input`: missing metadata is unknown, not
false, and names/attachment flags are not evidence. Known denial blocks image
HTTP requests, including when an optimistic explicit declaration conflicts with
discovered denial. Both canonical and raw Chat Completions image paths validate
PNG/JPEG framing, dimensions, pixels and bounded payloads before HTTP. Limits:
eight images, 4 MiB compressed bytes per image, 16 MiB total, 4096 pixels per axis,
8 megapixels per image and 16 megapixels total. JPEG entropy decoding is strict.
Remote image URLs are screened, not fetched: DNS resolution, redirects, remote
bytes and dimensions remain unverified.

Building the providers from source requires **CMake and a C compiler** for
bundled static libjpeg-turbo. NASM and system libjpeg are not required.
The Node callback wire format remains text/metadata-only; the observation API
above is a Rust API, not implicit JavaScript image forwarding.

Local Linux verification exercised the workspace suite, warnings-denied Clippy,
Rust 1.88 all-target checking, rustdoc/doctests, the executable sensor/tool
roundtrip, and a real localhost HTTP SDK run with PNG delivery, saved-frame
omission and fresh capture after restore. No hardware or live model was used.

```sh
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo +1.88.0 check --workspace --all-targets --locked
cargo run --locked -p unravel-agent-runtime --example tool_roundtrip
```


## Release

The [release workflow](.github/workflows/release.yml) publishes the Node
package to npm, both Rust crates to crates.io, and creates a GitHub
release for a `v<version>` tag. Configure `CARGO_REGISTRY_TOKEN` as a repository
Actions secret. On npm, configure trusted publishing for
`@unravelai/unravel-agent-runtime` from the GitHub repository
`unravelaidk/unravel-agent-runtime` and the workflow file `release.yml`.
Enable direct `npm publish` as an allowed action for the trusted publisher.
The release job requests an OIDC identity token and uses Node.js 24 with npm
CLI 11 to publish with provenance. npm trusted publishing must be configured
before pushing a release tag. For a new npm package, create the package
under your npm organization and configure the trusted publisher before the
first release.

To release a new version, update the workspace `version` in `Cargo.toml`, the
runtime dependency version in `crates/unravel-agent-providers/Cargo.toml`, and
the npm version in `packages/unravel-agent-runtime/package.json`. Regenerate
`Cargo.lock` and the npm lockfile, run the checks below, commit and push the
changes, then push a matching tag such as `v0.1.1`. The workflow checks that
the tag and package versions match before publishing.

```sh
cargo fmt --all -- --check
cargo test --workspace --locked
cd packages/unravel-agent-runtime
npm ci
npm run build
```

The provider crate depends on the runtime crate, so the workflow publishes
the runtime first and retries the provider publish while crates.io updates its
index. The npm and crates.io jobs run independently. They skip versions that
are already public when a failed workflow is rerun.

## License

This repository is licensed under AGPL-3.0-only. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for license and attribution details.
