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
